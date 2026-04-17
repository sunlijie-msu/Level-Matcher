"""
Dataset Parser for Nuclear Level Matcher
=========================================

# High-level Structure and Workflow Explanation:
======================================

Workflow Diagram:
[Start] -> [Raw Evaluated Nuclear Structure Data File Strings] -> [Parser Engine] -> [Structured Output]
                                     |
                                     v
                             [Inference Rules]
                            /        |        \
      [Uncertainties (Precision)] [Spin-Parity Lists]  [Gamma Decays]
                            \        |        /
                             v       v       v
                     [Standardized JSON Schema (Levels/Gammas)]

Technical Steps:
1. Parse raw strings from Evaluated Nuclear Structure Data File (ENSDF) for energies, uncertainties, spins, and parities.
2. Infer uncertainties from precision (significant figures) where explicit values are missing.
3. Standardize data into a consistent JSON schema for the pipeline.
4. Output structured files to `data/raw/` for ingestion by Feature Engineer.

Architecture:
- `infer_uncertainty_from_precision`: Heuristic engine for precision-based uncertainty estimation.
- `parse_spin_parity`: Normalizes ENSDF Spin-Parity strings into structured spin/parity hypothesis lists.
- `parse_ensdf_line`: Slices individual ENSDF L records into structured level data dictionaries.
- `convert_log_to_datasets`: Main driver that reads the evaluator log and writes per-dataset JSON files.
"""

import json
import re
import os
import argparse

def infer_uncertainty_from_precision(value_string):
    """
    Infers uncertainty from reported precision in Evaluated Nuclear Structure Data File (ENSDF) evaluatorInput string.
    
    Nuclear physics convention: Uncertainty = 0.5 × least_significant_digit_place_value
    
    Examples:
    - "2000" → ±5 keV (integer, precision to 10s place)
    - "2.0E3" → ±500 keV (scientific notation, 1 decimal in mantissa)
    - "2.00E3" → ±50 keV (scientific notation, 2 decimals in mantissa)
    - "1234.5" → ±0.5 keV (1 decimal place)
    - "567.89" → ±0.05 keV (2 decimal places)
    
    Returns: Inferred uncertainty as float, or 5.0 (conservative default) if cannot parse
    """
    if not value_string:
        return 5.0
    
    value_string = value_string.strip()
    
    # Handle scientific notation (e.g., "2.0E3" or "1.5e+02")
    if 'E' in value_string.upper():
        parts = value_string.upper().split('E')
        if len(parts) != 2:
            return 5.0
        
        mantissa = parts[0]
        try:
            exponent = int(parts[1])
        except ValueError:
            return 5.0
        
        # Count decimal places in mantissa (DO NOT strip trailing zeros)
        # Example: "2.0E3" has 1 decimal → ±500 keV
        # Example: "2.00E3" has 2 decimals → ±50 keV
        if '.' in mantissa:
            decimal_part = mantissa.split('.')[1]
            decimal_places = len(decimal_part)
            
            # Uncertainty: 5 × 10^(-decimal_places) × 10^exponent
            mantissa_uncertainty = 5.0 * (10 ** (-decimal_places))
            return mantissa_uncertainty * (10 ** exponent)
        else:
            # Integer mantissa: "2E3" → ±5×10^3 = ±5000
            return 5.0 * (10 ** exponent)
    
    # Handle regular decimal notation
    if '.' in value_string:
        # Count decimal places WITHOUT stripping trailing zeros
        # Example: "2000.0" has 1 decimal place → precision to 0.1 keV → ±0.5 keV
        # Example: "1234.56" has 2 decimal places → precision to 0.01 keV → ±0.05 keV
        decimal_part = value_string.split('.')[1]
        decimal_places = len(decimal_part)
        return 5.0 * (10 ** (-decimal_places))
    else:
        # Integer: precision to nearest 10
        return 5.0

def parse_spin_parity(spin_parity_string):
    """
    Parses ENSDF Spin-Parity string into a list of structured spin/parity hypotheses.

    Handles the following ENSDF patterns:
    - Simple assignments:   "2-", "3/2+"
    - Tentative single:     "(2)-", "(3/2)+"
    - Tentative lists:      "(2,3)-", "(1,2)+"   [parity outside parentheses]
    - Tentative ranges:     "(1:3)-", "(0:2)+"   [parity outside parentheses]
    - Bare ranges:          "1:3"                [no tentative markers, no parity]
    - Mixed-parity lists:   "1+,2-"
    """
    if not spin_parity_string or spin_parity_string.lower() in ['unknown', 'none']:
        return []

    clean_string = spin_parity_string.strip().rstrip('.')
    if not clean_string:
        return []

    # 1. Extract the global trailing parity sign.
    #    In ENSDF, parity often follows the closing parenthesis: "(2,3)-", "(1:3)+".
    global_parity = None
    working_string = clean_string
    if working_string[-1] in ('+', '-'):
        global_parity = working_string[-1]
        working_string = working_string[:-1].strip()

    if not working_string:
        return []

    # 2. Detect and strip global tentative parentheses enclosing the spin expression.
    #    Examples: "(2,3)" after parity removal, "(1:3)" after parity removal.
    is_globally_tentative = working_string.startswith('(') and working_string.endswith(')')
    if is_globally_tentative:
        working_string = working_string[1:-1]

    if not working_string:
        return []

    # 3. Range support: "1:3" or "(1:3)-" (after stripping) -> "1:3".
    #    Step by 2 in 2J space to correctly cover both integer (2,4,6) and
    #    half-integer (1,3,5) spin sequences with a single rule.
    if ':' in working_string and ',' not in working_string:
        range_parts = working_string.split(':')
        if len(range_parts) == 2:
            try:
                start_spin = float(eval(range_parts[0].strip()))
                end_spin   = float(eval(range_parts[1].strip()))
                start_two_j = int(round(start_spin * 2))
                end_two_j   = int(round(end_spin   * 2))
                is_tentative = is_globally_tentative or '(' in clean_string
                # range stop is exclusive, so use end_two_j + 2 to include the endpoint
                # Parity from outside the closing parenthesis is a firm assignment
                # regardless of whether the spin range itself is tentative.
                # Example: "(1:3)-" → spin tentative, parity '-' is firm.
                return [{
                    "twoTimesSpin": two_j,
                    "isTentativeSpin": is_tentative,
                    "parity": global_parity,
                    "isTentativeParity": False
                } for two_j in range(start_two_j, end_two_j + 2, 2)]
            except (ValueError, SyntaxError):
                pass

    # 4. List parsing: "2,3", "1+,2-", or after stripping "2,3" from "(2,3)-".
    results = []
    for part in [part.strip() for part in working_string.split(',')]:
        if not part:
            continue

        # Extract parity local to this part (e.g. "1+" in the list "1+,2-").
        # Fall back to the global parity if no local parity sign is present.
        part_parity = global_parity
        if part[-1] in ('+', '-'):
            part_parity = part[-1]
            part = part[:-1].strip()

        # Detect local tentative parentheses on this individual spin token.
        is_locally_tentative = '(' in part or ')' in part
        is_tentative = is_globally_tentative or is_locally_tentative

        spin_raw = part.replace('(', '').replace(')', '').strip()
        if not spin_raw:
            continue

        try:
            spin_value_parsed = float(eval(spin_raw))
            two_times_spin_value = int(round(spin_value_parsed * 2))
            # Parity written outside the closing parenthesis is a firm assignment.
            # Parity found inside the tentative block (i.e. not from global_parity)
            # is itself tentative alongside the spin.
            # Example: "(1,2)-" → isTentativeParity = False (parity outside parens)
            # Example: "(2+)"   → isTentativeParity = True  (parity inside parens)
            parity_is_from_outside_parens = (
                (part_parity is not None)
                and (part_parity == global_parity)
                and (global_parity is not None)
            )
            results.append({
                "twoTimesSpin": two_times_spin_value,
                "isTentativeSpin": is_tentative,
                "parity": part_parity,
                "isTentativeParity": (not parity_is_from_outside_parens) and is_tentative and (part_parity is not None)
            })
        except (ValueError, SyntaxError):
            continue

    return results

def calculate_absolute_uncertainty(value_string, uncertainty_string):
    """
    Calculates absolute uncertainty based on Evaluated Nuclear Structure Data File convention.
    123(12) -> 12
    123.4(12) -> 1.2
    0.123(4) -> 0.0004
    """
    if not uncertainty_string:
        return 0.0
    
    value_string = value_string.strip()
    if '.' in value_string:
        decimal_part = value_string.split('.')[1]
        decimals = len(decimal_part)
    else:
        decimals = 0
        
    return float(uncertainty_string) * (10 ** -decimals)

def format_evaluator_input(value_string, uncertainty_string):
    if not uncertainty_string:
        return value_string
    return f"{value_string} {uncertainty_string}"

def parse_ensdf_line(line):
    """
    Parses a single ENSDF L record using fixed-width column slicing.
    Applies precision-based uncertainty inference when an explicit uncertainty is absent.
    Skips all non-L record types (comment, gamma, header, and identification records).
    """
    if len(line) < 8:
        return None, None

    record_type = line[7]
    # Require both: record type 'L' at position 7 AND a blank at position 6.
    # Comment records (cL) also carry 'L' at position 7 but have 'c' at position 6
    # and must be excluded so their text content is never parsed as level data.
    if record_type != 'L' or line[6] == 'c':
        return None, None
        
    energy_string = line[9:19].strip()
    uncertainty_string = line[19:21].strip()
    
    if not energy_string:
        return None, None
        
    energy_value = float(energy_string)
    
    # Calculate uncertainty: explicit if provided, otherwise infer from precision
    if uncertainty_string:
        uncertainty_value = calculate_absolute_uncertainty(energy_string, uncertainty_string)
    elif energy_value == 0.0:
        # Ground state (0.0 keV) is the absolute reference => 0 uncertainty
        uncertainty_value = 0.0
    else:
        uncertainty_value = infer_uncertainty_from_precision(energy_string)
    
    spin_parity_raw = line[22:39].strip()
    data = {
        "energy": {
            "unit": "keV",
            "value": energy_value,
            "uncertainty": {
                "value": uncertainty_value,
                "type": "symmetric" if uncertainty_string else "inferred"
            }
        },
        "isStable": False,
        "gamma_decays": []
    }
    if energy_string:
        data["energy"]["evaluatorInput"] = format_evaluator_input(energy_string, uncertainty_string)

    spin_parity_values = parse_spin_parity(spin_parity_raw)
    if spin_parity_values or spin_parity_raw:
        data["spinParity"] = {}
        if spin_parity_values:
            data["spinParity"]["values"] = spin_parity_values
        if spin_parity_raw:
            data["spinParity"]["evaluatorInput"] = spin_parity_raw

    return "L", data

def convert_log_to_datasets(log_path):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    datasets = {}
    current_dataset_id = None

    for line in lines:
        raw_line = line # Keep raw bytes for fixed-width slicing
        line = line.strip()
        if not line: continue

        if line.startswith("# Dataset"):
            parts = line.split("Dataset")
            if len(parts) > 1:
                current_dataset_id = parts[1].split(":")[0].strip()
                datasets[current_dataset_id] = []
                print(f"Processing Dataset {current_dataset_id}...")
            continue

        record_type, record_data = parse_ensdf_line(raw_line)

        if record_type == "L" and current_dataset_id is not None:
            datasets[current_dataset_id].append(record_data)

    # Output JSON Files
    for dataset_code, level_list in datasets.items():
        gammas_table_list = []
        gamma_counter = 0
        
        for level_index, level_item in enumerate(level_list):
            if "gamma_decays" in level_item:
                level_gamma_indices = []
                initial_level_energy = level_item["energy"]["value"]
                
                for gamma_data in level_item["gamma_decays"]:
                    gamma_energy_value = gamma_data["energy_value"]
                    final_energy_target = initial_level_energy - gamma_energy_value
                    
                    # Match Final Level
                    best_match_index = 0
                    minimum_difference = 1e9
                    for candidate_index, candidate_level in enumerate(level_list):
                        difference = abs(candidate_level["energy"]["value"] - final_energy_target)
                        if difference < minimum_difference:
                            minimum_difference = difference
                            best_match_index = candidate_index
                            
                    final_level_index = best_match_index if minimum_difference <= 50.0 else 0
                    
                    # Structuring Gamma Entry
                    gamma_entry = {
                        "energy": {
                            "unit": "keV",
                            "value": gamma_energy_value,
                            "uncertainty": { "value": gamma_data["energy_uncertainty"], "type": "symmetric" } if gamma_data["energy_uncertainty"] > 0 else {"type": "unreported"}
                        },
                        "gammaIntensity": {
                            "value": gamma_data["relative_intensity_value"],
                            "uncertainty": { "value": gamma_data["intensity_uncertainty_value"], "type": "symmetric" } if gamma_data["intensity_uncertainty_value"] > 0 else {"type": "unreported"}
                        },
                        "initialLevel": level_index,
                        "finalLevel": final_level_index
                    }
                    if gamma_data.get("energy_input_string"):
                        gamma_entry["energy"]["evaluatorInput"] = gamma_data["energy_input_string"]
                    if gamma_data.get("intensity_input_string"):
                        gamma_entry["gammaIntensity"]["evaluatorInput"] = gamma_data["intensity_input_string"]
                        
                    gammas_table_list.append(gamma_entry)
                    level_gamma_indices.append(gamma_counter)
                    gamma_counter += 1
                
                if level_gamma_indices:
                    level_item["gammas"] = level_gamma_indices
                del level_item["gamma_decays"]
        
        output_data_structure = { "levelsTable": { "levels": level_list } }
        if gammas_table_list:
            output_data_structure["gammasTable"] = { "gammas": gammas_table_list }
            
        # Robustly determine output directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, 'data', 'raw', f"test_dataset_{dataset_code}.json")
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data_structure, f, indent=4)
        print(f"Dataset {dataset_code} saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?', default="evaluatorInput_34Cl.log")
    args = parser.parse_args()
    convert_log_to_datasets(args.input_file)
