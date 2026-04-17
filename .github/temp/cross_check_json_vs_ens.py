"""
Cross-check generated JSON datasets against source ENS files.

# High-level Structure and Workflow Explanation:
======================================

Workflow Diagram:

  [K ENS file]      [L ENS file]
       |                 |
       v                 v
  [Independent ENS fixed-width parser]
       |                 |
       +--------+--------+
                |
                v
      [Expected levels and gammas]
                |
                v
      [Compare against JSON outputs]
                |
                v
      [Mismatch report + spot-check]

Technical Steps:
1. Parse the ENS files directly using fixed-width columns for L and G records.
2. Build expected level and gamma structures independently of Dataset_Parser.py runtime state.
3. Compare source-derived values against JSON fields for completeness and exact stored strings.
4. Run a reproducible 15% random spot-check over levels and gammas.
5. Print all mismatches and summary counts.
"""

import json
import math
import random
from fractions import Fraction
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def infer_uncertainty_from_precision(value_string):
    value_string = value_string.strip()
    if not value_string:
        return 5.0
    if "E" in value_string.upper():
        mantissa, exponent = value_string.upper().split("E", 1)
        exponent_value = int(exponent)
        if "." in mantissa:
            decimal_places = len(mantissa.split(".", 1)[1])
            return 5.0 * (10 ** (-decimal_places)) * (10 ** exponent_value)
        return 5.0 * (10 ** exponent_value)
    if "." in value_string:
        decimal_places = len(value_string.split(".", 1)[1])
        return 5.0 * (10 ** (-decimal_places))
    return 5.0


def calculate_absolute_uncertainty(value_string, uncertainty_string):
    value_string = value_string.strip()
    if not uncertainty_string:
        return 0.0
    decimal_places = len(value_string.split(".", 1)[1]) if "." in value_string else 0
    return float(uncertainty_string) * (10 ** (-decimal_places))


def format_evaluator_input(value_string, uncertainty_string):
    if not uncertainty_string:
        return value_string
    return f"{value_string} {uncertainty_string}"


def parse_spin_value(spin_token):
    token = spin_token.strip()
    if not token:
        raise ValueError("empty spin token")
    if "/" in token:
        return float(Fraction(token))
    return float(token)


def parse_spin_parity_reference(spin_parity_string):
    if not spin_parity_string or spin_parity_string.lower() in ["unknown", "none"]:
        return []

    clean_string = spin_parity_string.strip().rstrip(".")
    if not clean_string:
        return []

    global_parity = None
    working_string = clean_string
    if working_string[-1] in ("+", "-"):
        global_parity = working_string[-1]
        working_string = working_string[:-1].strip()

    if not working_string:
        return []

    is_globally_tentative = working_string.startswith("(") and working_string.endswith(")")
    if is_globally_tentative:
        working_string = working_string[1:-1]

    if not working_string:
        return []

    if ":" in working_string and "," not in working_string:
        start_raw, end_raw = [part.strip() for part in working_string.split(":", 1)]
        start_spin = parse_spin_value(start_raw)
        end_spin = parse_spin_value(end_raw)
        start_two_j = int(round(start_spin * 2))
        end_two_j = int(round(end_spin * 2))
        is_tentative = is_globally_tentative
        return [
            {
                "twoTimesSpin": two_j,
                "isTentativeSpin": is_tentative,
                "parity": global_parity,
                "isTentativeParity": False if global_parity is not None else False,
            }
            for two_j in range(start_two_j, end_two_j + 2, 2)
        ]

    results = []
    for part in [part.strip() for part in working_string.split(",")]:
        if not part:
            continue
        part_parity = global_parity
        if part[-1] in ("+", "-"):
            part_parity = part[-1]
            part = part[:-1].strip()
        is_locally_tentative = "(" in part or ")" in part
        is_tentative = is_globally_tentative or is_locally_tentative
        spin_raw = part.replace("(", "").replace(")", "").strip()
        if not spin_raw:
            continue
        spin_value = parse_spin_value(spin_raw)
        parity_is_from_outside_parens = (
            part_parity is not None and global_parity is not None and part_parity == global_parity
        )
        results.append(
            {
                "twoTimesSpin": int(round(spin_value * 2)),
                "isTentativeSpin": is_tentative,
                "parity": part_parity,
                "isTentativeParity": (not parity_is_from_outside_parens) and is_tentative and (part_parity is not None),
            }
        )
    return results


def parse_level_record(line):
    if len(line) < 8 or not (line[6:7] == " " and line[7:8] == "L"):
        return None
    energy_string = line[9:19].strip()
    uncertainty_string = line[19:21].strip()
    spin_parity_raw = line[22:39].strip()
    if not energy_string:
        return None
    energy_value = float(energy_string)
    if uncertainty_string:
        uncertainty_value = calculate_absolute_uncertainty(energy_string, uncertainty_string)
        uncertainty_type = "symmetric"
    elif energy_value == 0.0:
        uncertainty_value = 0.0
        uncertainty_type = "inferred"
    else:
        uncertainty_value = infer_uncertainty_from_precision(energy_string)
        uncertainty_type = "inferred"
    return {
        "energy": {
            "unit": "keV",
            "value": energy_value,
            "uncertainty": {
                "value": uncertainty_value,
                "type": uncertainty_type,
            },
            "evaluatorInput": format_evaluator_input(energy_string, uncertainty_string),
        },
        "spinParity": {
            "evaluatorInput": spin_parity_raw,
            "values": parse_spin_parity_reference(spin_parity_raw),
        } if spin_parity_raw or parse_spin_parity_reference(spin_parity_raw) else None,
        "isStable": False,
        "gamma_decays": [],
    }


def parse_gamma_record(line):
    if len(line) < 23 or not (line[6:7] == " " and line[7:8] == "G"):
        return None
    energy_string = line[9:19].strip()
    uncertainty_string = line[19:21].strip()
    intensity_string = line[22:29].strip()
    intensity_uncertainty_string = line[29:31].strip()
    if not energy_string:
        return None
    energy_value = float(energy_string)
    if uncertainty_string:
        energy_uncertainty = calculate_absolute_uncertainty(energy_string, uncertainty_string)
        energy_uncertainty_type = "symmetric"
    else:
        energy_uncertainty = infer_uncertainty_from_precision(energy_string)
        energy_uncertainty_type = "symmetric" if energy_uncertainty > 0 else "unreported"
    try:
        relative_intensity_value = float(intensity_string) if intensity_string else 0.0
    except ValueError:
        relative_intensity_value = 0.0
    try:
        intensity_uncertainty_value = (
            float(intensity_uncertainty_string)
            if intensity_uncertainty_string and intensity_uncertainty_string not in ("LT", "GT")
            else 0.0
        )
    except ValueError:
        intensity_uncertainty_value = 0.0
    return {
        "energy": {
            "unit": "keV",
            "value": energy_value,
            "uncertainty": {
                "value": energy_uncertainty,
                "type": "symmetric" if energy_uncertainty > 0 else "unreported",
            },
            "evaluatorInput": format_evaluator_input(energy_string, uncertainty_string),
        },
        "gammaIntensity": {
            "value": relative_intensity_value,
            "uncertainty": {
                "value": intensity_uncertainty_value,
                "type": "symmetric" if intensity_uncertainty_value > 0 else "unreported",
            },
            "evaluatorInput": format_evaluator_input(intensity_string, intensity_uncertainty_string) if intensity_string else "",
        },
    }


def load_expected_dataset_from_ens(ens_path):
    with open(ens_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    level_list = []
    current_level = None
    for line in lines:
        level_data = parse_level_record(line)
        if level_data is not None:
            current_level = level_data
            level_list.append(current_level)
            continue
        gamma_data = parse_gamma_record(line)
        if gamma_data is not None and current_level is not None:
            current_level["gamma_decays"].append(gamma_data)

    gammas_table = []
    gamma_counter = 0
    for level_index, level_item in enumerate(level_list):
        level_gamma_indices = []
        initial_level_energy = level_item["energy"]["value"]
        for gamma_data in level_item["gamma_decays"]:
            gamma_energy_value = gamma_data["energy"]["value"]
            final_energy_target = initial_level_energy - gamma_energy_value
            best_match_index = 0
            minimum_difference = 1e9
            for candidate_index, candidate_level in enumerate(level_list):
                difference = abs(candidate_level["energy"]["value"] - final_energy_target)
                if difference < minimum_difference:
                    minimum_difference = difference
                    best_match_index = candidate_index
            final_level_index = best_match_index if minimum_difference <= 50.0 else 0
            gamma_entry = {
                "energy": gamma_data["energy"],
                "gammaIntensity": gamma_data["gammaIntensity"],
                "initialLevel": level_index,
                "finalLevel": final_level_index,
            }
            gammas_table.append(gamma_entry)
            level_gamma_indices.append(gamma_counter)
            gamma_counter += 1
        if level_gamma_indices:
            level_item["gammas"] = level_gamma_indices
        del level_item["gamma_decays"]
    result = {"levelsTable": {"levels": level_list}}
    if gammas_table:
        result["gammasTable"] = {"gammas": gammas_table}
    return result


def compare_values(mismatches, dataset_label, path, expected_value, actual_value):
    if expected_value != actual_value:
        mismatches.append(
            f"{dataset_label} | {path} | expected={expected_value!r} | actual={actual_value!r}"
        )


def compare_float(mismatches, dataset_label, path, expected_value, actual_value, tolerance=1e-9):
    if not math.isclose(expected_value, actual_value, rel_tol=0.0, abs_tol=tolerance):
        mismatches.append(
            f"{dataset_label} | {path} | expected={expected_value!r} | actual={actual_value!r}"
        )


def cross_check_dataset(dataset_label, ens_path, json_path):
    mismatches = []
    expected = load_expected_dataset_from_ens(ens_path)
    with open(json_path, "r", encoding="utf-8") as handle:
        actual = json.load(handle)

    expected_levels = expected["levelsTable"]["levels"]
    actual_levels = actual["levelsTable"]["levels"]
    compare_values(mismatches, dataset_label, "levels.count", len(expected_levels), len(actual_levels))

    for level_index, (expected_level, actual_level) in enumerate(zip(expected_levels, actual_levels)):
        level_path = f"levels[{level_index}]"
        compare_values(mismatches, dataset_label, f"{level_path}.energy.unit", expected_level["energy"]["unit"], actual_level["energy"]["unit"])
        compare_float(mismatches, dataset_label, f"{level_path}.energy.value", expected_level["energy"]["value"], actual_level["energy"]["value"])
        compare_float(mismatches, dataset_label, f"{level_path}.energy.uncertainty.value", expected_level["energy"]["uncertainty"]["value"], actual_level["energy"]["uncertainty"]["value"])
        compare_values(mismatches, dataset_label, f"{level_path}.energy.uncertainty.type", expected_level["energy"]["uncertainty"]["type"], actual_level["energy"]["uncertainty"]["type"])
        compare_values(mismatches, dataset_label, f"{level_path}.energy.evaluatorInput", expected_level["energy"]["evaluatorInput"], actual_level["energy"].get("evaluatorInput"))
        compare_values(mismatches, dataset_label, f"{level_path}.isStable", expected_level["isStable"], actual_level["isStable"])

        expected_spin = expected_level.get("spinParity")
        actual_spin = actual_level.get("spinParity")
        compare_values(mismatches, dataset_label, f"{level_path}.spinParity.present", expected_spin is not None, actual_spin is not None)
        if expected_spin is not None and actual_spin is not None:
            compare_values(mismatches, dataset_label, f"{level_path}.spinParity.evaluatorInput", expected_spin.get("evaluatorInput"), actual_spin.get("evaluatorInput"))
            expected_values = expected_spin.get("values", [])
            actual_values = actual_spin.get("values", [])
            compare_values(mismatches, dataset_label, f"{level_path}.spinParity.values.count", len(expected_values), len(actual_values))
            for spin_index, (expected_value, actual_value) in enumerate(zip(expected_values, actual_values)):
                spin_path = f"{level_path}.spinParity.values[{spin_index}]"
                compare_values(mismatches, dataset_label, f"{spin_path}.twoTimesSpin", expected_value["twoTimesSpin"], actual_value["twoTimesSpin"])
                compare_values(mismatches, dataset_label, f"{spin_path}.isTentativeSpin", expected_value["isTentativeSpin"], actual_value["isTentativeSpin"])
                compare_values(mismatches, dataset_label, f"{spin_path}.parity", expected_value["parity"], actual_value["parity"])
                compare_values(mismatches, dataset_label, f"{spin_path}.isTentativeParity", expected_value["isTentativeParity"], actual_value["isTentativeParity"])

        compare_values(mismatches, dataset_label, f"{level_path}.gammas", expected_level.get("gammas", []), actual_level.get("gammas", []))

    expected_gammas = expected.get("gammasTable", {}).get("gammas", [])
    actual_gammas = actual.get("gammasTable", {}).get("gammas", [])
    compare_values(mismatches, dataset_label, "gammas.count", len(expected_gammas), len(actual_gammas))
    for gamma_index, (expected_gamma, actual_gamma) in enumerate(zip(expected_gammas, actual_gammas)):
        gamma_path = f"gammas[{gamma_index}]"
        compare_float(mismatches, dataset_label, f"{gamma_path}.energy.value", expected_gamma["energy"]["value"], actual_gamma["energy"]["value"])
        compare_float(mismatches, dataset_label, f"{gamma_path}.energy.uncertainty.value", expected_gamma["energy"]["uncertainty"].get("value", 0.0), actual_gamma["energy"]["uncertainty"].get("value", 0.0))
        compare_values(mismatches, dataset_label, f"{gamma_path}.energy.uncertainty.type", expected_gamma["energy"]["uncertainty"]["type"], actual_gamma["energy"]["uncertainty"]["type"])
        compare_values(mismatches, dataset_label, f"{gamma_path}.energy.evaluatorInput", expected_gamma["energy"].get("evaluatorInput"), actual_gamma["energy"].get("evaluatorInput"))
        compare_float(mismatches, dataset_label, f"{gamma_path}.gammaIntensity.value", expected_gamma["gammaIntensity"]["value"], actual_gamma["gammaIntensity"]["value"])
        compare_float(mismatches, dataset_label, f"{gamma_path}.gammaIntensity.uncertainty.value", expected_gamma["gammaIntensity"]["uncertainty"].get("value", 0.0), actual_gamma["gammaIntensity"]["uncertainty"].get("value", 0.0))
        compare_values(mismatches, dataset_label, f"{gamma_path}.gammaIntensity.uncertainty.type", expected_gamma["gammaIntensity"]["uncertainty"]["type"], actual_gamma["gammaIntensity"]["uncertainty"]["type"])
        compare_values(mismatches, dataset_label, f"{gamma_path}.gammaIntensity.evaluatorInput", expected_gamma["gammaIntensity"].get("evaluatorInput"), actual_gamma["gammaIntensity"].get("evaluatorInput"))
        compare_values(mismatches, dataset_label, f"{gamma_path}.initialLevel", expected_gamma["initialLevel"], actual_gamma["initialLevel"])
        compare_values(mismatches, dataset_label, f"{gamma_path}.finalLevel", expected_gamma["finalLevel"], actual_gamma["finalLevel"])

    spot_check_messages = []
    random_generator = random.Random(34017)
    spot_level_count = max(1, math.ceil(len(expected_levels) * 0.15))
    for level_index in sorted(random_generator.sample(range(len(expected_levels)), spot_level_count)):
        expected_level = expected_levels[level_index]
        actual_level = actual_levels[level_index]
        if expected_level["energy"]["evaluatorInput"] != actual_level["energy"].get("evaluatorInput"):
            spot_check_messages.append(f"level index {level_index} energy evaluatorInput mismatch")
        expected_spin = expected_level.get("spinParity") or {}
        actual_spin = actual_level.get("spinParity") or {}
        if expected_spin.get("evaluatorInput") != actual_spin.get("evaluatorInput"):
            spot_check_messages.append(f"level index {level_index} spin evaluatorInput mismatch")
    if actual_gammas:
        spot_gamma_count = max(1, math.ceil(len(expected_gammas) * 0.15))
        for gamma_index in sorted(random_generator.sample(range(len(expected_gammas)), spot_gamma_count)):
            expected_gamma = expected_gammas[gamma_index]
            actual_gamma = actual_gammas[gamma_index]
            if expected_gamma["energy"].get("evaluatorInput") != actual_gamma["energy"].get("evaluatorInput"):
                spot_check_messages.append(f"gamma index {gamma_index} energy evaluatorInput mismatch")
            if expected_gamma["initialLevel"] != actual_gamma["initialLevel"] or expected_gamma["finalLevel"] != actual_gamma["finalLevel"]:
                spot_check_messages.append(f"gamma index {gamma_index} level linkage mismatch")

    return {
        "dataset": dataset_label,
        "mismatches": mismatches,
        "spot_check_messages": spot_check_messages,
        "level_count": len(expected_levels),
        "gamma_count": len(expected_gammas),
    }


def main():
    reports = [
        cross_check_dataset(
            "K",
            DATA_DIR / "raw" / "Cl34_33s_p_g.ens",
            DATA_DIR / "json" / "test_dataset_K.json",
        ),
        cross_check_dataset(
            "L",
            DATA_DIR / "raw" / "Cl34_33s_p_p_resonances.ens",
            DATA_DIR / "json" / "test_dataset_L.json",
        ),
    ]

    total_mismatches = 0
    for report in reports:
        print(f"Dataset {report['dataset']}: levels={report['level_count']} gammas={report['gamma_count']}")
        if report["mismatches"]:
            print("  Mismatches:")
            for mismatch in report["mismatches"]:
                print(f"    {mismatch}")
        else:
            print("  Mismatches: none")
        if report["spot_check_messages"]:
            print("  Spot-check issues:")
            for message in report["spot_check_messages"]:
                print(f"    {message}")
        else:
            print("  Spot-check: passed")
        total_mismatches += len(report["mismatches"]) + len(report["spot_check_messages"])

    print(f"Total issues: {total_mismatches}")


if __name__ == "__main__":
    main()