"""
Feature Engineering For Nuclear Level Matching
======================================

# High-level Structure and Module Architecture:
-------------------------

Functional Architecture:
                                   [Scoring Config & Physics Rules]
                                                 |
                                                 v
[Data Ingestion]                      [Similarity Kernels]
(parse_json_datasets)                 (Energy, Spin, Parity, Gamma)
       |                                         |
       v                                         v
[Standardized Levels] --------+----> [Feature Extraction] ----> [5D Feature Vector]
                              |      (extract_features)
                              |
                              +----> [Synthetic Generator] ----> [Training Data]
                                     (generate_synthetic_    (Features + Labels)
                                      training_data)

Technical Components:
1. **Data Ingestion** (`parse_json_datasets`):
   - Parses NNDC JSON schema with nested uncertainty structure.
   - Extracts: Level properties (Energy, Jπ) and Gamma Decay fingerprints.
   - Attaches gamma_decays list with energy/intensity and their uncertainties.
   - **Uncertainty Handling**: Dataset_Parser.py infers uncertainties from precision during ENSDF parsing.
     JSON contains either explicit uncertainties (type="symmetric") or inferred (type="inferred").
     Feature_Engineer trusts parser output; fallbacks only for edge cases (manual data, corrupted JSON).

2. **Similarity Scoring** (`calculate_*_similarity`):
   - **Energy**: Gaussian kernel exp(-Sigma_Scale×z²) where z=ΔE/σ_combined. Sigma_Scale=0.1 (lenient).
   - **Spin/Parity**: Optimistic matching (maximum across all Jπ option pairs). Scores from Scoring_Config.
     Neutral_Score=0.5 when data missing. Match_Firm=1.0, Match_Strong=0.8, Mismatch thresholds vary.
   - **Gamma Decay Pattern**: Dual-mode statistical compatibility with subset-robust normalization.
     • Intensity Mode: Greedy one-to-one matching with Z-test thresholds (Energy≤3σ, Intensity≤3σ).
       Average intensity if statistically compatible, minimum if inconsistent.
       Normalizes by smaller dataset total (handles partial observations).
     • Binary Mode: Energy-only overlap coefficient when intensities absent.

3. **5D Feature Vector** (`extract_features`):
   - Monotonic-increasing features: [Energy_Similarity, Spin_Similarity, Parity_Similarity, 
     Specificity, Gamma_Decay_Pattern_Similarity].
   - **Specificity**: Ambiguity penalty = 1/f(multiplicity) where multiplicity = options_count_1 × options_count_2.
     Formula configurable: sqrt (default), linear, log, or tunable via Scoring_Config['Specificity']['Formula'].
   - Features use precision-based uncertainty inference when data missing (no hardcoded defaults).

4. **Training Data** (`generate_synthetic_training_data`):
   - Grid-based feature space coverage + random sampling (500 points) + perfect-match boost (50 points).
   - Labels: Hard-veto mismatches→0.0; probability = Energy_Similarity × sqrt(Physics_Confidence) × Specificity.
   - **Feature Correlation** ("Physics Rescue"):
     • **Quantum/Gamma Match** (Score ≥ 0.85): Triggers Rescue (effective_energy = energy^Rescue_Exponent).
     • Uses `Rescue_Exponent` (default 0.5) to dampen energy penalties when physics strongly agrees.
     Teaches XGBoost that structural fingerprints (Gamma) definitively identify levels despite calibration shifts.

"""

import re
import numpy as np
import json
import os

# ==========================================
# Configuration: Scoring Parameters
# ==========================================
# This section quantifies the similarity scoring for energy, spin, parity, and level property sensitivity/quality/specificity/certainty.
# Range: 0.0 (No Match) to 1.0 (Perfect Match).
# Modify these values to adjust the empirical physics scale.

Scoring_Config = {
    'Energy': {
        # Controls how strictly energy values must match. 
        # - Higher value (e.g. 1.0) = stricter (score drops fast if energy differs).
        # - Lower value (e.g. 0.1) = looser (score stays high even with differences).
         # Sigma_Scale=0.1 (loose):   1σ→90.5%, 2σ→67.0%, 3σ→40.7%, 4σ→20.2%, 5σ→8.2%
        'Sigma_Scale': 0.1
    },
    'Spin': {
        # Similarity scores for Spin (J) comparisons (0.0 to 1.0)
        'Match_Firm': 1.0,         # both firm, e.g., 2 vs 2
        'Match_Strong': 0.8,    # any tentative, e.g., 2 vs (2)
        'Mismatch_Weak': 0.2,      # any tentative, e.g., 2 vs (3) with ΔJ=1
        'Mismatch_Strong': 0.05,    # both firm, e.g., 2 vs 3 with ΔJ=1
        'Mismatch_Firm': 0.0               # any ΔJ ≥ 2
    },
    'Parity': {
        # Similarity scores for Parity (Pi) comparisons (0.0 to 1.0)
        'Match_Firm': 1.0,         # both firm, e.g., + vs + or - vs -
        'Match_Strong': 0.8,    # any tentative, e.g., + vs (+)
        'Mismatch_Weak': 0.05, # any tentative, e.g., + vs (-)
        'Mismatch_Firm': 0.0       # both firm, e.g., + vs -
    },
    'Specificity': {
        # Ambiguity penalty for levels with multiple Jπ options
        # Formula determines how harshly to penalize multiplicity (options_count_1 × options_count_2)
        'Formula': 'sqrt',  # Options: 'sqrt', 'linear', 'log', 'tunable'
        # Only used if Formula='tunable': specificity = 1/(1 + Alpha*(mult-1))
        # Higher Alpha → steeper penalty. Example: Alpha=0.5 gives mult=9→18% (82% penalty)
        'Alpha': 0.5
    },
    'Feature_Correlation': {
        # Controls how perfect spin/parity matching can "rescue" mediocre energy similarity.
        # Physics rationale: If two levels have identical quantum numbers (Jπ) and are isolated
        # (no nearby levels with same Spin-Parity), energy disagreement may be due to measurement error
        # or calibration issues rather than genuine mismatch.
        #
        # When Enabled: If spin_similarity >= Threshold AND parity_similarity >= Threshold,
        # the effective energy similarity is boosted: effective_energy = energy^Rescue_Exponent
        # Example: Rescue_Exponent=0.5 (sqrt) transforms energy=0.4→0.63, energy=0.6→0.77
        'Enabled': True,
        'Threshold': 0.85,        # Minimum spin/parity similarity to trigger rescue (firm matches only)
        'Rescue_Exponent': 0.5    # Exponent for energy boost: energy → energy^exponent (0.5 = square root, 0.7 = gentler)
    },
    'General': {
        # Score used when data is missing (e.g. unknown). Perform **Manual Imputation**. Machine Learning models receive a concrete number of 0.5, not a missing value, so it treats it like any other feature value.
        'Neutral_Score': 0.5
    }
}

def parse_json_datasets(dataset_codes):
    """
    Parses JSON files (modern schema) for the given dataset codes and returns a list of standardized level dictionaries.
    Input files: test_dataset_{code}.json
        Standardized Keys: dataset_code, level_id, energy_value, energy_uncertainty, spin_parity_list, spin_parity_string.
        FRIBND: `spin_parity_string` is the string taken from `spinParity.evaluatorInput`.
    Example Output (includes an ENSDF-style (3/2,5/2)+ case):
      [
        {
            'dataset_code': 'A',
            'level_id': 'A_1000',
            'energy_value': 1000.0,
            'energy_uncertainty': 1.0,
            'spin_parity_list': [
                {'twoTimesSpin': 3, 'isTentativeSpin': True, 'parity': '+', 'isTentativeParity': False},
                {'twoTimesSpin': 5, 'isTentativeSpin': True, 'parity': '+', 'isTentativeParity': False}
            ],
            'spin_parity_string': '(3/2,5/2)+'
        }
      ]
    """
    levels = []
    
    # Robustly determine data directory relative to this script.
    # Primary location is data/json/ because Dataset_Parser now writes standardized
    # datasets there. Fall back to data/raw/ to preserve compatibility with older
    # example repositories and pre-existing test fixtures.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preferred_data_dir = os.path.join(base_dir, 'data', 'json')
    fallback_data_dir = os.path.join(base_dir, 'data', 'raw')
    
    for dataset_code in dataset_codes:
        filename = os.path.join(preferred_data_dir, f"test_dataset_{dataset_code}.json")
        if not os.path.exists(filename):
            filename = os.path.join(fallback_data_dir, f"test_dataset_{dataset_code}.json")

        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Format: New ENSDF JSON schema (levels_table -> levels, gammas_table -> gammas)
                if isinstance(data, dict) and 'levelsTable' in data:
                    raw_levels = data['levelsTable'].get('levels', [])
                    gammas_table_list = data.get('gammasTable', {}).get('gammas', [])
                    
                    for level_index, level_item in enumerate(raw_levels):
                        energy_value = level_item.get('energy', {}).get('value')
                        energy_uncertainty = level_item.get('energy', {}).get('uncertainty', {}).get('value', 5.0)
                        spin_parity_list = level_item.get('spinParity', {}).get('values', [])
                        spin_parity_string = level_item.get('spinParity', {}).get('evaluatorInput')
                        
                        # Extract gamma decays for this level from gammas_table_list
                        gamma_decays = []
                        if 'gammas' in level_item and gammas_table_list:
                            # Level has gamma indices pointing to gammas_table_list
                            for gamma_index in level_item['gammas']:
                                if 0 <= gamma_index < len(gammas_table_list):
                                    gamma_entry = gammas_table_list[gamma_index]
                                    gamma_energy = gamma_entry['energy']
                                    gamma_intensity = gamma_entry.get('gammaIntensity', {})
                                    
                                    gamma_decays.append({
                                        'energy': gamma_energy['value'],
                                        'energy_uncertainty': gamma_energy['uncertainty'].get('value', 5.0),
                                        'intensity': gamma_intensity.get('value', 0),
                                        'intensity_uncertainty': gamma_intensity.get('uncertainty', {}).get('value', 0)
                                    })
                        
                        if energy_value is not None:
                            levels.append({
                                'dataset_code': dataset_code,
                                'level_id': f"{dataset_code}_{int(energy_value)}",
                                'energy_value': float(energy_value),
                                'energy_uncertainty': float(energy_uncertainty),
                                'spin_parity_list': spin_parity_list,
                                'spin_parity_string': spin_parity_string,
                                'gamma_decays': gamma_decays
                            })
    return levels

def calculate_z_score(value_1, uncertainty_1, value_2, uncertainty_2):
    """Calculates statistical separation (Z-score in sigma units)."""
    combined_uncertainty = np.sqrt(uncertainty_1**2 + uncertainty_2**2)
    if combined_uncertainty == 0:
        combined_uncertainty = 1.0
    return abs(value_1 - value_2) / combined_uncertainty

def calculate_energy_similarity(energy_1, energy_uncertainty_1, energy_2, energy_uncertainty_2):
    """
    # Energy similarity: exp(-Sigma_Scale × z²)
    # where z = ΔE/σ_c = energy difference in units of combined uncertainty
    #
    # Example: Level A (E=1000±5 keV) vs Level B (E=1010±3 keV)
    #   ΔE=10 keV, σ_c=√(25+9)=5.83 keV, z=1.72 → similarity=exp(-0.1×1.72²)=0.744 (74.4%)
    #
    # Energy similarity decay with increasing energy separation (z_score in sigma units):
    # Sigma_Scale=0.1 (loose):   1σ→90.5%, 2σ→67.0%, 3σ→40.7%, 4σ→20.2%, 5σ→8.2%
    #   Lenient: tolerates large separations
    # Sigma_Scale=0.2 (moderate): 1σ→81.9%, 2σ→44.9%, 3σ→16.5%, 4σ→4.1%, 5σ→0.7%
    #   Standard: penalizes >2σ strongly
    # Sigma_Scale=0.5 (strict):   1σ→60.7%, 2σ→13.5%, 3σ→1.1%, 4σ→0.0%, 5σ→0.0%
    #   Aggressive: rejects >2σ
    # Sigma_Scale=1.0 (extreme):  1σ→36.8%, 2σ→1.8%, 3σ→0.0%, 4σ→0.0%, 5σ→0.0%
    #   Ultra-strict: even 1σ penalized
    """
    if energy_1 is None or energy_2 is None:
        return 0.0
    z_score = calculate_z_score(energy_1, energy_uncertainty_1, energy_2, energy_uncertainty_2)
    return np.exp(-Scoring_Config['Energy']['Sigma_Scale'] * z_score**2)


def calculate_spin_similarity(spin_parity_list_1, spin_parity_list_2):
    """
    Calculates spin similarity (0 to 1) for matching levels across experimental datasets.
    
    ENSDF Notation Distinctions:
      - Firm multiple:    2,3      = J is definitely 2 or 3 (limited to 2 options)
      - Tentative multiple: (2,3)  = J is tentatively 2 or 3 (and possibly others)
      - Firm single:      2        = J is definitely 2
      - Tentative single: (2)      = J is tentatively 2
    
    Matching Strategy (Optimistic):
      - Compare ALL spin combinations between two levels using nested loop
      - Keep the BEST (maximum) similarity score across all pairs
      - Rationale: If any option matches well, levels likely identical
      - Example: Level A has J=2, Level B has J=(2,3)
        → Compare 2 vs 2 (score 0.9, tentative penalty)
        → Compare 2 vs 3 (score 0.2, adjacent mismatch)
        → Return max(0.9, 0.2) = 0.9 (optimistic: assume match)
    
    Scoring Rules:
      - ΔJ = 0 with both firm, e.g., 2 vs 2: Strongest match → score 1.0
      - ΔJ = 0 with any tentative, e.g., 2 vs (2): Good match with penalty → score 0.9
      - ΔJ = 1 with both firm, e.g., 2 vs 3: Strong mismatch → score 0.05
      - ΔJ = 1 with any tentative, e.g., 2 vs (3): Weak mismatch → score 0.2
      - ΔJ ≥ 2: Strong mismatch regardless of tentativeness → score 0.0
    """
    # Guard: If either level has no spin data, return neutral score (cannot compare)
    if not spin_parity_list_1 or not spin_parity_list_2:
        return Scoring_Config['General']['Neutral_Score']

    # Extract spin values (J) from both levels
    # Input: twoTimesSpin (2J stored as integer in JSON, e.g., 3 means J=3/2)
    # Output: list of (J, isTentative) tuples
    spins_1 = [
        (state['twoTimesSpin'] / 2.0, state['isTentativeSpin'])
        for state in spin_parity_list_1 if state.get('twoTimesSpin') is not None
    ]
    spins_2 = [
        (state['twoTimesSpin'] / 2.0, state['isTentativeSpin'])
        for state in spin_parity_list_2 if state.get('twoTimesSpin') is not None
    ]

    # Guard: If filtering removed all spin values (all were None), return neutral score
    if not spins_1 or not spins_2:
        return Scoring_Config['General']['Neutral_Score']

    # Optimistic matching: Compare all spin pairs and keep the best (maximum) similarity score
    # If ANY spin combination matches, accept two levels match compatibility
    maximum_similarity_score = 0.0
    
    for spin_1, spin1_is_tentative in spins_1:
        for spin_2, spin2_is_tentative in spins_2:
            # Calculate spin difference ΔJ and determine if both measurements are firm
            spin_difference = abs(spin_1 - spin_2)
            both_firm = (not spin1_is_tentative) and (not spin2_is_tentative)
            
            # Apply scoring rules based on ΔJ and firmness
            # ΔJ = 0: Match      → 1.0 (both firm) or 0.9 (any tentative)
            # ΔJ = 1: Adjacent   → 0.05 (both firm) or 0.2 (any tentative)
            # ΔJ ≥ 2: Mismatch  → 0.0 always
            
            if spin_difference == 0.0:
                # Identical spins → evidence for same level
                if both_firm:
                    pair_similarity = Scoring_Config['Spin']['Match_Firm']
                else:
                    pair_similarity = Scoring_Config['Spin']['Match_Strong']
                    
            elif spin_difference == 1.0:
                # Adjacent spins (ΔJ=1) → evidence for different levels
                if both_firm:
                    pair_similarity = Scoring_Config['Spin']['Mismatch_Strong']
                else:
                    pair_similarity = Scoring_Config['Spin']['Mismatch_Weak']
                    
            else:
                # Large spin difference (ΔJ≥2) → definitely different levels
                pair_similarity = Scoring_Config['Spin']['Mismatch_Firm']

            # Keep the best score across all spin pairs
            if pair_similarity > maximum_similarity_score:
                maximum_similarity_score = pair_similarity
    
    return maximum_similarity_score

def calculate_parity_similarity(spin_parity_list_1, spin_parity_list_2):
    """
    Calculates parity similarity (0 to 1) for matching levels across experimental datasets.
    
    ENSDF Notation Distinctions (Parity is binary: + or -, never both):
      - Firm:      +        = Parity is definitely positive
      - Firm:      -        = Parity is definitely negative
      - Tentative: (+)      = Parity is tentatively positive
      - Tentative: (-)      = Parity is tentatively negative
    
    Note: Unlike spin, parity CANNOT have multiple values like "+,-" because parity is a 
    discrete binary quantum number. A level has either positive OR negative parity, never both.
    Multiple parity entries in spin_parity_list occur when BOTH spin AND parity are uncertain
    (e.g., "2+,3-" means J=2 with π=+ OR J=3 with π=-, correlated uncertainty).
    
    Matching Strategy (Optimistic):
      - Compare ALL parity combinations between two levels using nested loop
      - Keep the BEST (maximum) similarity score across all pairs
      - Rationale: If any option matches well, levels likely identical
      - Example: Level A has +, Level B has correlated options (2+, 3-)
        → Compare + vs + (score 1.0, both firm)
        → Compare + vs - (score 0.0, firm mismatch)
        → Return max(1.0, 0.0) = 1.0 (optimistic: assume J=2,π=+ match)
    
    Scoring Rules:
      - Same parity with both firm, e.g., + vs +: Strongest match → score 1.0
      - Same parity with any tentative, e.g., + vs (+): Good match with penalty → score 0.9
      - Different parity with both firm, e.g., + vs -: Strong mismatch → score 0.0
      - Different parity with any tentative, e.g., + vs (-): Weak mismatch → score 0.1
    """
    # Guard: If either level has no spin/parity data, return neutral score (cannot compare)
    if not spin_parity_list_1 or not spin_parity_list_2:
        return Scoring_Config['General']['Neutral_Score']

    # Extract parity values (+/-) from both levels
    # Input: parity ('+' or '-' string in JSON)
    # Output: list of (parity, isTentative) tuples
    parities_1 = [
        (state['parity'], state['isTentativeParity'])
        for state in spin_parity_list_1 if state.get('parity') is not None
    ]
    parities_2 = [
        (state['parity'], state['isTentativeParity'])
        for state in spin_parity_list_2 if state.get('parity') is not None
    ]

    # Guard: If filtering removed all parity values (all were None), return neutral score
    if not parities_1 or not parities_2:
        return Scoring_Config['General']['Neutral_Score']

    # Optimistic matching: Compare all parity pairs and keep the best (maximum) similarity score
    # If ANY parity combination matches, accept two levels match compatibility
    maximum_parity_similarity_score = 0.0
    
    for parity_1, is_tentative_1 in parities_1:
        for parity_2, is_tentative_2 in parities_2:
            # Determine if parities match and if both measurements are firm
            parities_match = (parity_1 == parity_2)
            both_firm = (not is_tentative_1) and (not is_tentative_2)
            
            # Apply scoring rules based on match/mismatch and firmness
            # Match:    → 1.0 (both firm) or 0.9 (any tentative)
            # Mismatch: → 0.0 (both firm) or 0.1 (any tentative)
            
            if parities_match:
                # Same parity → evidence for same level
                if both_firm:
                    pair_similarity = Scoring_Config['Parity']['Match_Firm']
                else:
                    pair_similarity = Scoring_Config['Parity']['Match_Strong']
            else:
                # Different parity → evidence for different levels
                if both_firm:
                    pair_similarity = Scoring_Config['Parity']['Mismatch_Firm']
                else:
                    pair_similarity = Scoring_Config['Parity']['Mismatch_Weak']
            
            # Keep the best score across all parity pairs
            if pair_similarity > maximum_parity_similarity_score:
                maximum_parity_similarity_score = pair_similarity
                
    return maximum_parity_similarity_score


def calculate_gamma_decay_pattern_similarity(gamma_decays_1, gamma_decays_2):
    """
    Compares gamma decay patterns between two nuclear levels to assess match likelihood.
    Returns similarity score from 0.0 (no match) to 1.0 (perfect match).
    
    Example 1: Good Match (Partial Observation vs Complete Dataset)
    ----------------------------------------------------------------
    Dataset A (Complete): Level 2000±1 keV
      - γ1: 2000±2 keV, BR=100
      - γ2: 1400±3 keV, BR=80±10
      - γ3: 900±3 keV, BR=25±5
      - γ4: 600±3 keV, BR=5±2
      - γ5: 300±3 keV, BR=5±2
    
    Dataset B (Partial): Level 2005±2 keV
      - γ1: 2005±2 keV, BR=85±10
      - γ2: 1405±3 keV, BR=100
    
    A vs B:
      1. Normalize: A=[100, 80±10, 25±5, 5±2, 5±2], B=[85±10, 100] (totals: 215 vs 185)
      2. Energy matching: A_γ1↔B_γ1 (ΔE=5, Z=1.8σ ✓), A_γ2↔B_γ2 (ΔE=5, Z=1.2σ ✓)
      3. Intensity check: 100 vs 85±10 (Z=1.5σ compatible), 80±10 vs 100 (Z=2.0σ compatible)
      4. Overlaps: (100+85)/2=92.5, (80+100)/2=90 → total=182.5
      5. Score: 182.5/min(215,185) = 0.986 (98.6% match)
    
    Example 2: Bad Match (Energy Match but Wrong Intensities)
    ----------------------------------------------------------
    Dataset C: Level 1999±2 keV
      - γ1: 899±2 keV, BR=65±10
      - γ2: 598±3 keV, BR=100
    
    A vs C:
      1. Normalize: A=[100, 80±10, 25±5, 5±2, 5±2], C=[65±10, 100] (totals: 215 vs 165)
      2. Energy matching: A_γ3↔C_γ1 (ΔE=1, Z=0.3σ ✓), A_γ4↔C_γ2 (ΔE=2, Z=0.5σ ✓)
      3. Intensity check: 25±5 vs 65±10 (Z=3.6σ inconsistent), 5±2 vs 100 (Z=47.5σ inconsistent)
      4. Overlaps: min(25,65)=25, min(5,100)=5 → total=30
      5. Score: 30/min(215,165) = 0.182 (18.2% match, severe intensity mismatch)
    
    Example 3: Binary Mode (No Intensity Data)
    -------------------------------------------
    Dataset D: Level 2002±2 keV
      - γ1: 902±2 keV (no BR)
      - γ2: 1400±3 keV (no BR)
    
    A vs D:
      1. Energy-only mode activated (D has no intensities)
      2. Energy matching: A_γ3↔D_γ1 (ΔE=2, Z=0.6σ ✓), A_γ2↔D_γ2 (ΔE=0, Z=0 ✓)
      3. Score: 2 matches / min(5,2) = 1.0 (100% match, subset D fully matches subset of A)
    
    Key Features:
    -------------
    - **Subset Matching**: Partial datasets score 1.0 if fully contained in complete dataset
    - **Intensity Consistency**: Uses average if Z<3σ, minimum if inconsistent (penalizes mismatches)
    - **Binary Mode Fallback**: Energy-only matching when intensities missing/unreported
    """
    
    # --- Guard Clause: Missing Data ---
    if not gamma_decays_1 or not gamma_decays_2:
        return Scoring_Config['General']['Neutral_Score']

    # =========================================================================
    # Phase 1: Normalize Intensities (Strongest=100)
    # =========================================================================
    def normalize_gamma_intensities(raw_decays):
        clean_list = []
        has_intensity = False
        maximum_intensity = 0.0
        
        for gamma in raw_decays:
            energy_value = float(gamma['energy'])
            energy_uncertainty = float(gamma['energy_uncertainty'])
            intensity_value = float(gamma.get('intensity', 0))
            intensity_uncertainty = float(gamma.get('intensity_uncertainty', 0))
            
            if energy_value <= 0:
                continue
            
            if intensity_value > 0:
                has_intensity = True
                if intensity_value > maximum_intensity:
                    maximum_intensity = intensity_value
            
            clean_list.append({
                'energy': energy_value, 
                'energy_uncertainty': energy_uncertainty, 
                'intensity': intensity_value, 
                'intensity_uncertainty': intensity_uncertainty
            })
            
        # Normalize strongest gamma to 100.0 (scales uncertainties proportionally)
        if has_intensity and maximum_intensity > 0:
            scale = 100.0 / maximum_intensity
            for gamma in clean_list:
                gamma['intensity'] *= scale
                gamma['intensity_uncertainty'] *= scale
                
        return clean_list, has_intensity

    gamma_list_1, has_intensity_1 = normalize_gamma_intensities(gamma_decays_1)
    gamma_list_2, has_intensity_2 = normalize_gamma_intensities(gamma_decays_2)
    
    if not gamma_list_1 or not gamma_list_2:
        return Scoring_Config['General']['Neutral_Score']

    # =========================================================================
    # Phase 2: Greedy Energy Matching (3.0 Sigma Threshold)
    # =========================================================================
    energy_match_threshold = 3.0
    intensity_match_threshold = 3.0

    # =========================================================================
    # Phase 3: Intensity Mode (Uses branching ratio data) vs Binary Mode (Energy only)
    # =========================================================================
    
    if has_intensity_1 and has_intensity_2:
        # Example: Dataset 1=[100, 80±10, 15, 35], Dataset 2=[85±10, 100]
        # Goal: Match each gamma in 1 to best energy match in 2 (one-to-one)
        
        intersection_sum = 0.0
        matched_indices_2 = set()
        
        for gamma_1 in gamma_list_1:
            # Find best energy match in 2 (smallest Z-score within 3.0 sigma)
            best_match_index = -1
            best_z_score_energy = 999.0
            
            for index, gamma_2 in enumerate(gamma_list_2):
                if index in matched_indices_2:
                    continue
                
                z_score_energy = calculate_z_score(
                    gamma_1['energy'], gamma_1['energy_uncertainty'], 
                    gamma_2['energy'], gamma_2['energy_uncertainty']
                )
                
                if z_score_energy <= energy_match_threshold and z_score_energy < best_z_score_energy:
                    best_z_score_energy = z_score_energy
                    best_match_index = index
            
            if best_match_index != -1:
                gamma_2 = gamma_list_2[best_match_index]
                matched_indices_2.add(best_match_index)
                
                # Intensity consistency check
                z_score_intensity = calculate_z_score(
                    gamma_1['intensity'], gamma_1['intensity_uncertainty'], 
                    gamma_2['intensity'], gamma_2['intensity_uncertainty']
                )
                
                if z_score_intensity <= intensity_match_threshold:
                    # Compatible intensities (e.g., 80±10 vs 100): use average
                    # Example: Pair 2 above: (80+100)/2 = 90
                    overlap = (gamma_1['intensity'] + gamma_2['intensity']) / 2.0
                else:
                    # Inconsistent intensities (e.g., 100±5 vs 5±1): use minimum
                    overlap = min(gamma_1['intensity'], gamma_2['intensity'])
                
                intersection_sum += overlap

        # Subset-robust scoring: normalize by smaller dataset
        # Example: intersection=182.5, min(230,185)=185 → score=182.5/185=0.986
        intensity_sum_1 = sum(gamma['intensity'] for gamma in gamma_list_1)
        intensity_sum_2 = sum(gamma['intensity'] for gamma in gamma_list_2)
        
        if min(intensity_sum_1, intensity_sum_2) == 0:
            return 0.0
        
        similarity_score = intersection_sum / min(intensity_sum_1, intensity_sum_2)
        return min(similarity_score, 1.0)
    
    else:
        # Binary mode: Energy-only matching (no intensity data)
        # Example: When intensities unreported, count energy matches within 3.0 sigma
        matches_count = 0
        matched_indices_2 = set()
        
        for gamma_1 in gamma_list_1:
            for index, gamma_2 in enumerate(gamma_list_2):
                if index in matched_indices_2: continue
                
                z_score_energy = calculate_z_score(
                    gamma_1['energy'], gamma_1['energy_uncertainty'], 
                    gamma_2['energy'], gamma_2['energy_uncertainty']
                )
                
                if z_score_energy <= energy_match_threshold:
                    matches_count += 1
                    matched_indices_2.add(index)
                    break
        
        # Subset-robust: matches_count / smaller_dataset_size
        minimum_length = min(len(gamma_list_1), len(gamma_list_2))
        if minimum_length == 0: return 0.0
        
        return float(matches_count) / float(minimum_length)


def extract_features(level_1, level_2):
    """
    Constructs five-dimensional feature vector for a pair of levels.
    
    Feature Vector Output: [Energy_Similarity, Spin_Similarity, Parity_Similarity, Specificity, Gamma_Decay_Pattern_Similarity]
    
    Design Principle:
    - All features are monotonic increasing: higher value strictly indicates better match likelihood.
    - All feature scores are drawn from Scoring_Config for full configurability and physics tuning.
    
    Feature Definitions (all scores sourced from Scoring_Config):
    
    1. Energy_Similarity (0-1):
       - Gaussian kernel: exp(-Sigma_Scale × z²) where z = ΔE/σ_combined
       - Sigma_Scale from Scoring_Config['Energy']['Sigma_Scale'] controls energy strictness (default 0.1 = lenient)
       - Default_Uncertainty from Scoring_Config['Energy']['Default_Uncertainty'] applied when data missing
       - Incorporates measurement uncertainties from both levels
       - Monotonic: Energy overlap increases → similarity increases
       - Independent of spin/parity information
    
    2. Spin_Similarity (0-1):
       - Optimistic matching: maximum compatibility across all spin pair combinations
       - All scoring thresholds come from Scoring_Config['Spin']:
         * ΔJ=0, both firm: Scoring_Config['Spin']['Match_Firm'] (typically 1.0)
         * ΔJ=0, any tentative: Scoring_Config['Spin']['Match_Strong'] (typically 0.8)
         * ΔJ=1, both firm: Scoring_Config['Spin']['Mismatch_Strong'] (typically 0.05)
         * ΔJ=1, any tentative: Scoring_Config['Spin']['Mismatch_Weak'] (typically 0.2)
         * ΔJ≥2: Scoring_Config['Spin']['Mismatch_Firm'] (typically 0.0)
       - Guard: Uses Scoring_Config['General']['Neutral_Score'] (typically 0.5) when spin data missing
       - Rationale: If any spin option matches, levels likely identical (optimistic strategy)
    
    3. Parity_Similarity (0-1):
       - Optimistic matching: maximum compatibility across all parity pair combinations
       - All scoring thresholds come from Scoring_Config['Parity']:
         * Same parity, both firm: Scoring_Config['Parity']['Match_Firm'] (typically 1.0)
         * Same parity, any tentative: Scoring_Config['Parity']['Match_Strong'] (typically 0.8)
         * Opposite parity, both firm: Scoring_Config['Parity']['Mismatch_Firm'] (typically 0.0)
         * Opposite parity, any tentative: Scoring_Config['Parity']['Mismatch_Weak'] (typically 0.05)
       - Guard: Uses Scoring_Config['General']['Neutral_Score'] (typically 0.5) when parity data missing
       - Rationale: If any parity option matches, levels likely identical (optimistic strategy)
    
    4. Specificity (0-1):
       - Ambiguity penalty: 1/f(multiplicity) where multiplicity = options_count_1 × options_count_2
       - Formula selection from Scoring_Config['Specificity']['Formula']:
         * 'sqrt' (default): specificity = 1/sqrt(multiplicity) - balanced penalty
         * 'linear': specificity = 1/multiplicity - aggressive penalty
         * 'log': specificity = 1/(1+log10(multiplicity)) - gentle penalty
         * 'tunable': specificity = 1/(1+Alpha×(multiplicity-1)) where Alpha from Scoring_Config['Specificity']['Alpha']
       - Single well-resolved level (1 option): 1.0 (no penalty)
       - Ambiguous levels (3×3=9 options): 0.33 using sqrt (67% penalty)
       - Rationale: Measurement ambiguity reduces confidence in match assessment
    
    5. Gamma_Decay_Pattern_Similarity (0-1):
       - Method: Subset-Robust Statistical Compatibility.
       - Logic: 
         a) Energy Matching: Greedy one-to-one matching using Z-score <= 3.0.
         b) Intensity Verification: 
            *   If Z_intensity <= 3.0 (Consistent): Uses Average intensity (statistical fluctuation assumption).
            *   If Z_intensity > 3.0 (Inconsistent): Uses Minimum intensity (penalize discrepancy).
         c) Subset Handling: Divides match sum by the *smaller* dataset's total intensity.
            Allows a small set of gammas (subset B) to perfectly match a larger set (complete A).
         d) Binary Mode: If intensity data missing, falls back to energy-only match counting.
       - Guard: Uses Scoring_Config['General']['Neutral_Score'] (typically 0.5) when gamma data missing.
       - Rationale: Robustly handles "observed subset" vs "complete dataset" scenarios.
    
    """

    energy_similarity = calculate_energy_similarity(
        level_1['energy_value'], level_1['energy_uncertainty'],
        level_2['energy_value'], level_2['energy_uncertainty']
    )
    
    spin_parity_list_1 = level_1.get('spin_parity_list', [])
    spin_parity_list_2 = level_2.get('spin_parity_list', [])

    spin_similarity = calculate_spin_similarity(spin_parity_list_1, spin_parity_list_2)
    parity_similarity = calculate_parity_similarity(spin_parity_list_1, spin_parity_list_2)

    # Feature 4: Specificity score (penalize ambiguous multiple options)
    # Configurable formula via Scoring_Config['Specificity']['Formula']
    # All formulas: specificity = f(multiplicity) where multiplicity = options_count_1 × options_count_2
    #
    # Physical interpretation: Ambiguity penalty scales with combination space.
    # Square root naturally represents uncertainty growth in quantum measurements.
    #
    # ENSDF spin option counts (typical real data):
    #   Single:   "2+" → 1 option
    #   Double:   "1/2+,3/2+" or "(1,2)+" → 2 options
    #   Multiple: "1/2,3/2,5/2" → 3 options
    #   Multiple:  "1/2:11/2" expand to "1/2,3/2,5/2,7/2,9/2,11/2" 6 options
    #
    # Formula Options (for multiplicity=9, both levels have 3 options):
    #   'sqrt':    1/sqrt(mult) = 33% (67% penalty) - BALANCED, current default
    #   'linear':  1/mult       = 11% (89% penalty) - AGGRESSIVE
    #   'log':     1/(1+log10(mult)) = 51% (49% penalty) - GENTLE
    #   'tunable': 1/(1+Alpha*(mult-1)) where Alpha from config - CUSTOMIZABLE
    #              Example: Alpha=0.5 gives mult=9→18% (82% penalty)
    #
    # Specificity penalty comparison table:
    # Multiplicity | sqrt   | linear | log    | tunable(α=0.5)
    # -------------|--------|--------|--------|---------------
    # 1            | 100%   | 100%   | 100%   | 100%
    # 2            | 71%    | 50%    | 77%    | 67%
    # 4            | 50%    | 25%    | 62%    | 40%
    # 9            | 33%    | 11%    | 51%    | 20%
    options_count_1 = len(spin_parity_list_1) if spin_parity_list_1 else 1
    options_count_2 = len(spin_parity_list_2) if spin_parity_list_2 else 1
    multiplicity = max(1, options_count_1 * options_count_2)
    
    formula = Scoring_Config['Specificity']['Formula']
    if formula == 'sqrt':
        specificity = 1.0 / np.sqrt(multiplicity)
    elif formula == 'linear':
        specificity = 1.0 / multiplicity
    elif formula == 'log':
        specificity = 1.0 / (1.0 + np.log10(multiplicity))
    elif formula == 'tunable':
        alpha = Scoring_Config['Specificity']['Alpha']
        specificity = 1.0 / (1.0 + alpha * (multiplicity - 1))
    else:
        raise ValueError(f"Unknown Specificity formula: {formula}. Valid options: 'sqrt', 'linear', 'log', 'tunable'")

    # ===== DISABLED CERTAINTY FEATURES (easily re-enable if needed) =====
    # Rationale: Certainty is redundant - already encoded in similarity scores:
    #   - Match: both firm → 1.0, any tentative → 0.9 (certainty fully predictable)
    #   - Mismatch: certainty adds independent signal but creates feature redundancy
    # To re-enable: uncomment lines below, change return to 6D vector, update training data
    #
    # spin_is_tentative_1 = any(state['isTentativeSpin'] for state in spin_parity_list_1 if 'isTentativeSpin' in state)
    # spin_is_tentative_2 = any(state['isTentativeSpin'] for state in spin_parity_list_2 if 'isTentativeSpin' in state)
    # spin_certainty = 0.0 if (spin_is_tentative_1 or spin_is_tentative_2) else 1.0
    #
    # parity_is_tentative_1 = any(state['isTentativeParity'] for state in spin_parity_list_1 if 'isTentativeParity' in state)
    # parity_is_tentative_2 = any(state['isTentativeParity'] for state in spin_parity_list_2 if 'isTentativeParity' in state)
    # parity_certainty = 0.0 if (parity_is_tentative_1 or parity_is_tentative_2) else 1.0
    # ===================================================================
    
    # Feature 5: Gamma decay pattern similarity
    gamma_decays_1 = level_1.get('gamma_decays', [])
    gamma_decays_2 = level_2.get('gamma_decays', [])
    gamma_decay_pattern_similarity = calculate_gamma_decay_pattern_similarity(gamma_decays_1, gamma_decays_2)
    
    return np.array([energy_similarity, spin_similarity, parity_similarity, specificity, gamma_decay_pattern_similarity])

def generate_synthetic_training_data():
    """
    Generates synthetic training data for XGBoost using the defined Scoring_Config.
    
    Strategy:
    - Uses a grid-based approach to cover all physics scenarios defined in Scoring_Config.
    - Applies a unified physics logic function (calculate_label) rather than hardcoded edge cases.
    - Implements Feature Correlation: Perfect spin/parity can "rescue" mediocre energy similarity.
    - Ensures the model learns:
        1. Vetoes: Mismatches (Score ~0.0) -> Label 0.0.
        2. Neutrality: Unknown physics (Score 0.5) -> High dependence on Energy.
        3. Match Quality: Higher scores -> Higher probability.
        4. Gamma Decay: Additional confirmation when decay patterns match.
        5. Feature Correlation: Perfect physics (Spin+Parity) relaxes energy strictness.
    
    Feature Correlation Physics Rationale:
    - If two levels have IDENTICAL quantum numbers (firm Jπ match) but disagreeing energies,
      the energy mismatch may be due to calibration error, not genuine mismatch.
    - Similarly, if two levels exhibit IDENTICAL gamma decay patterns (Fingerprint match),
      they are likely the same physical state despite energy value offsets.
    - XGBoost learns this interaction: when (Spin+Parity is Perfect) OR (Gamma is Perfect),
      the model assigns higher probability even with moderate energy_similarity (e.g., 0.4-0.7).
    """
    training_points = []
    
    # 1. Gather all possible discrete scores from Config
    spin_scores = list(Scoring_Config['Spin'].values()) + [Scoring_Config['General']['Neutral_Score']]
    parity_scores = list(Scoring_Config['Parity'].values()) + [Scoring_Config['General']['Neutral_Score']]
    gamma_scores = [0.0, 0.3, Scoring_Config['General']['Neutral_Score'], 0.7, 0.85, 1.0]  # Standardized 0.85 for rescue threshold
    
    # 2. Feature Correlation Configuration
    correlation_enabled = Scoring_Config['Feature_Correlation']['Enabled']
    correlation_threshold = Scoring_Config['Feature_Correlation']['Threshold']
    rescue_exponent = Scoring_Config['Feature_Correlation']['Rescue_Exponent']
    
    # 3. Define Physics Logic for Labeling (with Feature Correlation)
    def calculate_label(energy_similarity, spin_similarity, parity_similarity, specificity, gamma_similarity):
        # HARD VETO: If physics explicitly mismatches (score <= near-zero), reject.
        # Allow small tolerance if config uses 0.05 for mismatched.
        if spin_similarity <= Scoring_Config['Spin'].get('Mismatch_Strong', 0.05) or \
           parity_similarity <= Scoring_Config['Parity'].get('Mismatch_Weak', 0.1) / 2:
            return 0.0

        # ADJUSTMENT FOR NEUTRAL/TENTATIVE:
        # Map raw scores to "Confidence Factors" to prevent multiplicative decay from killing good matches.
        # If Score is Neutral (0.5), treat it as "No Info" -> Factor ~0.85 (slight penalty vs perfect).
        def calculate_confidence_factor(value):
            if value == Scoring_Config['General']['Neutral_Score']:
                return 0.85
            return value

        spin_factor = calculate_confidence_factor(spin_similarity)
        parity_factor = calculate_confidence_factor(parity_similarity)
        gamma_factor = calculate_confidence_factor(gamma_similarity)

        # Feature Correlation: Physics Rescue
        # Condition 1: Perfect Quantum Numbers (Spin/Parity >= Threshold)
        # Condition 2: Strong Gamma Decay Fingerprint (Gamma >= Threshold)
        # Physics rationale: If internal structure (Gamma/Spin/Parity) matches strongly,
        # energy disagreement is likely due to calibration. We "rescue" the energy score
        # by boosting it (reducing the penalty) rather than using the raw low score.
        # NOTE: Rescue != Firm Match. It prevents a 0% veto (Survival), allowing the candidate
        # to remain in consideration. It typically raises probability from <1% to ~15-25%
        # for poor energy matches, which may pass the pairwise filter but not yet cluster.
        effective_energy = energy_similarity
        if correlation_enabled:
            is_quantum_match = (spin_similarity >= correlation_threshold and parity_similarity >= correlation_threshold)
            is_gamma_match = (gamma_similarity >= correlation_threshold)

            if is_quantum_match or is_gamma_match:
                 # Apply rescue: energy → energy^exponent (e.g., sqrt transforms 0.04→0.2, 0.4→0.63)
                 effective_energy = energy_similarity ** rescue_exponent

        # PROBABILITY FORMULA:
        # Base confidence from Energy * Physics Quality * Specificity
        # Use geometric mean of physics factors to balance them.
        physics_confidence = np.sqrt(spin_factor * parity_factor * gamma_factor)
        
        probability = effective_energy * physics_confidence * specificity
        return min(max(probability, 0.0), 0.99)

    # 4. Systematic Grid Generation (Cover the feature space)
    # Energy grid: dense near 0 (mismatch) and 1 (match)
    energy_grid = np.concatenate([
        np.linspace(0.0, 0.3, 10),  # Low energy (mismatch zone)
        np.linspace(0.3, 0.8, 10),  # Mid energy
        np.linspace(0.8, 1.0, 20)   # High energy (match zone)
    ])

    for energy in energy_grid:
        for spin in set(spin_scores):
            for parity in set(parity_scores):
                # Specificity mostly 1.0, but add some noise/variation
                for specificity in [1.0, 0.7, 0.5]:
                    for gamma in gamma_scores:
                        label = calculate_label(energy, spin, parity, specificity, gamma)
                        training_points.append(([energy, spin, parity, specificity, gamma], label))

    # 5. Feature Correlation Teaching Set (Critical for learning the interaction)
    # Explicitly oversample the region where energy is mediocre but physics is perfect.
    # This teaches XGBoost to split on Spin/Parity/Gamma first, then apply different energy slopes.
    perfect_spin = Scoring_Config['Spin']['Match_Firm']
    perfect_parity = Scoring_Config['Parity']['Match_Firm']
    imperfect_spin = Scoring_Config['Spin']['Match_Strong']  # e.g., 0.8 (tentative match)
    imperfect_parity = Scoring_Config['Parity']['Match_Strong']  # e.g., 0.8 (tentative match)
    neutral_score = Scoring_Config['General']['Neutral_Score']
    rescue_threshold = Scoring_Config['Feature_Correlation']['Threshold']
    
    # Mediocre AND Low energy range where correlation effect is most visible
    # Validation Note: Added low values (0.05, 0.1) to ensure model learns rescue even at poor energy matches.
    mediocre_energy_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for energy in mediocre_energy_values:
        # Case A: Perfect Quantum Numbers -> Label should be BOOSTED (rescued)
        # Model learns: [mediocre energy, perfect spin, perfect parity] -> high probability
        label_boosted_quantum_numbers = calculate_label(energy, perfect_spin, perfect_parity, 1.0, neutral_score)
        training_points.append(([energy, perfect_spin, perfect_parity, 1.0, neutral_score], label_boosted_quantum_numbers))
        
        # Case B: Perfect Gamma Fingerprint -> Label should be BOOSTED (rescued)
        # Model learns: [mediocre energy, unknown spin, unknown parity, Match Gamma] -> high probability
        # Teach both Perfect (1.0) and Threshold (0.85) cases
        label_boosted_gamma = calculate_label(energy, neutral_score, neutral_score, 1.0, 1.0)
        training_points.append(([energy, neutral_score, neutral_score, 1.0, 1.0], label_boosted_gamma))
        
        # Explicitly teach the threshold value (0.85) to ensure the cut-off is learned
        label_boosted_gamma_threshold = calculate_label(energy, neutral_score, neutral_score, 1.0, rescue_threshold)
        training_points.append(([energy, neutral_score, neutral_score, 1.0, rescue_threshold], label_boosted_gamma_threshold))

        # Case C: Imperfect Physics -> Label should be STANDARD (no rescue)
        # Model learns: [mediocre energy, imperfect physics] -> standard (lower) probability
        # This CONTRAST forces the tree to split on Physics/Gamma condition
        label_standard = calculate_label(energy, imperfect_spin, imperfect_parity, 1.0, neutral_score)
        training_points.append(([energy, imperfect_spin, imperfect_parity, 1.0, neutral_score], label_standard))
        
        # Case D: Mixed Physics (one perfect, one imperfect) -> No rescue
        # Ensures rescue only triggers when BOTH spin AND parity are perfect (unless Gamma saves it)
        label_mixed_1 = calculate_label(energy, perfect_spin, imperfect_parity, 1.0, neutral_score)
        training_points.append(([energy, perfect_spin, imperfect_parity, 1.0, neutral_score], label_mixed_1))
        label_mixed_2 = calculate_label(energy, imperfect_spin, perfect_parity, 1.0, neutral_score)
        training_points.append(([energy, imperfect_spin, perfect_parity, 1.0, neutral_score], label_mixed_2))

    # 6. Random Sampling (Fill gaps and prevent overfitting)
    np.random.seed(42)
    for _ in range(500):
        energy = np.random.uniform(0, 1)
        spin = np.random.choice(list(set(spin_scores)))
        parity = np.random.choice(list(set(parity_scores)))
        # Generate specificity roughly following 1/sqrt(N) distribution
        specificity = np.random.choice([1.0, 0.707, 0.577, 0.5, 0.4])
        gamma = np.random.choice(gamma_scores)
        
        label = calculate_label(energy, spin, parity, specificity, gamma)
        training_points.append(([energy, spin, parity, specificity, gamma], label))

    # 7. Additional Boost for "Perfect" cases to ensure 1.0-like behavior
    # Forces model to be very confident when everything is perfect
    for _ in range(50):
        energy = np.random.uniform(0.95, 1.0)
        spin = Scoring_Config['Spin']['Match_Firm']
        parity = Scoring_Config['Parity']['Match_Firm']
        specificity = 1.0
        gamma = np.random.choice([Scoring_Config['General']['Neutral_Score'], 0.9, 1.0])
        training_points.append(([energy, spin, parity, specificity, gamma], 0.99))

    training_features = np.array([record[0] for record in training_points])
    training_labels = np.array([record[1] for record in training_points])
    
    return training_features, training_labels


