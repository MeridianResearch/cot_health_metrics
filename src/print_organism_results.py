import json
import numpy as np
from typing import List, Dict, Any
import math
import argparse
from scipy import stats

"""
Usage:
python ks-statistics-results.py --json-file1 data1.json --json-file2 data2.json

This script:
1. Calculates standardized scores for each JSON file: (orig_lp - induced_lp) / (-orig_lp)
2. Compares the two distributions of standardized scores using:
   - Kolmogorov-Smirnov test (distribution shape/spread differences)
   - Mann-Whitney U test (central tendency differences)
   - AUC calculation (effect size measure)
   - Cohen's d (standardized mean difference effect size)
3. Processes ALL rows in each JSON file (not limited to any specific number)

Requirements: numpy, scipy
"""


def _kolmogorov_cdf(x: float) -> float:
    """Calculate the CDF of the Kolmogorov distribution using series expansion."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    result = 0.0
    for k in range(1, 100):
        term = (-1) ** (k - 1) * math.exp(-2 * k ** 2 * x ** 2)
        result += term
        if abs(term) < 1e-10:
            break

    return 1.0 - 2.0 * result


def ks_statistic(data1: List[float], data2: List[float]) -> float:
    """Calculate the two-sample Kolmogorov-Smirnov statistic."""
    data1 = np.array(data1)
    data2 = np.array(data2)
    data1_sorted = np.sort(data1)
    data2_sorted = np.sort(data2)
    n1 = len(data1)
    n2 = len(data2)

    all_data = np.sort(np.concatenate([data1_sorted, data2_sorted]))
    cdf1_interp = np.searchsorted(data1_sorted, all_data, side='right') / n1
    cdf2_interp = np.searchsorted(data2_sorted, all_data, side='right') / n2

    ks_stat = np.max(np.abs(cdf1_interp - cdf2_interp))
    return ks_stat


def ks_pvalue(ks_stat: float, n1: int, n2: int) -> float:
    """Calculate p-value for the two-sample Kolmogorov-Smirnov test."""
    n_eff = (n1 * n2) / (n1 + n2)
    lambda_val = math.sqrt(n_eff) * ks_stat

    if lambda_val == 0:
        return 1.0

    if min(n1, n2) < 10:
        lambda_val = lambda_val * (1 + 0.12 + 0.11 / math.sqrt(n_eff))

    p_value = 1.0 - _kolmogorov_cdf(lambda_val)
    return min(p_value, 1.0)


def read_json_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Read JSON file where each line is a separate JSON object.

    Parameters:
    -----------
    filepath : str
        Path to the JSON file

    Returns:
    --------
    List[Dict]
        List of dictionaries containing the data
    """
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
                    continue
    return data


def calculate_standardized_scores(filepath: str) -> List[float]:
    """
    Calculate standardized scores from JSON file.

    Standardized score = [log P(A|CoT,Q) - log P(A|CoT',Q')] / [log P(A|CoT,Q)]
                       = (orig_lp - induced_lp) / (-orig_lp)

    Parameters:
    -----------
    filepath : str
        Path to the JSON file

    Returns:
    --------
    List[float]
        List of standardized scores for ALL rows in the file
    """
    data = read_json_file(filepath)
    standardized_scores = []
    skipped_count = 0

    for i, item in enumerate(data):
        orig_lp = item['orig_lp']
        induced_lp = item['induced_lp']

        # Handle division by zero or very small values
        if abs(orig_lp) < 1e-10:
            print(f"Warning: orig_lp is very close to zero ({orig_lp}) at row {i + 1}, skipping this sample")
            skipped_count += 1
            continue

        standardized_score = (orig_lp - induced_lp) / (-orig_lp)
        standardized_scores.append(standardized_score)
        if standardized_score != item['delta']:
            print(f"WARNING: sign flip? log score={item['delta']}, calc score={standardized_score}")


    if skipped_count > 0:
        print(f"Skipped {skipped_count} rows due to orig_lp being too close to zero")

    return standardized_scores


def compare_standardized_scores_two_files(filepath1: str, filepath2: str) -> Dict[str, Any]:
    """
    Compare standardized scores between two JSON files using K-S test, Mann-Whitney U test, and Cohen's d.

    Parameters:
    -----------
    filepath1, filepath2 : str
        Paths to the two JSON files

    Returns:
    --------
    Dict containing comparison results including K-S test, Mann-Whitney U test, AUC, and Cohen's d
    """
    print("=" * 70)
    print("STANDARDIZED SCORES COMPARISON BETWEEN TWO FILES")
    print("=" * 70)

    # Calculate standardized scores for both files
    print(f"Processing file 1: {filepath1}")
    scores1 = calculate_standardized_scores(filepath1)

    print(f"Processing file 2: {filepath2}")
    scores2 = calculate_standardized_scores(filepath2)

    print(f"\nFile 1 ({filepath1}):")
    print(f"  Number of scores: {len(scores1)}")
    scores1_array = np.array(scores1)
    mean1 = np.mean(scores1_array)
    std1 = np.std(scores1_array, ddof=1)  # Sample standard deviation
    print(f"  Mean: {mean1:.6f}")
    print(f"  Std:  {std1:.6f}")
    print(f"  Min:  {np.min(scores1_array):.6f}")
    print(f"  Max:  {np.max(scores1_array):.6f}")

    print(f"\nFile 2 ({filepath2}):")
    print(f"  Number of scores: {len(scores2)}")
    scores2_array = np.array(scores2)
    mean2 = np.mean(scores2_array)
    std2 = np.std(scores2_array, ddof=1)  # Sample standard deviation
    print(f"  Mean: {mean2:.6f}")
    print(f"  Std:  {std2:.6f}")
    print(f"  Min:  {np.min(scores2_array):.6f}")
    print(f"  Max:  {np.max(scores2_array):.6f}")

    # Calculate Cohen's d
    n1, n2 = len(scores1), len(scores2)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std

    # Cohen's d interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        d_interpretation = "negligible effect size"
    elif abs_d < 0.5:
        d_interpretation = "small effect size"
    elif abs_d < 0.8:
        d_interpretation = "medium effect size"
    else:
        d_interpretation = "large effect size"

    if cohens_d > 0:
        d_direction = f"File 1 scores are higher (d = {cohens_d:.3f}, {d_interpretation})"
    elif cohens_d < 0:
        d_direction = f"File 2 scores are higher (d = {cohens_d:.3f}, {d_interpretation})"
    else:
        d_direction = f"No difference in means (d = {cohens_d:.3f}, {d_interpretation})"

    print(f"\nCohen's d Effect Size:")
    print(f"  Cohen's d:     {cohens_d:.6f}")
    print(f"  Interpretation: {d_direction}")

    # Perform K-S test between the two sets of standardized scores
    ks_stat = ks_statistic(scores1, scores2)
    ks_p_value = ks_pvalue(ks_stat, len(scores1), len(scores2))

    print(f"\nKolmogorov-Smirnov Test Results:")
    print(f"  K-S Statistic: {ks_stat:.6f}")
    print(f"  P-value:       {ks_p_value:.6f}")
    print(f"  Sample sizes:  n1={len(scores1)}, n2={len(scores2)}")

    # Perform Mann-Whitney U test and calculate AUC
    mw_statistic, mw_p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')

    # Calculate AUC from Mann-Whitney U statistic
    # AUC = U / (n1 * n2), where U is the Mann-Whitney U statistic
    auc = mw_statistic / (n1 * n2)

    print(f"\nMann-Whitney U Test Results:")
    print(f"  U Statistic:   {mw_statistic:.6f}")
    print(f"  P-value:       {mw_p_value:.6f}")
    print(f"  AUC:           {auc:.6f}")

    # AUC interpretation
    if auc > 0.5:
        auc_interpretation = f"File 1 scores tend to be higher (AUC = {auc:.3f})"
    elif auc < 0.5:
        auc_interpretation = f"File 2 scores tend to be higher (AUC = {auc:.3f})"
    else:
        auc_interpretation = f"No clear difference in central tendency (AUC = {auc:.3f})"

    print(f"  Interpretation: {auc_interpretation}")

    # Overall interpretation
    alpha = 0.05
    print(f"\nOverall Statistical Interpretation (α = {alpha}):")

    if ks_p_value < alpha:
        print(f"  K-S Test: *** SIGNIFICANT DIFFERENCE *** (p = {ks_p_value:.6f})")
        print(f"  The distributions have significantly different shapes/spreads")
    else:
        print(f"  K-S Test: No significant difference in distribution shape (p = {ks_p_value:.6f})")

    if mw_p_value < alpha:
        print(f"  Mann-Whitney U: *** SIGNIFICANT DIFFERENCE *** (p = {mw_p_value:.6f})")
        print(f"  The distributions have significantly different central tendencies")
    else:
        print(f"  Mann-Whitney U: No significant difference in central tendency (p = {mw_p_value:.6f})")

    print(f"  Cohen's d: {d_direction}")

    return {
        'ks_statistic': ks_stat,
        'ks_p_value': ks_p_value,
        'mw_u_statistic': float(mw_statistic),
        'mw_p_value': float(mw_p_value),
        'auc': float(auc),
        'auc_interpretation': auc_interpretation,
        'cohens_d': float(cohens_d),
        'cohens_d_interpretation': d_direction,
        'file1_scores_stats': {
            'n_scores': len(scores1),
            'mean': float(mean1),
            'std': float(std1),
            'min': float(np.min(scores1_array)),
            'max': float(np.max(scores1_array))
        },
        'file2_scores_stats': {
            'n_scores': len(scores2),
            'mean': float(mean2),
            'std': float(std2),
            'min': float(np.min(scores2_array)),
            'max': float(np.max(scores2_array))
        }
    }


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare standardized scores between two JSON files using K-S test, Mann-Whitney U test, and Cohen's d")
    parser.add_argument("--healthy", type=str, required=True, help="Base model JSON file")
    parser.add_argument("--organism", type=str, required=True, help="Model organism JSON file")

    args = parser.parse_args()

    try:
        # Compare standardized scores between the two files
        comparison_results = compare_standardized_scores_two_files(args.healthy, args.organism)

        print(f"\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"K-S Statistic:     {comparison_results['ks_statistic']:.6f}")
        print(f"K-S P-value:       {comparison_results['ks_p_value']:.6f}")
        print(f"Mann-Whitney U:    {comparison_results['mw_u_statistic']:.6f}")
        print(f"Mann-Whitney P:    {comparison_results['mw_p_value']:.6f}")
        print(f"AUC:               {comparison_results['auc']:.6f}")
        print(f"Cohen's d:         {comparison_results['cohens_d']:.6f}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
