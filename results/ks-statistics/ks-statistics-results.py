import json
import numpy as np
from typing import List, Dict, Any
import math
import argparse


# Include the KS functions from previous code
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


def analyze_ks_from_json(filepath: str) -> Dict[str, Any]:
    """
    Analyze K-S test results from JSON file containing orig_lp and induced_lp values.

    Parameters:
    -----------
    filepath : str
        Path to the JSON file

    Returns:
    --------
    Dict containing analysis results
    """
    # Read the data
    data = read_json_file(filepath)

    if not data:
        raise ValueError("No data found in file")

    print(f"Loaded {len(data)} samples from {filepath}")
    print(f"Sample data structure: {data[0]}")

    # Extract the two distributions
    orig_lp_values = [item['orig_lp'] for item in data]
    induced_lp_values = [item['induced_lp'] for item in data]

    # Basic statistics
    orig_lp_array = np.array(orig_lp_values)
    induced_lp_array = np.array(induced_lp_values)

    print(f"\nOriginal LP distribution:")
    print(f"  Mean: {np.mean(orig_lp_array):.4f}")
    print(f"  Std:  {np.std(orig_lp_array):.4f}")
    print(f"  Min:  {np.min(orig_lp_array):.4f}")
    print(f"  Max:  {np.max(orig_lp_array):.4f}")

    print(f"\nInduced LP distribution:")
    print(f"  Mean: {np.mean(induced_lp_array):.4f}")
    print(f"  Std:  {np.std(induced_lp_array):.4f}")
    print(f"  Min:  {np.min(induced_lp_array):.4f}")
    print(f"  Max:  {np.max(induced_lp_array):.4f}")

    # Calculate K-S statistic and p-value
    ks_stat = ks_statistic(orig_lp_values, induced_lp_values)
    n1, n2 = len(orig_lp_values), len(induced_lp_values)
    p_value = ks_pvalue(ks_stat, n1, n2)

    print(f"\nKolmogorov-Smirnov Test Results:")
    print(f"  K-S Statistic: {ks_stat:.6f}")
    print(f"  P-value:       {p_value:.6f}")
    print(f"  Sample sizes:  n1={n1}, n2={n2}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"\n  *** SIGNIFICANT DIFFERENCE ***")
        print(f"  The distributions are significantly different (p < {alpha})")
    else:
        print(f"\n  No significant difference found (p >= {alpha})")

    # If delta values are provided, also analyze them
    if 'delta' in data[0]:
        delta_values = [item['delta'] for item in data]
        delta_array = np.array(delta_values)

        print(f"\nDelta values analysis:")
        print(f"  Mean delta: {np.mean(delta_array):.6f}")
        print(f"  Std delta:  {np.std(delta_array):.6f}")
        print(f"  Max delta:  {np.max(delta_array):.6f}")

        # If deltas are individual K-S statistics, we could combine them
        # using Fisher's method or similar, but this is less common

    return {
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'n_samples': len(data),
        'orig_lp_stats': {
            'mean': float(np.mean(orig_lp_array)),
            'std': float(np.std(orig_lp_array)),
            'min': float(np.min(orig_lp_array)),
            'max': float(np.max(orig_lp_array))
        },
        'induced_lp_stats': {
            'mean': float(np.mean(induced_lp_array)),
            'std': float(np.std(induced_lp_array)),
            'min': float(np.min(induced_lp_array)),
            'max': float(np.max(induced_lp_array))
        }
    }


def quick_ks_test_from_json(filepath: str) -> tuple:
    """
    Quick function to get just the K-S statistic and p-value.

    Returns:
    --------
    tuple: (ks_statistic, p_value)
    """
    data = read_json_file(filepath)
    orig_lp = [item['orig_lp'] for item in data]
    induced_lp = [item['induced_lp'] for item in data]

    ks_stat = ks_statistic(orig_lp, induced_lp)
    p_val = ks_pvalue(ks_stat, len(orig_lp), len(induced_lp))

    return ks_stat, p_val


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str, required=True)
    args = parser.parse_args()  # Parse the command-line arguments
    filepath = args.json_file  # Access the value of the --json-file argument

    try:
        # Full analysis
        results = analyze_ks_from_json(filepath)
        print(results)
        # ks_stat, p_value = quick_ks_test_from_json(filepath)
        # print(f"K-S Statistic: {ks_stat:.6f}, P-value: {p_value:.6f}")
    except Exception as e:
        print(f"Error during analysis: {e}")