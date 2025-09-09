from datasets import load_dataset
import numpy as np
from datetime import datetime
import json
import math
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union
import math


# Format as string

def get_datetime_str():
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(datetime_str)
    return datetime_str

#!/usr/bin/env python3
import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_cot_effect(directory: str, target_m2: str):
    """
    Scans JSONL files in `directory` named like
      output_logprobs_{m1}_{target_m2}.jsonl
    and plots the mean log-probability of M2 producing the correct
    answer with vs. without chain-of-thought from each M1.
    """
    pattern = os.path.join(directory, f'output_logprobs_*_{target_m2}.jsonl')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern {pattern}")

    m1_names = []
    no_cot_means = []
    with_cot_means = []

    # Parse each file
    for filepath in files:
        filename = os.path.basename(filepath)
        m1 = filename.replace(f'output_logprobs_', '').replace(f'_{target_m2}.jsonl', '')
        m1_names.append(m1)

        qa_vals = []
        qr_vals = []
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                qa_vals.append(data['logprobsM2_QA1_sum'])
                qr_vals.append(data['logprobsM2_QR1A1_sum'])

        no_cot_means.append(np.mean(qa_vals))
        with_cot_means.append(np.mean(qr_vals))

    # Plotting
    x = np.arange(len(m1_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, no_cot_means, width, label='Without CoT')
    ax.bar(x + width/2, with_cot_means, width, label='With CoT')

    ax.set_xticks(x)
    ax.set_xticklabels(m1_names, rotation=45, ha='right')
    ax.set_ylabel('Average Mean Log Probability')
    ax.set_title(f'{target_m2}: Answer Log Probs With vs. Without CoT')
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/qwen06b/cot.png")

def main():
    parser = argparse.ArgumentParser(
        description="Plot CoT effect for Qwen3-0.6B across different M1 models"
    )
    parser.add_argument(
        "directory",
        help="Path to folder containing the JSONL files"
    )
    parser.add_argument(
        "--model2",
        default="Qwen3-0.6B",
        help="Target small model name (default: Qwen3-0.6B)"
    )
    args = parser.parse_args()
    plot_cot_effect(args.directory, args.model2)


if __name__ == "__main__":
    main()



# plot_logprobs_histograms(jsonl_path,output_dir)