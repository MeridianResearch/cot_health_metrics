from datasets import load_dataset
import numpy as np
from datetime import datetime
import json
import math
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from enum import Enum
class SupportedModel(Enum):
    QWEN3_0_6B             = "Qwen/Qwen3-0.6B"
    QWEN3_1_7B             = "Qwen/Qwen3-1.7B"
    DEEPSEEK_QWEN_1_5B     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    MINI_THINK_BASE_1B     = "Wladastic/Mini-Think-Base-1B"
    GEMMA_2_2B             = "google/gemma-2-2b"
    # COGITO_LLAMA_3B      = "deepcogito/cogito-v1-preview-llama-3B"  # unverified
    # PHI_2                = "microsoft/phi-2"



# Format as string

def get_datetime_str():
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(datetime_str)
    return datetime_str




# def plot_logprobs_histograms(
#     jsonl_path: str,
#     out_dir: str = "plots",
#     bins: int = 50,
#     prefix: str = "metric",
#     focus_percentile_range: tuple = (2, 98)
# ):
#     """
#     Loads float fields from a JSONL file and plots a histogram for each,
#     with auto-focusing on dense value regions.
# 
#     Args:
#         jsonl_path: path to JSONL file
#         out_dir: directory to save plots
#         bins: number of histogram bins
#         prefix: file prefix for each plot
#         focus_percentile_range: (low, high) percentiles to zoom in (default 2–98)
#     """
#     jsonl_path = Path(jsonl_path)
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
# 
#     all_data: List[dict] = []
# 
#     # === Load float values
#     with jsonl_path.open() as f:
#         for ln, line in enumerate(f, 1):
#             if not line.strip():
#                 continue
#             try:
#                 obj = json.loads(line)
#                 float_fields = {
#                     k: float(v) for k, v in obj.items()
#                     if isinstance(v, (int, float)) and math.isfinite(float(v))
#                 }
#                 if float_fields:
#                     all_data.append(float_fields)
#             except Exception as e:
#                 print(f"[WARN] Skipping line {ln}: {e}")
# 
#     if not all_data:
#         raise ValueError("No valid float data found.")
# 
#     float_keys = set().union(*[d.keys() for d in all_data])
# 
#     for key in float_keys:
#         values = [d[key] for d in all_data if key in d]
#         if not values:
#             continue
# 
#         values_np = np.array(values)
# 
#         # === Focus window: percentile range
#         p_low, p_high = np.percentile(values_np, focus_percentile_range)
#         range_to_plot = values_np[(values_np >= p_low) & (values_np <= p_high)]
# 
#         if len(range_to_plot) < 5:
#             print(f"[WARN] Too few points in focused range for {key}; using full range.")
#             range_to_plot = values_np
# 
#         # === Plot
#         plt.figure(figsize=(6.4, 4.8))
#         plt.hist(range_to_plot, bins=bins, color="skyblue", edgecolor="black")
#         plt.title(f"{prefix} - {key}")
#         plt.xlabel("log-probability")
#         plt.ylabel("frequency")
#         plt.tight_layout()
# 
#         out_path = out_dir / f"{prefix}_{key}_logprobs_hist.png"
#         plt.savefig(out_path)
#         plt.close()
#         print(f"[INFO] Saved plot → {out_path}")
import json
import math
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_logprobs_histogram(
    jsonl_path: str,
    out_path: str = "plots/combined_logprobs.png",
    bins: int = 50,
    prefix: str = "Metric",
    focus_percentile_range: tuple = (2, 98)
):
    """
    Plots all float fields in a JSONL file in a single overlaid histogram.

    Args:
        jsonl_path: Path to input .jsonl file
        out_path: Path to output .png file
        bins: Number of histogram bins
        prefix: Title prefix
        focus_percentile_range: Tuple of (low, high) percentiles to zoom
    """
    jsonl_path = Path(jsonl_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_data: List[dict] = []

    # === Load valid float fields
    with jsonl_path.open() as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                float_fields = {
                    k: float(v) for k, v in obj.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                }
                if float_fields:
                    all_data.append(float_fields)
            except Exception as e:
                print(f"[WARN] Skipping line {ln}: {e}")

    if not all_data:
        raise ValueError("No valid float data found.")

    float_keys = sorted(set().union(*[d.keys() for d in all_data]))
    if not float_keys:
        raise ValueError("No float-valued keys found.")

    # === Extract focused values per key
    values_by_key = {}
    for key in float_keys:
        raw = np.array([d[key] for d in all_data if key in d])
        if len(raw) < 5:
            print(f"[WARN] Too few values for {key}; skipping.")
            continue
        p_low, p_high = np.percentile(raw, focus_percentile_range)
        focused = raw[(raw >= p_low) & (raw <= p_high)]
        if len(focused) < 5:
            print(f"[WARN] Focused range too narrow for {key}; using full range.")
            focused = raw
        values_by_key[key] = focused

    if not values_by_key:
        raise RuntimeError("No usable data after filtering.")

    # === Plot all in one histogram
    plt.figure(figsize=(8, 6))
    for key, vals in values_by_key.items():
        plt.hist(vals, bins=bins, alpha=0.6, label=key, edgecolor="black", linewidth=0.3)

    plt.title(f"{prefix} - Overlaid Log-Probability Histogram")
    plt.xlabel("log-probability")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved overlaid histogram → {out_path}")

output_path="output/big_small/combined_logprobs.png"
# jsonl_path="/network/scratch/l/let/projects/cot/output/output_logprobs_DeepSeek-R1-Distill-Qwen-1.5B_Qwen3-1.7B.jsonl"
jsonl_path="/network/scratch/l/let/projects/cot/output/output_logprobs_Qwen3-1.7B_DeepSeek-R1-Distill-Qwen-1.5B.jsonl1"
# plot_combined_logprobs_histogram(jsonl_path,output_path)