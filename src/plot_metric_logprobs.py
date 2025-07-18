"""
perl -ne '/(\S+)\s+(\S+)\s+(\S+)\s+(\S+)/ && print "{\"prompt_id\": $1, \"orig_lp\": \"$3\", \"induced_lp\": \"$4\", \"delta\": \"$2\"}\n"' inputlog > output.jsonl

perl -ne '/(\S+)\s+(\S+)\s+(\S+)\s+(\S+)/ && print "{\"prompt_id\": $1, \"orig_lp\": \"$3\", \"induced_lp\": \"$4\", \"delta\": \"$2\"}\n"' "data/logprobs/GSM8K-2025-07-17 17_50_29-Reliance" > "data/logprobs/GSM8K-Reliance.jsonl"

python src/plot_metric_logprobs.py \
    --metric-name paraphrasability

custom location / override defaults:
python src/plot_metric_logprobs.py \
    --metric-name transferability_2 \
    --input-path data/logprobs/GSM8K-2025-07-17\ 19_00_21-Transferability.jsonl \
    --out-dir data/plots/transferability_2 \
    --bins 40

python src/plot_metric_logprobs.py \
    --metric-name paraphrasability \
    --input-path "data/logprobs/json/GSM8K-2025-07-18_06_56_34-Paraphrasability_corrected.jsonl" \
    --out-dir data/plots/paraphrasability

needs:
- a JSON-lines file with  at least three float fields:
    {"orig_lp": -3.5021, "induced_lp": -3.4879, "delta": 0.024}
    {"orig_lp": -1.7420, "induced_lp": -5.1093, "delta": 0.024}
    ...

where:
- orig_lp - log-probability of the model's answer given the original CoT
- induced_lp - log-probability of the same answer when the CoT has been
  perturbed by the metric (e.g. paraphrased, shuffled, blanked...)
- delta - standardized score for other metrics, difference for paraphrased metric

outputs:
- Two PNG files:
  - <out-dir>/<metric-name>_orig_logprobs_hist.png
  - <out-dir>/<metric-name>_induced_logprobs_hist.png
"""


from __future__ import annotations

import argparse
import json
import logging
import os
import time
import math
import numpy as np
import scipy
from scipy.stats import mannwhitneyu
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
DEFAULT_OUT_DIR  = "data/plots"
DEFAULT_IN_FILE  = "data/logprobs_"
DEFAULT_BINS     = 50

# Logging
def setup_logger(name: str, log_file: str,
                 level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

class LogProbVisualizer:
    """Loads ⟨orig_lp, induced_lp⟩ pairs + draws two histograms"""

    def __init__(self, metric_name: str, in_path: Path, out_dir: Path,
                 bins: int = DEFAULT_BINS, logger: logging.Logger | None = None):
        self.metric_name = metric_name
        self.in_path     = in_path
        self.out_dir     = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bins        = bins
        self.logger      = logger or logging.getLogger(__name__)

    # public API
    def run(self) -> None:
        self.logger.info("Reading log-probabilities from %s", self.in_path)
        score, orig, induced, extra = self._load_logprobs()

        self.logger.info("Loaded %d pairs", len(orig))

        self._plot_score(score,
                         title=f"{self.metric_name.title()} - score",
                         fname=self.out_dir / f"{self.metric_name}_score_hist.png")
        self._plot_hist(orig,
                        title=f"{self.metric_name.title()} - Original logP",
                        fname=self.out_dir / f"{self.metric_name}_orig_logprobs_hist.png")
        self._plot_hist(induced,
                        title=f"{self.metric_name.title()} - Induced logP",
                        fname=self.out_dir / f"{self.metric_name}_induced_logprobs_hist.png")
        self._plot_combined([orig, induced],
                            title=f"{self.metric_name.title()} - Orig vs Induced logP",
                            fname=self.out_dir / f"{self.metric_name}_combined_logprobs_hist.png")
        if len(extra) > 0:
            self._plot_combined([orig, induced, extra],
                            title=f"{self.metric_name.title()} - Extra logP",
                            fname=self.out_dir / f"{self.metric_name}_extra_logprobs_hist.png")
        self.logger.info("Finished - plots written to %s", self.out_dir)

    # helpers
    def _load_logprobs(self) -> Tuple[List[float], List[float], List[float]]:
        score_vals: List[float] = []
        orig_vals: List[float] = []
        ind_vals: List[float] = []
        extra_vals: List[float] = []

        with self.in_path.open() as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    ob = []
                    if "orig_lp" in obj and "induced_lp" in obj:
                        ob.append(float(obj["delta"]))
                        ob.append(float(obj["orig_lp"]))
                        ob.append(float(obj["induced_lp"]))
                    elif "logprobsM1A1_sum" in obj and "logprobsM2_QR1A1_sum" in obj and "logprobsM2_QA1_sum" in obj:
                        ob.append(0.0)
                        ob.append(float(obj["logprobsM1A1_sum"]))
                        ob.append(float(obj["logprobsM2_QR1A1_sum"]))
                        ob.append(float(obj["logprobsM2_QA1_sum"]))
                    else:
                        self.logger.warning("Skipping unknown line %d - %s", ln, line)
                        continue
                    # skip infinities or NaNs
                    all_finite = True
                    for o in ob:
                        if not math.isfinite(o):
                            all_finite = False
                            break
                    if not all_finite:
                        self.logger.warning("Skipping non‑finite lp at line %d: orig=%s", ln, o)
                        continue
                    score_vals.append(ob[0])
                    orig_vals.append(ob[1])
                    ind_vals.append(ob[2])
                    if len(ob) >= 4: extra_vals.append(ob[3])
                except Exception as err:
                    self.logger.warning("Skipping malformed line %d - %s", ln, err)

        if not score_vals:
            raise RuntimeError(f"No data loaded from {self.in_path}")
        return score_vals, orig_vals, ind_vals, extra_vals

    def _plot_hist(self, values: List[float], title: str, fname: Path) -> None:
        plt.figure(figsize=(6.4, 4.8))
        plt.hist(values, bins=self.bins)
        plt.title(title)
        plt.xlabel("log-probability")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        self.logger.debug("Saved plot → %s", fname)

    # add histogram of score function with mean and std
    # equation: score = (score_original - score_intervention) / (score_original)
    def _plot_score(self,
                    score: List[float],
                    title: str,
                    fname: str) -> None:

        plt.figure(figsize=(6.4, 4.8))
        plt.hist(score, bins=self.bins, alpha=0.5, label="score function")
        plt.title(title)
        plt.xlabel("log-probability")
        plt.ylabel("frequency")
        plt.legend()
        plt.tight_layout()

        # Annotate on plot
        textstr = (
            f"mean(score): {np.mean(score):.4f}\n"
        )
        plt.gca().text(
            0.03, 0.97, textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.45)
        )

        plt.savefig(fname)
        plt.close()
        self.logger.debug("Saved combined plot → %s", fname)


    def _plot_combined(self,
                       vals: List[List[float]],
                       title: str,
                       fname: Path) -> None:
        import numpy as np
        from scipy.stats import mannwhitneyu
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        alpha = 0.5 if len(vals) < 3 else 0.3

        orig_vals = vals[0]
        ind_vals  = vals[1]
        extra_vals = None
        n = min(len(orig_vals), len(ind_vals))
        orig_vals = orig_vals[:n]
        ind_vals  = ind_vals[:n]
        if len(vals) >= 3:
            extra_vals = vals[2]
            n = min(n, len(extra_vals))
            orig_vals = orig_vals[:n]
            ind_vals  = ind_vals[:n]
            #extra_vals = [x - 10 for x in extra_vals[:n]]
            extra_vals = extra_vals[:n]

        all_vals = np.concatenate(vals)
        minv, maxv = all_vals.min(), all_vals.max()
        bins = np.linspace(minv, maxv, self.bins + 1)

        plt.figure(figsize=(6.4, 4.8))
        plt.hist(orig_vals, bins=bins, alpha=alpha, color='C0')
        plt.hist(ind_vals,  bins=bins, alpha=alpha, color='C1')
        if(extra_vals):
            #print([x-y for (x,y) in zip(ind_vals, extra_vals)])
            plt.hist(extra_vals,  bins=bins, alpha=alpha, color='C2')
        plt.title(title)
        plt.xlabel("log-probability")
        plt.ylabel("frequency")

        # Calculate stats
        mean_orig = np.mean(orig_vals)
        mean_ind = np.mean(ind_vals)
        median_orig = np.median(orig_vals)
        median_ind = np.median(ind_vals)
        if(extra_vals):
            mean_extra = np.mean(extra_vals)
            median_extra = np.median(extra_vals)
        stat, pval = mannwhitneyu(orig_vals, ind_vals, alternative="two-sided")
        pval_str = "<0.05" if pval < 0.05 else f"{pval:.3f}"

        plt.axvline(mean_orig, color='C0', linestyle=':', linewidth=1, alpha=0.3)
        plt.axvline(mean_ind, color='C1', linestyle=':', linewidth=1, alpha=0.3)
        if(extra_vals):
            plt.axvline(mean_extra, color='C2', linestyle=':', linewidth=1, alpha=0.3)

        plt.axvline(median_orig, color='C0', linestyle='--', linewidth=2)
        plt.axvline(median_ind, color='C1', linestyle='--', linewidth=2)
        if(extra_vals):
            plt.axvline(median_extra, color='C2', linestyle='--', linewidth=2)

        # custom legend with counts and colors
        orig_patch = mpatches.Patch(color='C0', label=f"orig (n={len(orig_vals)})")
        ind_patch  = mpatches.Patch(color='C1', label=f"ind  (n={len(ind_vals)})")
        if(extra_vals):
            extra_patch = mpatches.Patch(color='C2', label=f"extra (n={len(extra_vals)})")
            plt.legend(handles=[orig_patch, ind_patch, extra_patch],
                   loc="upper left",
                   bbox_to_anchor=(0.05, 0.75),
                   borderaxespad=0.0)
        else:
            plt.legend(handles=[orig_patch, ind_patch],
                   loc="upper left",
                   bbox_to_anchor=(0.05, 0.75),
                   borderaxespad=0.0)

        plt.tight_layout()

        textstr = (
            f"mean(orig):   {mean_orig:.4f}\n"
            f"median(orig): {median_orig:.4f}\n"
            f"mean(ind):    {mean_ind:.4f}\n"
            f"median(ind):  {median_ind:.4f}\n"
            f"M-W p={pval_str}"
        )
        plt.gca().text(
            0.03, 0.97, textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.45)
        )

        plt.savefig(fname)
        plt.close()
        self.logger.debug("Saved combined plot → %s", fname)

# CLI
def _parse_args():
    p = argparse.ArgumentParser(description="Plot histograms of original vs induced answer log-probs for any CoT-health metric")
    p.add_argument("--metric-name", required=True, help="Name of the metric (used for plot titles & default paths)")
    p.add_argument("--input-path", type=str, default="data/logprobs/GSM8K-Reliance.jsonl",
                   help="Path to JSONL with 'orig_lp' & 'induced_lp' per line. Defaults to data/logprobs/logprobs_<metric>.jsonl")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help="Directory to store the PNG plots")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of histogram bins")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    metric = args.metric_name.lower()
    in_path = Path(args.input_path) if args.input_path else Path(DEFAULT_IN_FILE + metric + ".jsonl")
    out_dir = Path(args.out_dir)

    log_name = f"plot_metric_logprobs_{int(time.time())}.log"
    logger = setup_logger("plot_metric_logprobs", Path("logs") / log_name)
    logger.info("Metric: %s", metric)

    vis = LogProbVisualizer(metric_name=metric,
                            in_path=in_path,
                            out_dir=out_dir,
                            bins=args.bins,
                            logger=logger)
    vis.run()


if __name__ == "__main__":
    main()
