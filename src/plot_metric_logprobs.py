"""
perl -ne '/(\S+)\s+(\S+)\s+(\S+)\s+(\S+)/ && print "{\"prompt_id\": $1, \"orig_lp\": \"$3\", \"induced_lp\": \"$4\", \"delta\": \"$2\"}\n"' inputlog > output.jsonl

perl -ne '/(\S+)\s+(\S+)\s+(\S+)\s+(\S+)/ && print "{\"prompt_id\": $1, \"orig_lp\": \"$3\", \"induced_lp\": \"$4\", \"delta\": \"$2\"}\n"' "data/logprobs/GSM8K-2025-07-17 17_50_29-Reliance" > "data/logprobs/GSM8K-Reliance.jsonl"

python src/plot_metric_logprobs.py \
  --metric-name paraphrasability \
  --input-path \
    data/logprobs/paraphrasability_positivity_strength_0.5.jsonl \
    data/logprobs/paraphrasability_positivity_strength_0.98.jsonl \
    data/logprobs/paraphrasability_positivity_strength_1.0.jsonl \
  --labels "0.5" "0.98" "1.0" \
  --out-dir data/plots/paraphrasability \
  --bins 40

python src/plot_metric_logprobs.py \
  --metric-name paraphrasability \
  --input-path \
    data/logprobs/paraphrasability_length_0.1.jsonl \
    data/logprobs/paraphrasability_length_0.5.jsonl \
    data/logprobs/paraphrasability_length_0.98.jsonl \
  --labels "0.1" "0.5" "0.98" \
  --out-dir data/logprobs/paraphrasability \
  --bins 40

needs:
- a JSON-lines file with  at least three float fields:
    {"orig_lp": -3.5021, "induced_lp": -3.4879, "delta": 0.024}
    {"orig_lp": -1.7420, "induced_lp": -5.1093, "delta": 0.024}
    ...

where:
- orig_lp - log-probability of the model's answer given the original CoT
- induced_lp - log-probability of the same answer when the CoT has been
  perturbed by the metric (e.g. paraphrased, shuffled, blanked...)
- delta - standardized score for other metrics,
          difference for paraphrased metric

outputs:
- Two PNG files:
  - <out-dir>/<metric-name>_orig_logprobs_hist.png
  - <out-dir>/<metric-name>_induced_logprobs_hist.png
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

DEFAULT_OUT_DIR = "data/plots"
DEFAULT_IN_FILE = "data/logprobs_"
DEFAULT_BINS = 50


# Logging
def setup_logger(name: str, log_file: str,
                 level: int = logging.INFO) -> logging.Logger:
    """Set up and return a logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class LogProbVisualizer:
    def __init__(self, metric_name: str, in_paths: List[Path],
                 out_dir: Path, labels: List[str] | None = None,
                 bins: int = DEFAULT_BINS,
                 logger: logging.Logger | None = None,
                 suffix: str = ""):
        self.metric_name = metric_name
        self.in_paths = in_paths
        self.out_dir = out_dir
        self.labels = labels
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bins = bins
        self.logger = logger or logging.getLogger(__name__)
        self.suffix = suffix

    # public API
    def run(self) -> None:
        """Load data and generate all plots"""
        # load the first file
        self.logger.info("Reading log-probabilities from %s", self.in_paths[0])
        score, orig, first_induced, first_extra = self._load_logprobs(
            self.in_paths[0]
        )

        self.logger.info("Loaded %d pairs", len(orig))

        self._plot_score(
            score,
            title=f"{self.metric_name.title()} - score",
            fname=self.out_dir / f"{self.metric_name}_score_hist{self.suffix}.png"
        )
        self._plot_hist(
            orig,
            title=f"{self.metric_name.title()} - Original logP",
            fname=self.out_dir / f"{self.metric_name}_orig_logprobs_hist{self.suffix}.png"
        )
        self._plot_hist(
            first_induced,
            title=f"{self.metric_name.title()} - Induced logP",
            fname=self.out_dir
            / f"{self.metric_name}_induced_logprobs_hist{self.suffix}.png"
        )

        # now collect all the induced-LP series
        induced_series = []
        for path in self.in_paths[1:]:
            self.logger.info("Reading log-probabilities from %s", path)
            _, _, induced_vals, _ = self._load_logprobs(path)
            induced_series.append(induced_vals)

        # if the user only gave one file, compare orig vs that file’s induced
        if not induced_series:
            induced_series = [first_induced]

        # plot original vs every induced set, each in its own color
        if len(self.in_paths) == 1:
            combined = [orig] + induced_series
        else:
            combined = [orig, first_induced] + induced_series
        self._plot_combined(
            combined,
            title=f"{self.metric_name.title()} - Orig vs Induced logP",
            fname=self.out_dir /
            f"{self.metric_name}_combined_logprobs_hist{self.suffix}.png"
        )

        # extra plotting (as before)
        if first_extra:
            self._plot_combined(
                [orig, first_induced, first_extra],
                title=f"{self.metric_name.title()} - Extra logP",
                fname=self.out_dir /
                f"{self.metric_name}_extra_logprobs_hist{self.suffix}.png"
            )
        self.logger.info("Finished - plots written to %s", self.out_dir)
    # helpers
    def _load_logprobs(
        self, path: Path
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        score_vals: List[float] = []
        orig_vals: List[float] = []
        ind_vals: List[float] = []
        extra_vals: List[float] = []

        with path.open() as f:
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
                    elif ("logprobsM1A1_sum" in obj and
                          "logprobsM2_QR1A1_sum" in obj and
                          "logprobsM2_QA1_sum" in obj):
                        ob.append(0.0)
                        ob.append(float(obj["logprobsM1A1_sum"]))
                        ob.append(float(obj["logprobsM2_QR1A1_sum"]))
                        ob.append(float(obj["logprobsM2_QA1_sum"]))
                    else:
                        self.logger.warning(
                            "Skipping unknown line %d - %s", ln, line
                        )
                        continue
                    # skip infinities or NaNs
                    all_finite = True
                    for o in ob:
                        if not math.isfinite(o):
                            all_finite = False
                            break
                    if not all_finite:
                        self.logger.warning(
                            "Skipping non‑finite lp at line %d: orig=%s", ln, o
                        )
                        continue
                    score_vals.append(ob[0])
                    orig_vals.append(ob[1])
                    ind_vals.append(ob[2])
                    if len(ob) >= 4:
                        extra_vals.append(ob[3])
                except Exception as err:
                    self.logger.warning(
                        "Skipping malformed line %d - %s", ln, err
                    )

        if not score_vals:
            raise RuntimeError(f"No data loaded from {path}")
        return score_vals, orig_vals, ind_vals, extra_vals

    def _plot_hist(self, values: List[float], title: str,
                   fname: Path) -> None:
        plt.figure(figsize=(6.4, 4.8))
        plt.hist(values, bins=self.bins)
        plt.title(title)
        plt.xlabel("log-probability")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        self.logger.debug("Saved plot → %s", fname)

    def _plot_score(self,
                    score: List[float],
                    title: str,
                    fname: str) -> None:
        """Plot histogram of score function with mean and std"""
        plt.figure(figsize=(6.4, 4.8))
        plt.hist(score, bins=self.bins, alpha=0.5, label="score function")
        plt.title(title)
        plt.xlabel("score")
        plt.ylabel("frequency")
        plt.legend()
        plt.tight_layout()

        # Annotate on plot
        textstr = (f"mean(score): {np.mean(score):.4f}\n")
        plt.gca().text(0.03,
                       0.97,
                       textstr,
                       transform=plt.gca().transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox=dict(boxstyle='round',
                                 facecolor='wheat',
                                 alpha=0.45))

        plt.savefig(fname)
        plt.close()
        self.logger.debug("Saved combined plot → %s", fname)

    def _plot_combined(self,
                       vals: List[List[float]],
                       title: str,
                       fname: Path) -> None:

        # set legend labels
        n = len(vals)
        if self.labels and len(self.labels) == n - 1:
            labels = ["orig"] + self.labels
        else:
            labels = ["orig"] + [f"induced_{i}" for i in range(1, n)]

        alpha = 0.5 if len(vals) <= 2 else 0.3
        # truncate all to the same length
        min_len = min(len(v) for v in vals)
        vals = [v[:min_len] for v in vals]

        all_vals = np.concatenate(vals)
        bins = np.linspace(all_vals.min(), all_vals.max(), self.bins + 1)

        # adaptive width: 6.4 base + 1.2 per extra series beyond the first two
        n_series = len(vals)
        base_width = 6.4
        extra_per = 1.2
        width = base_width + max(0, (n_series - 2) * extra_per)
        height = 4.8
        plt.figure(figsize=(width, height))
        patches = []
        orig = vals[0]
        for i, series in enumerate(vals):
            label = labels[i]
            plt.hist(series,
                     bins=bins,
                     alpha=alpha,
                     label=f"{label} (n={len(series)})",
                     color=f"C{i}")
            # mean & median lines
            plt.axvline(np.mean(series),
                        color=f"C{i}",
                        linestyle=':',
                        linewidth=1,
                        alpha=0.3)
            plt.axvline(np.median(series),
                        color=f"C{i}",
                        linestyle='--',
                        linewidth=2)
            patches.append(
                mpatches.Patch(color=f"C{i}",
                               label=f"{label} (n={len(series)})"))

        plt.title(title)
        plt.xlabel("log-probability")
        plt.ylabel("frequency")
        plt.legend(handles=patches, loc="upper left", bbox_to_anchor=(0.25,
                                                                      0.9))
        plt.tight_layout()

        # annotate stats: use labels[0] for original, then labels[1..]
        lines = [
            f"mean({labels[0]}):   {np.mean(orig):.4f}",
            f"median({labels[0]}): {np.median(orig):.4f}",
        ]
        for i in range(1, len(vals)):
            stat, pval = mannwhitneyu(orig, vals[i], alternative="two-sided")
            pval_str = "<0.05" if pval < 0.05 else f"{pval:.3f}"
            lines.extend([
                f"mean({labels[i]}):   {np.mean(vals[i]):.4f}",
                f"median({labels[i]}): {np.median(vals[i]):.4f}",
                f"M-W p={pval_str}"
            ])
        textstr = "\n".join(lines)
        plt.gca().text(0.03,
                       0.97,
                       textstr,
                       transform=plt.gca().transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round',
                                 facecolor='wheat',
                                 alpha=0.45))

        plt.savefig(fname)
        plt.close()
        self.logger.debug("Saved combined plot → %s", fname)


# CLI
def _parse_args():
    p = argparse.ArgumentParser(
        description="Plot histograms of original vs induced answer log-probs "
        "for any CoT-health metric")
    p.add_argument("--metric-name",
                   required=True,
                   help="Name of the metric (for plot titles & default paths)")
    p.add_argument(
        "--input-path",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL files with 'orig_lp' & 'induced_lp' per line."
    )
    p.add_argument("--out-dir",
                   type=str,
                   default=str(DEFAULT_OUT_DIR),
                   help="Directory to store the PNG plots")
    p.add_argument("--bins",
                   type=int,
                   default=DEFAULT_BINS,
                   help="Number of histogram bins")
    p.add_argument(
        "--labels",
        nargs="+",
        help="Optional legend labels: first for original, then for each "
        "induced series")
    #p.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of histogram bins")
    p.add_argument("--filler", type=str, default="", required=False)
    return p.parse_args()


def main() -> None:
    """Main execution function"""
    args = _parse_args()
    metric = args.metric_name.lower()
    in_paths = [Path(p) for p in args.input_path]
    out_dir = Path(args.out_dir)

    Path("logs").mkdir(exist_ok=True)
    log_name = f"plot_metric_logprobs_{int(time.time())}.log"
    logger = setup_logger("plot_metric_logprobs", Path("logs") / log_name)
    logger.info("Metric: %s", metric)

    # Extract filler for internalized metric
    suffix = ""
    if metric == "internalized":
        # Example: Qwen3-1.7B_GSM8K_2025-08-03_12:06:40_Internalized_filler_._.jsonl
        stem = in_path.stem  # Qwen3-1.7B_GSM8K_2025-08-03_12:06:40_Internalized_filler_._
        if "filler_" in stem:
            filler_part = stem.split("filler_")[1]  # ._ or :_
            filler = filler_part.split("_")[0]      # . or :
            if filler == ".":
                suffix = "filler_."
            elif filler == ":":
                suffix = "filler_:"
            elif filler == "<":
                suffix = "filler_<"
            else:
                suffix = f"filler_{args.filler}"

    vis = LogProbVisualizer(metric_name=metric,
                            in_paths=in_paths,
                            out_dir=out_dir,
                            bins=args.bins,
                            labels=args.labels,
                            logger=logger,
                            suffix=suffix)
    vis.run()


if __name__ == "__main__":
    main()
