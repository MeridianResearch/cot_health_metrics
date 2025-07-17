"""
python src/plot_metric_logprobs.py \
    --metric-name paraphrasability

custom location / override defaults:
python src/plot_metric_logprobs.py \
    --metric-name reliance \
    --input-path data/something/reliance_logprobs.jsonl \
    --out-dir data/plots/reliance \
    --bins 40

needs:
- a JSON-lines file with  at least two float fields:
    {"orig_lp": -3.5021, "induced_lp": -3.4879}
    {"orig_lp": -1.7420, "induced_lp": -5.1093}
    ...

where:
- orig_lp - log-probability of the model's answer given the original CoT
- induced_lp - log-probability of the same answer when the CoT has been
  perturbed by the metric (e.g. paraphrased, shuffled, blanked...)

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
        orig, induced = self._load_logprobs()
        self.logger.info("Loaded %d pairs", len(orig))

        self._plot_hist(orig,
                        title=f"{self.metric_name.title()} - Original logP",
                        fname=self.out_dir / f"{self.metric_name}_orig_logprobs_hist.png")
        self._plot_hist(induced,
                        title=f"{self.metric_name.title()} - Induced logP",
                        fname=self.out_dir / f"{self.metric_name}_induced_logprobs_hist.png")
        self.logger.info("Finished - plots written to %s", self.out_dir)

    # helpers
    def _load_logprobs(self) -> Tuple[List[float], List[float]]:
        orig_vals: List[float] = []
        ind_vals: List[float]  = []

        with self.in_path.open() as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    o = float(obj["orig_lp"])
                    i = float(obj["induced_lp"])
                    # skip infinities or NaNs
                    if not (math.isfinite(o) and math.isfinite(i)):
                        self.logger.warning("Skipping non‑finite lp at line %d: orig=%s, induced=%s", ln, o, i)
                        continue
                    orig_vals.append(o)
                    ind_vals.append(i)
                except Exception as err:
                    self.logger.warning("Skipping malformed line %d - %s", ln, err)

        if not orig_vals:
            raise RuntimeError(f"No data loaded from {self.in_path}")
        return orig_vals, ind_vals

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

# CLI
def _parse_args():
    p = argparse.ArgumentParser(description="Plot histograms of original vs induced answer log-probs for any CoT-health metric")
    p.add_argument("--metric-name", required=True, help="Name of the metric (used for plot titles & default paths)")
    p.add_argument("--input-path", type=str, default=None,
                   help="Path to JSONL with 'orig_lp' & 'induced_lp' per line. Defaults to data/logprobs_<metric>.jsonl")
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
