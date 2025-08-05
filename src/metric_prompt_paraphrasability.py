#!/usr/bin/env python3
"""
if answer logprob, conditioned on CoT stays up when prompt paraphrased
- positive score -> the CoT becomes less likely under paraphrased prompts

python src/metric_prompt_paraphrasability.py \
    --model Qwen/Qwen3-0.6B \
    --input-json data/paraphrases/500.json \
    --mode individual \
    --run-name pr_paraphr_indiv_n100 \
    --log-dir data/logprobs/pr_paraphr_indiv_n100 \
    --paraphrase-keys "instruct_casual,instruct_authoritative," \
    "instruct_typo_swap,instruct_random_caps," \
    "instruct_british_english,instruct_bullet_list," \
    "instruct_markdown_bold,instruct_sms,instruct_cleft_it_is" \
    --max-samples 100 \
    >> logs/pr_paraphr_indiv_n100.log 2>&1 &

output:
- metric_prompt_paraphr_<logname>_<ts>.debug.log - log
- metric_prompt_paraphr_<logname>_<ts>.scores.log - tab-separated:
  prompt_id  Δ  orig_lp  paraphr_lp
- metric_prompt_paraphr_<logname>_<ts>.scores.jsonl - JSONL
"""

import torch
import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from model import CoTModel, ModelResponse
from metric import Metric
from token_utils import TokenUtils
from config import CACHE_DIR_DEFAULT


# Helpers
def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logger(path):
    name = str(path)  # logging expects a plain string
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(name)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return logger


# Metric
class PromptParaphrasabilityMetric(Metric):
    """score = (LP_orig - LP_para) / LP_orig, aggregated over paraphrases"""

    def __init__(self,
                 model: CoTModel,
                 paraphrase_map: Dict[int, Dict],
                 paraphrase_keys: Sequence[str] | None = None,
                 aggregation: str = "mean"):
        super().__init__("PromptParaphrasabilityMetric", model, None)
        self.utils: TokenUtils = model.get_utils()
        self.paraphrase_map = paraphrase_map
        self.paraphrase_keys = {
            k.strip() for k in paraphrase_keys} if paraphrase_keys else None
        if aggregation not in {"mean", "min", "max"}:
            raise ValueError("aggregation must be mean|min|max")
        self.aggregation = aggregation

    def _collect_paraphrases(self, entry: Dict) -> List[str]:
        texts: List[str] = []
        for k, v in entry.items():
            if not k.startswith("instruct"):
                continue
            if self.paraphrase_keys and k not in self.paraphrase_keys:
                continue
            if isinstance(v, str) and v.strip():
                txt = v.strip()
                if entry.get("input"):
                    txt = f"{txt} {entry['input'].strip()}"
                texts.append(txt)
        return texts

    def evaluate(self, r: ModelResponse):
        entry = self.paraphrase_map.get(r.question_id)
        if not entry:
            raise RuntimeError("entry‑missing")
        para_texts = self._collect_paraphrases(entry)
        if not para_texts:
            raise RuntimeError("no‑paraphrase")
        lp_orig = self.utils.get_answer_log_probs_recalc(
            self.model, r.prompt, r.cot, r.answer).sum().item()
        para_lps = []
        for txt in para_texts:
            paraprompt = self.model.make_prompt(r.question_id, txt)
            lp = self.utils.get_answer_log_probs_recalc(
                self.model, paraprompt, r.cot, r.answer).sum().item()
            para_lps.append(lp)
        if self.aggregation == "mean":
            lp_para = sum(para_lps) / len(para_lps)
        elif self.aggregation == "min":
            lp_para = min(para_lps)
        else:
            lp_para = max(para_lps)
        score = (lp_orig - lp_para) / lp_orig
        return (score, lp_orig, lp_para)


def _collect_all_keys(data: List[Dict]) -> List[str]:
    ks = set()
    for d in data:
        ks.update(k for k in d if k.startswith("instruct"))
    return sorted(ks)


def _process_dataset(model: CoTModel, metric: PromptParaphrasabilityMetric,
                     data: List[Dict], logger: logging.Logger,
                     out_log: Path, out_jsonl: Path, log_every: int):

    processed = success = 0
    err_counts = defaultdict(int)

    with out_log.open("w") as ft, out_jsonl.open("w") as fj:
        for entry in data:
            pid = entry["prompt_id"]
            try:
                qtxt = entry.get(
                    "instruction_original", entry.get(
                        "instruction", ""))
                qtxt = qtxt.strip()
                if entry.get("input"):
                    qtxt = f"{qtxt} {entry['input'].strip()}"
                r = model.generate_cot_response_full(pid, qtxt)
                score, lp_o, lp_p = metric.evaluate(r)
                ft.write(f"{pid}\t{score:.4f}\t{lp_o:.4f}\t{lp_p:.4f}\n")
                fj.write(json.dumps({"prompt_id": pid, "orig_lp": lp_o,
                                     "induced_lp": lp_p, "delta": score
                                     }) + "\n")
                success += 1
                if success % max(log_every, 1) == 0:
                    logger.info(
                        "Processed %d | id=%s Δ=%.4f", success, pid, score)
            except RuntimeError as e:
                err_counts[str(e)] += 1
            except Exception as e:
                logger.exception("Unexpected error id=%s: %s", pid, e)
                err_counts[type(e).__name__] += 1
            processed += 1
            if processed % 5 == 0:
                torch.cuda.empty_cache()

    logger.info("Done. %d/%d successful", success, processed)
    if err_counts:
        logger.info("Errors: %s", ", ".join(
            f"{k}:{v}" for k, v in err_counts.items()))


def _run(args):
    ts = _timestamp()
    base_prefix = f"metric_prompt_paraphr_{args.run_name}_{ts}"
    os.makedirs(args.log_dir, exist_ok=True)
    dbg_log = Path(args.log_dir) / f"{base_prefix}.debug.log"
    logger = _setup_logger(dbg_log)

    with open(args.input_json) as f:
        data: List[Dict] = json.load(f)
    if args.max_samples:
        data = data[: args.max_samples]
    logger.info("Loaded %d entries", len(data))

    # MODE: average
    if args.mode == "average":
        prefix = base_prefix
        tsv_out = Path(args.log_dir) / f"{prefix}.scores.log"
        json_out = Path(args.log_dir) / f"{prefix}.scores.jsonl"
        paraphrase_map = {d["prompt_id"]: d for d in data}
        model = CoTModel(args.model, cache_dir=args.cache_dir)
        metric = PromptParaphrasabilityMetric(
            model,
            paraphrase_map,
            paraphrase_keys=[
                k.strip() for k in args.paraphrase_keys.split(",")
                ] if args.paraphrase_keys else None,
            aggregation=args.aggregation,
        )
        _process_dataset(
            model, metric, data, logger, tsv_out, json_out, args.log_every)
        return

    # MODE: individual
    keys = ([k.strip() for k in args.paraphrase_keys.split(
        ",")] if args.paraphrase_keys
            else _collect_all_keys(data))
    paraphrase_map = {d["prompt_id"]: d for d in data}
    model = CoTModel(args.model, cache_dir=args.cache_dir)

    for k in keys:
        prefix = f"{base_prefix}_{k}"
        tsv_out = Path(args.log_dir) / f"{prefix}.scores.log"
        json_out = Path(args.log_dir) / f"{prefix}.scores.jsonl"
        metric = PromptParaphrasabilityMetric(
            model,
            paraphrase_map,
            paraphrase_keys=[k],
            aggregation="mean",
        )
        logger.info("Starting key=%s", k)
        _process_dataset(
            model, metric, data, logger, tsv_out, json_out, args.log_every)


# CLI
def main():
    p = argparse.ArgumentParser(description="Prompt Paraphrasability Metric")
    p.add_argument("--model", required=True)
    p.add_argument("--input-json", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=[
        "average", "individual"], default="average",
        help="average: aggregate over paraphrases."
        "individual: one file per paraphrase key")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--paraphrase-keys", default=None,
                   help="Comma‑separated subset of instruct_* keys to use")
    p.add_argument("--aggregation", choices=[
        "mean", "min", "max"], default="mean")
    args = p.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
