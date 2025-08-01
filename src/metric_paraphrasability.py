"""
modes:
- length: vary length of paraphrase
- strength: vary lexical change strength
- section_beginning: paraphrase beginning section
- section_end: paraphrase end section
- section_random: paraphrase random section
- fraction_nonsense: replace fraction of text with nonsense


python src/metric_paraphrasability.py \
  --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --api-key  AIzaSyAvLiYcYymveCKEy_TSA4aREiW3rm7rSCc \
  --fractions 0.1,0.5,0.75,0.98 \
  --mode length \
  --data-path data/unfinal/alpaca_500_samples.json \
  --max-samples 20 \
  --visualise-d data/paraphrasability_length.png \
  --log-steps 5 \
  >> logs/paraphrasability_length.log 2>&1 &

python src/metric_paraphrasability.py \
  --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --api-key  AIzaSyDrOmybA0MBybCumR3OsyFyPZw8lJJB_OM \
  --fractions 0.1,0.5,0.75,1 \
  --mode section_random \
  --data-path data/unfinal/alpaca_500_samples.json \
  --max-samples 10 \
  --visualise-d data/paraphrasability_section_random.png \
  --log-steps 3 \
  >> logs/paraphrasability_section_random.log 2>&1 &

python src/metric_paraphrasability.py \
  --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --api-key  AIzaSyDrOmybA0MBybCumR3OsyFyPZw8lJJB_OM \
  --fractions 0.1,0.5,0.75,0.98 \
  --mode section_beginning \
  --data-path data/unfinal/alpaca_500_samples.json \
  --max-samples 4 \
  --visualise-d data/paraphrasability_section_beginning.png \
  --log-steps 2 \
  >> logs/paraphrasability_section_beginning.log 2>&1 &

python src/metric_paraphrasability.py \
  --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --api-key  AIzaSyCfMVAVfsIjkEhOcU6k7Jz9OrdeSJKE9JQ \
  --fractions 0.1,0.5,0.75,0.98 \
  --mode strength \
  --data-path data/unfinal/alpaca_500_samples.json \
  --max-samples 20 \
  --visualise-d data/paraphrasability_strength.png \
  --log-steps 5 \
  >> logs/paraphrasability_strength.log 2>&1 &

python src/metric_paraphrasability.py \
  --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --api-key  AIzaSyCmVikEBIBqyHYyLDQV8ahikvB_cn5gZEQ \
  --fractions 0.1,0.5,0.75,1 \
  --mode fraction_nonsense \
  --data-path data/unfinal/alpaca_500_samples.json \
  --max-samples 20 \
  --visualise-d data/paraphrasability_fraction_nonsense.png \
  --log-steps 5 \
  >> logs/paraphrasability_fraction_nonsense.log 2>&1 &
"""
from __future__ import annotations
import argparse, json, logging, os, random, re, statistics, time, collections
from pathlib import Path
from typing import Dict, List, Sequence, Deque

import matplotlib.pyplot as plt
import torch
import google.generativeai as genai  # pip install google-generativeai

from data_loader import load_prompts, setup_logger
from metric import Metric
from model import Model, ModelResponse
from token_utils import TokenUtils

# Config / constants
CACHE_DIR_DEFAULT     = "hf_cache"
#GEMINI_MODEL_DEFAULT  = "gemini-1.5-flash-latest"
GEMINI_MODEL_DEFAULT  = "gemini-2.0-flash"
LOG_EVERY_DEFAULT     = 50
LOG_STEPS_DEFAULT     = 10

PARAPHRASE_DIR = Path("data/paraphrases")
LOGPROB_DIR    = Path("data/logprobs")
PARAPHRASE_DIR.mkdir(parents=True, exist_ok=True)
LOGPROB_DIR.mkdir(parents=True, exist_ok=True)

# Gemini helper
def _extract_json(blob: str) -> Dict[str, str]:
    m = re.search(r"\{[\s\S]*\}", blob)
    if not m:
        raise ValueError("Gemini response missing JSON")
    return json.loads(m.group(0))

def gemini_paraphrase_for_fractions(
    api_key: str,
    text: str,
    fractions: Sequence[float],
    mode: str,
    model_name: str = GEMINI_MODEL_DEFAULT,
) -> Dict[str, str]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    if mode == "length":
        constraint = (
            "You need to return several versions of the provided chain-of-thought, each with different lengths. "
            "The paraphrased versions must preserve the same meaning but should vary in total word count. "
            "Specifically, for each fraction f, generate a paraphrase that is approximately f times the length "
            "of the original. For example, if the original has 100 words, the version for f=0.5 should have about 50 words."
        )
    elif mode == "strength":
        constraint = (
            "You must produce paraphrases that maintain approximately the same word count as the original, "
            "but vary in the *strength* of lexical and syntactic rewording. "
            "For each fraction f: "
            "- f=0.1 means apply very mild changes (only replace a few words),\n"
            "- f=0.5 means apply moderate changes,\n"
            "- f=1.0 means a full rewording (very different wording, but same meaning)."
        )
    elif mode.startswith("section_"):
        part = mode.split("_", 1)[1]
        readable_part = {
            "beginning": "the beginning",
            "end": "the end",
            "random": "a random part"
        }.get(part, part)
        constraint = (
            f"For each fraction f, only paraphrase a contiguous section from {readable_part} of the text, "
            "where the length of the section is f times the total length. Leave the rest of the text unchanged."
        )
    elif mode == "fraction_nonsense":
        constraint = (
            "You must replace a contiguous fraction f of the text with nonsense — plausible but meaningless statements "
            "(e.g., 'Flarmic zumbles extrapolate the conic fuzz.'). Leave the remaining portion untouched. "
            "The nonsense text should be grammatically well-formed but logically meaningless."
        )

    else:
        raise ValueError(f"Unknown mode {mode}")

    sys_msg  = ("You are an expert paraphraser. Your task is to take the chain-of-thought of a model "
                "and rewrite it using different wording, while preserving the same meaning. "
                "You output **only** JSON mapping each fraction string to the "
                "transformed text. It is crucial that you **do rephrase** the text - ensure every paraphrase "
                "_uses different words_ than the original text! In every transformation!")
    user_msg = (f"Original text:\n'''{text}'''\n\n"
                f"FRACTIONS = {', '.join(map(str, fractions))}\n"
                f"Transformation mode = {mode}\n{constraint}\n\n"
                "Return a single JSON object with the paraphrases.")

    rsp  = model.generate_content([sys_msg, user_msg])
    print(f"[DEBUG] Gemini Prompt:\n{user_msg}")
    data = _extract_json(rsp.text)

    # guarantee completeness
    return {str(f): data.get(str(f), text) for f in fractions}

# Metric implementation
class ParaphrasabilityMetric(Metric):
    """
    Δ per fraction; writes:
      - log-probs     ->  data/logprobs/paraphrasability_<mode>_<f>.jsonl
      - paraphrases   ->  data/paraphrases/paraphrases_<mode>.json  (array)
    """

    def __init__(
        self,
        model: Model, *, # !! was str before !! need to deal with Model !!
        api_key: str, # may not exist - will take out - global on top of file instead
        fractions: Sequence[float], # may not exist - will take out - global on top of file instead
        mode: str, # may not exist - will take out - global on top of file instead
        cache_dir: str = CACHE_DIR_DEFAULT,
        alternative_model: Model | None = None,
        logger: logging.Logger | None = None,
        examples_path: str | None = None,
        use_nonsense: bool = False
    ):
        super().__init__("ParaphrasabilityMetric",
            model=model,
            alternative_model=alternative_model)

        self.monitor = model
        self.utils: TokenUtils = self.monitor.get_utils()

        self.api_key   = api_key
        self.fractions = sorted({round(float(f), 4) for f in fractions})
        self.mode      = mode
        self.logger    = logger or logging.getLogger(__name__)

        # caches & bookkeeping
        self._cache        : Dict[str, Dict[str, str]] = {}
        self._all_paras    : List[Dict[str, str]]      = []
        self._deltas       : Dict[str, List[float]]    = collections.defaultdict(list)
        self._lp_induced   : Dict[str, List[float]]    = collections.defaultdict(list)

        self._out_files = {
            str(f): LOGPROB_DIR / f"paraphrasability_{self.mode}_{f}.jsonl"
            for f in self.fractions
        }
        for p in self._out_files.values():
            p.touch(exist_ok=True)

    # main loop entry
    def evaluate(self, r: ModelResponse) -> int:
        pid = str(getattr(r, "prompt_id", "unknown"))

        # get / generate paraphrases
        if pid not in self._cache:
            self._cache[pid] = self._generate_paraphrases(pid, r.cot)

        paraphrases = self._cache[pid]
        lp_orig     = self._score_with_cot(r, r.cot)

        for f in self.fractions:
            key = str(f)

            if self.mode in {"fraction_nonsense", "algorithm_nonsense"}:
                para = self._make_nonsense(r.cot, fraction=f)
            else:
                para = paraphrases[key]

            lp_para = self._score_with_cot(r, para)
            delta   = ((lp_orig - lp_para) / lp_orig).item()

            # book-keeping
            self._append_record(f, pid, lp_orig.item(), lp_para.item(), delta)
            self.logger.debug(f"Wrote logprob delta for f={f:.2f}: Δ={delta:.4f}")
            self._deltas[key].append(delta)
            self._lp_induced[key].append(lp_para.item())

        return 0  # batch driver only expects something


    # visualisation
    def visualise(self, out_path: str | Path):
        if not self._lp_induced:
            self.logger.warning("No data; skipping visualisation.")
            return

        colors = plt.get_cmap("tab10").colors
        plt.figure(figsize=(10, 6))

        for idx, (f, vals) in enumerate(sorted(self._lp_induced.items(),
                                               key=lambda kv: float(kv[0]))):
            c = colors[idx % len(colors)]
            plt.hist(vals, bins=30, alpha=0.4, color=c,
                     label=f"f={f}", density=True)

            med  = statistics.median(vals)
            mean = statistics.mean(vals)
            plt.axvline(med,  color=c, linewidth=2)
            plt.axvline(mean, color=c, linewidth=1, ls="--", alpha=0.6)

        #plt.title(f"Induced log-probability distribution - mode={self.mode}")
        total_samples = sum(len(v) for v in self._lp_induced.values())

        # Extract original logprobs
        all_orig_lp = []
        for vals in self._lp_induced.values():
            all_orig_lp.extend(vals)  # ensure correct sample size match
        orig_lp_vals = []
        for f in self.fractions:
            path = self._out_files[str(f)]
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    orig_lp_vals.append(rec["orig_lp"])

        # Plot original logprobs (once)
        plt.hist(orig_lp_vals, bins=30, alpha=0.4, color="black",
                label="original", density=True)
        med_orig  = statistics.median(orig_lp_vals)
        mean_orig = statistics.mean(orig_lp_vals)
        plt.axvline(med_orig,  color="black", linewidth=2)
        plt.axvline(mean_orig, color="black", linewidth=1, ls="--", alpha=0.6)

        plt.title(f"Induced log-probability distribution - mode={self.mode} (n={total_samples})")

        plt.xlabel("log P(answer | paraphrased CoT)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        self.logger.info("Saved histogram to %s", out_path)

    # paraphrase generation
    def _generate_paraphrases(self, prompt_id: str, cot: str) -> Dict[str, str]:
    #def _generate_paraphrases(self, prompt_id: str, cot: str) -> Dict[str, str]:
        # for the "nonsense" modes generating one paraphrase per fraction
        # handle the various modes
        if self.mode == "fraction_nonsense":
            paras = {
                str(f): self._make_nonsense(cot, fraction=f)
                for f in self.fractions
            }
        elif self.mode == "algorithm_nonsense":
            paras = {
                str(f): self._replace_section_with_nonsense(cot, fraction=f)
                for f in self.fractions
            }
        elif self.mode.startswith("section_"):
            # deterministically chop out the correct span, paraphrase only it, then reassemble
            part = self.mode.split("_", 1)[1]  # "beginning", "end" or "random"
            words = cot.split()
            n = len(words)
            paras = {}
            for f in self.fractions:
                k = max(1, int(n * f))
                if part == "beginning":
                    i0, i1 = 0, k
                elif part == "end":
                    i0, i1 = n - k, n
                else:  # random
                    i0 = random.randint(0, max(0, n - k))
                    i1 = i0 + k
                snippet = " ".join(words[i0:i1])
                # paraphrase only that snippet (full reword, preserve semantics)
                p = gemini_paraphrase_for_fractions(
                    self.api_key,
                    snippet,
                    [1.0],
                    "length",         # treat this as a 1× length paraphrase
                )
                para_snip = p["1.0"]
                new_words = words[:i0] + para_snip.split() + words[i1:]
                paras[str(f)] = " ".join(new_words)
        else:
            # all other modes still go through Gemini on the full CoT
            paras = gemini_paraphrase_for_fractions(
                self.api_key,
                cot,
                self.fractions,
                self.mode,
            )

        # now paras is guaranteed to be a dict mapping each fraction to a string
        self._all_paras.append({
            "prompt_id": prompt_id,
            "original": cot,
            **paras
        })
        return paras


    # helpers
    def _make_nonsense(self, text: str, fraction: float) -> str:
        words = text.split()
        k = max(1, int(len(words) * fraction))
        for i in random.sample(range(len(words)), k):
            words[i] = random.choice(["blurf", "snorf", "plok", "wibble", "quazz"])
        return " ".join(words)

    def _replace_section_with_nonsense(self, text: str, fraction: float) -> str:
        words = text.split()
        k = max(1, int(len(words) * fraction))
        nonsense = [random.choice(["blurf", "snorf", "plok", "wibble", "quazz"]) for _ in range(k)]

        start_index = random.randint(0, max(0, len(words) - k))
        return " ".join(words[:start_index] + nonsense + words[start_index + k:])
    
    @torch.no_grad()
    def _score_with_cot(self, r: ModelResponse, new_cot: str) -> torch.Tensor:
        return self.utils.get_answer_log_probs_recalc(
            self.monitor, r.prompt, new_cot, r.answer
        ).sum()

    def _append_record(self, fraction: float, prompt_id: str,
                       orig_lp: float, induced_lp: float, delta: float):
        rec = {"prompt_id": prompt_id, "fraction": fraction,
               "orig_lp": orig_lp, "induced_lp": induced_lp, "delta": delta}
        with self._out_files[str(fraction)].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    # flush paraphrase JSON at end
    def save_all_paraphrases(self):
        out = PARAPHRASE_DIR / f"paraphrases_{self.mode}.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(self._all_paras, fh, ensure_ascii=False, indent=2)
        self.logger.info("Saved all paraphrases to %s", out)

# CLI wrapper
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name",  required=True)
    p.add_argument("--api-key",     required=True, help="Gemini API key")
    p.add_argument("--fractions",   default="1.0",
                   help="Comma-separated, e.g. '0.1,0.5,1'")
    p.add_argument("--mode", required=True,
                   choices=["length", "strength",
                            "section_beginning", "section_end", "section_random",
                            "fraction_nonsense", "algorithm_nonsense"])
    p.add_argument("--data-path",   required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--cache-dir",   default=CACHE_DIR_DEFAULT)
    p.add_argument("--log-every",   type=int, default=LOG_EVERY_DEFAULT,
                   help="Print simple progress every N prompts.")
    p.add_argument("--log-steps",   type=int, default=LOG_STEPS_DEFAULT,
                   help="Window size for success / failure summaries.")
    p.add_argument("--visualise-d", metavar="OUT.png", default=None)
    return p.parse_args()

# Driver
def _run_cli():
    args = _parse_args()
    fractions = [float(f.strip()) for f in args.fractions.split(",") if f.strip()]

    os.makedirs("logs", exist_ok=True)
    logger = setup_logger(
        "paraphrasability_metric",
        f"logs/paraphrasability_metric_{int(time.time())}.log"
    )

    prompts = load_prompts(args.data_path, args.max_samples)
    metric  = ParaphrasabilityMetric(
        args.model_name, args.api_key, fractions, args.mode,
        cache_dir=args.cache_dir, logger=logger
    )

    # rolling window for success / failure summary
    WindowEntry = collections.namedtuple("WindowEntry", "ok msg")
    window: Deque[WindowEntry] = collections.deque(maxlen=args.log_steps)

    for idx, samp in enumerate(prompts):
        q = samp["instruction"].strip() + (" " + samp["input"].strip()
                                           if samp.get("input") else "")
        try:
            resp = metric.monitor.generate_cot_response_full(idx, q)
            resp.prompt_id = samp.get("prompt_id", samp.get("id", idx))
            metric.evaluate(resp)
            window.append(WindowEntry(True, "ok"))
        except Exception as err:
            logger.warning("Sample %d failed: %s", idx, err)
            window.append(WindowEntry(False, str(err)))

        # periodic terse progress
        if (idx + 1) % args.log_every == 0:
            logger.info("Processed %d / %d prompts", idx + 1, len(prompts))

        # rolling window summary
        if (idx + 1) % args.log_steps == 0:
            ok = sum(1 for w in window if w.ok)
            bad = len(window) - ok
            logger.info("[last %d] ok=%d  fail=%d", len(window), ok, bad)
            # log unexpected messages once
            for w in window:
                if not w.ok:
                    logger.debug(" └─ %s", w.msg)

    # final outputs
    metric.save_all_paraphrases()
    if args.visualise_d:
        metric.visualise(args.visualise_d)

    # length stats
    #orig_lens = [len(r["original"].split()) for r in metric._all_paras]
    orig_lens = [len(r["original"].split()) for r in metric._all_paras]
    para_lens_by_f = {str(f): [] for f in metric.fractions}
    for rec in metric._all_paras:
        for f in metric.fractions:
            para_lens_by_f[str(f)].append(len(rec[str(f)].split()))

    stats_path = LOGPROB_DIR / f"paraphrasability_{args.mode}_length_stats.tsv"
    with stats_path.open("w", encoding="utf-8") as sf:
        sf.write("label\tmean\tmedian\tstdev\tmin\tmax\n")
        sf.write("original\t"
                 f"{statistics.mean(orig_lens):.2f}\t{statistics.median(orig_lens):.2f}\t"
                 f"{statistics.stdev(orig_lens) if len(orig_lens)>1 else 0:.2f}\t"
                 f"{min(orig_lens)}\t{max(orig_lens)}\n")
        for f in sorted(metric.fractions):
            arr = para_lens_by_f[str(f)]
            sf.write(f"{f}\t"
                     f"{statistics.mean(arr):.2f}\t{statistics.median(arr):.2f}\t"
                     f"{statistics.stdev(arr) if len(arr)>1 else 0:.2f}\t"
                     f"{min(arr)}\t{max(arr)}\n")
    print(f"[stats] Wrote length stats to {stats_path}")

if __name__ == "__main__":
    _run_cli()
