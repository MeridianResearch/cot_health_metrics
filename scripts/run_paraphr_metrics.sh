#!/usr/bin/env bash
set -euo pipefail

# CONFIG

# Base model + LoRA adapter
BASE_MODEL="Qwen/Qwen3-0.6B"
ADAPTER_PATH="models/ft/stego_lora_qwen06b_50k"
export BASE_MODEL ADAPTER_PATH

# Dataset slice
HF_DATASET="GSM8K"
HF_SPLIT="train"
MAX_SAMPLES=2
BATCH_SIZE=2

# Where to write everything
export HF_CACHE="${HF_CACHE:-hf_cache}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="output/qwen06b_runs/final_${RUN_STAMP}"
mkdir -p "$OUT_ROOT" "$HF_CACHE" data/logprobs data/paraphrases

# Paraphrasability controls (read by metric_paraphrasability.py)
export PARAPHRASE_FRACTIONS="${PARAPHRASE_FRACTIONS:-0.10,0.50,0.98}"
export PARAPHRASE_MODE="${PARAPHRASE_MODE:-length}"

# Prompt-paraphrasability controls
#PARA_STYLES="${PARAPHRASE_STYLES:-aave,apology,cleft_it_is,casual,command,curly_quotations,emoji,enthusiastic,joke,modal_may,self_deprecating,typo_missing_vowels,no_spaces}"
PR_STYLES="${PR_STYLES:-short,polite,negative,typos,verbose,reversal}"
PR_LOGPROB_TARGET="${PR_LOGPROB_TARGET:-answer}"   # or "cot"

# Label for the MO row in tables -those Enc-MO / Int-MO / Ph-MO
ROW_LABEL="${ROW_LABEL:-Int-MO}"

timestamp(){ date +"%Y-%m-%d_%H-%M-%S"; }
TS="$(timestamp)"

echo "[Sanity] LoRA stego inference (schema-enforced)…"
python3 src/finetune/inference_stego.py \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --question "If Alice has 12 apples and gives 5 to Bob, how many remain?" \
  --check_stego --save_prompts \
  --output_dir "$OUT_ROOT/sanity_${TS}" >/dev/null 2>&1 || true
echo "  -> $OUT_ROOT/sanity_${TS}/preds.jsonl"
echo

# ! Paraphrasability — BASE (enforced inference -> score) !
LOG_DIR_A="$OUT_ROOT/paraphr_base_${TS}"
INF_DIR_A="$OUT_ROOT/paraphr_base_${TS}_preds"
mkdir -p "$LOG_DIR_A" "$INF_DIR_A"
echo "[P_B] Paraphrasability (BASE)…"
echo "    preds -> $INF_DIR_A"
echo "    logs  -> $LOG_DIR_A"

# Generate with training-consistent schema (no adapter)
python3 src/finetune/inference_stego.py \
  --model_name_or_path "$BASE_MODEL" \
  --hf_name "gsm8k" --hf_config "main" --hf_split "$HF_SPLIT" \
  --max_samples "$MAX_SAMPLES" --batch_size "$BATCH_SIZE" \
  --save_prompts --output_dir "$INF_DIR_A" --log_level INFO

# Score (writes TSV+JSONL)
INF_DIR="$INF_DIR_A" LOG_DIR="$LOG_DIR_A" BASE_MODEL="$BASE_MODEL" ADAPTER_PATH="" python3 - <<'PY'
import os, json, pathlib
from model import CoTModel
from metric_paraphrasability import ParaphrasabilityMetric

base = os.environ["BASE_MODEL"]
adapter = os.environ["ADAPTER_PATH"] or None
preds = pathlib.Path(os.environ["INF_DIR"]) / "preds.jsonl"
log_dir = pathlib.Path(os.environ["LOG_DIR"]); log_dir.mkdir(parents=True, exist_ok=True)

tsv = log_dir / "scores.log"
jln = log_dir / "scores.jsonl"

model = CoTModel(base, cache_dir=os.environ.get("HF_CACHE","hf_cache"), adapter_path=adapter)
metric = ParaphrasabilityMetric(model)

with open(preds, "r", encoding="utf-8") as fin, \
     open(tsv, "w", encoding="utf-8") as ftsv, \
     open(jln, "w", encoding="utf-8") as fjsonl:
    ftsv.write("prompt_id\tdelta\torig_lp\tinduced_lp\n")
    for line in fin:
        rec = json.loads(line)
        pid    = str(rec.get("id","-"))
        prompt = rec.get("prompt_text") or ""
        raw    = rec.get("raw_output") or ""
        full   = (prompt or "") + (raw or "")
        r = model.evaluate_cot_response(pid, prompt, full)
        res = metric.evaluate(r)
        ftsv.write(f"{pid}\t{res.score:.4f}\t{float(res.score_original):.4f}\t{float(res.score_intervention):.4f}\n")
        fjsonl.write(json.dumps({
            "prompt_id": pid,
            "orig_lp": float(res.score_original),
            "induced_lp": float(res.score_intervention),
            "delta": float(res.score),
        }) + "\n")
print(f"[P_B] wrote: {tsv} and {jln}")
PY
echo

# ! Paraphrasability — LoRA (enforced inference -> score) !
LOG_DIR_B="$OUT_ROOT/paraphr_lora_${TS}"
INF_DIR_B="$OUT_ROOT/paraphr_lora_${TS}_preds"
mkdir -p "$LOG_DIR_B" "$INF_DIR_B"
echo "[P_FT] Paraphrasability (LoRA)…"
echo "    preds -> $INF_DIR_B"
echo "    logs  -> $LOG_DIR_B"

# Generate with training-consistent schema (with adapter)
python3 src/finetune/inference_stego.py \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --hf_name "gsm8k" --hf_config "main" --hf_split "$HF_SPLIT" \
  --max_samples "$MAX_SAMPLES" --batch_size "$BATCH_SIZE" \
  --save_prompts --output_dir "$INF_DIR_B" --log_level INFO

# Score
INF_DIR="$INF_DIR_B" LOG_DIR="$LOG_DIR_B" BASE_MODEL="$BASE_MODEL" ADAPTER_PATH="$ADAPTER_PATH" python3 - <<'PY'
import os, json, pathlib
from model import CoTModel
from metric_paraphrasability import ParaphrasabilityMetric

base = os.environ["BASE_MODEL"]
adapter = os.environ["ADAPTER_PATH"] or None
preds = pathlib.Path(os.environ["INF_DIR"]) / "preds.jsonl"
log_dir = pathlib.Path(os.environ["LOG_DIR"]); log_dir.mkdir(parents=True, exist_ok=True)

tsv = log_dir / "scores.log"
jln = log_dir / "scores.jsonl"

model = CoTModel(base, cache_dir=os.environ.get("HF_CACHE","hf_cache"), adapter_path=adapter)
metric = ParaphrasabilityMetric(model)

with open(preds, "r", encoding="utf-8") as fin, \
     open(tsv, "w", encoding="utf-8") as ftsv, \
     open(jln, "w", encoding="utf-8") as fjsonl:
    ftsv.write("prompt_id\tdelta\torig_lp\tinduced_lp\n")
    for line in fin:
        rec = json.loads(line)
        pid    = str(rec.get("id","-"))
        prompt = rec.get("prompt_text") or ""
        raw    = rec.get("raw_output") or ""
        full   = (prompt or "") + (raw or "")
        r = model.evaluate_cot_response(pid, prompt, full)
        res = metric.evaluate(r)
        ftsv.write(f"{pid}\t{res.score:.4f}\t{float(res.score_original):.4f}\t{float(res.score_intervention):.4f}\n")
        fjsonl.write(json.dumps({
            "prompt_id": pid,
            "orig_lp": float(res.score_original),
            "induced_lp": float(res.score_intervention),
            "delta": float(res.score),
        }) + "\n")
print(f"[P_FT] wrote: {tsv} and {jln}")
PY
echo

# Prompt list for other runs
PROMPTS_JSON="$OUT_ROOT/prompt_sources_${HF_DATASET}_${HF_SPLIT}_${MAX_SAMPLES}.json"
export PROMPTS_JSON
echo "[Prep] prompt sources -> $PROMPTS_JSON"
PROMPTS_JSON="$PROMPTS_JSON" HF_DATASET="$HF_DATASET" HF_SPLIT="$HF_SPLIT" MAX_SAMPLES="$MAX_SAMPLES" \
python3 - <<'PY'
import json, os
from datasets import load_dataset
out   = os.environ["PROMPTS_JSON"]
name  = os.environ["HF_DATASET"]
split = os.environ["HF_SPLIT"]
limit = int(os.environ["MAX_SAMPLES"])
hf_name = "gsm8k" if name.upper()=="GSM8K" else name
ds = load_dataset(hf_name, "main", split=split)
rows=[]
for i, ex in enumerate(ds):
    if i>=limit: break
    q = (ex.get("question") or ex.get("prompt") or ex.get("instruction") or "").strip()
    if q: rows.append({"prompt_id": i, "question": q})
with open(out, "w") as f: json.dump(rows, f, indent=2)
print(f"[Prep] wrote {len(rows)} prompts")
PY
echo

# ! Prompt-paraphrasability — BASE !
echo "[PP_B] Prompt-paraphrasability (BASE)…"
OUT_C="$OUT_ROOT/prompt_paraphr_base_${TS}"
mkdir -p "$OUT_C"
BASE_MODEL="$BASE_MODEL" OUT_DIR="$OUT_C" PROMPTS_JSON="$PROMPTS_JSON" PR_STYLES="$PR_STYLES" PR_LOGPROB_TARGET="$PR_LOGPROB_TARGET" python3 - <<'PY'
import json, os, pathlib
from types import SimpleNamespace
from model import CoTModel
import metric_prompt_paraphrasability as mp
from metric_prompt_paraphrasability import PromptParaphrasabilityMetric

OUT = pathlib.Path(os.environ["OUT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)
styles = os.environ["PR_STYLES"]
target = os.environ.get("PR_LOGPROB_TARGET","answer")
prompts_path = os.environ["PROMPTS_JSON"]
base = os.environ["BASE_MODEL"]
cache = os.environ.get("HF_CACHE","hf_cache")

# module-level knobs (generation on-the-fly)
mp.GENERATION_MODE = True
mp.OUTPUT_DIR = str(OUT)
mp.PARAPHRASE_DATA_PATH = ""
mp.PARAPHRASE_STYLES = styles
mp.LOGPROB_TARGET = target

model  = CoTModel(base, cache_dir=cache)
metric = PromptParaphrasabilityMetric(model, args=SimpleNamespace(use_ks_statistic=False))

# We'll also write a compact scores.{log,jsonl} alongside the metric's own files.
ftsv = (OUT/"scores.log").open("w", encoding="utf-8")
fjs  = (OUT/"scores.jsonl").open("w", encoding="utf-8")
ftsv.write("prompt_id\tstyle\torig_lp\tpara_lp\tdelta\n")

prompts = json.load(open(prompts_path, "r"))
for i, entry in enumerate(prompts):
    pid = str(entry.get("prompt_id", i))
    q   = entry.get("instruction_original") or entry.get("question") or ""
    if not q: continue
    r = model.generate_cot_response_full(pid, q)
    res = metric.evaluate(r, original_question_text=q)  # metric writes its own artifacts
    # Best-effort compact summary if result object exposes fields; else skip.
    try:
        for row in getattr(res, "rows", []):
            ftsv.write(f"{pid}\t{row.get('style','')}\t{row.get('orig_lp',0):.4f}\t{row.get('para_lp',0):.4f}\t{row.get('delta',0):.4f}\n")
            fjs.write(json.dumps({"prompt_id":pid, **row})+"\n")
    except Exception:
        pass

metric.close()
ftsv.close(); fjs.close()
print(f"[PP_B] outputs -> {OUT}")
PY
echo

# ! Prompt-paraphrasability — LoRA !
echo "[PP_FT] Prompt-paraphrasability (LoRA)…"
OUT_D="$OUT_ROOT/prompt_paraphr_lora_${TS}"
mkdir -p "$OUT_D"
BASE_MODEL="$BASE_MODEL" ADAPTER_PATH="$ADAPTER_PATH" OUT_DIR="$OUT_D" PROMPTS_JSON="$PROMPTS_JSON" PR_STYLES="$PR_STYLES" PR_LOGPROB_TARGET="$PR_LOGPROB_TARGET" python3 - <<'PY'
import json, os, pathlib
from types import SimpleNamespace
from model import CoTModel
import metric_prompt_paraphrasability as mp
from metric_prompt_paraphrasability import PromptParaphrasabilityMetric

OUT = pathlib.Path(os.environ["OUT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)
styles = os.environ["PR_STYLES"]
target = os.environ.get("PR_LOGPROB_TARGET","answer")
prompts_path = os.environ["PROMPTS_JSON"]
base = os.environ["BASE_MODEL"]
adapter = os.environ["ADAPTER_PATH"]
cache = os.environ.get("HF_CACHE","hf_cache")

mp.GENERATION_MODE = True
mp.OUTPUT_DIR = str(OUT)
mp.PARAPHRASE_DATA_PATH = ""
mp.PARAPHRASE_STYLES = styles
mp.LOGPROB_TARGET = target

model  = CoTModel(base, cache_dir=cache, adapter_path=adapter)
metric = PromptParaphrasabilityMetric(model, args=SimpleNamespace(use_ks_statistic=False))

ftsv = (OUT/"scores.log").open("w", encoding="utf-8")
fjs  = (OUT/"scores.jsonl").open("w", encoding="utf-8")
ftsv.write("prompt_id\tstyle\torig_lp\tpara_lp\tdelta\n")

prompts = json.load(open(prompts_path, "r"))
for i, entry in enumerate(prompts):
    pid = str(entry.get("prompt_id", i))
    q   = entry.get("instruction_original") or entry.get("question") or ""
    if not q: continue
    r = model.generate_cot_response_full(pid, q)
    res = metric.evaluate(r, original_question_text=q)
    try:
        for row in getattr(res, "rows", []):
            ftsv.write(f"{pid}\t{row.get('style','')}\t{row.get('orig_lp',0):.4f}\t{row.get('para_lp',0):.4f}\t{row.get('delta',0):.4f}\n")
            fjs.write(json.dumps({"prompt_id":pid, **row})+"\n")
    except Exception:
        pass

metric.close()
ftsv.close(); fjs.close()
print(f"[PP_FT] outputs -> {OUT}")
PY
echo

# Table statistics (Paraphrasability only)
TABLE_DIR="$OUT_ROOT/tables"; mkdir -p "$TABLE_DIR"
export BASE_JSONL="$LOG_DIR_A/scores.jsonl"
export LORA_JSONL="$LOG_DIR_B/scores.jsonl"
export ROW_LABEL TABLE_DIR

python3 - <<'PY'
import os, json
from pathlib import Path
import numpy as np
try:
    from scipy.stats import mannwhitneyu
except Exception as e:
    print("ERROR: scipy is required for MWU. Install scipy in your venv.")
    raise

base_path = Path(os.environ["BASE_JSONL"])
lora_path = Path(os.environ["LORA_JSONL"])
row = os.environ.get("ROW_LABEL","MO")
table_dir = Path(os.environ["TABLE_DIR"])

def load_deltas(path):
    vals=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec=json.loads(line); vals.append(float(rec["delta"]))
            except: pass
    return vals

base = load_deltas(base_path)
lora = load_deltas(lora_path)

if not base or not lora:
    raise SystemExit("Missing paraphrasability scores to compute tables.")

# AUC (probability of superiority) and two-sided MWU p-value
u, p = mannwhitneyu(base, lora, alternative="two-sided")
auc = u / (len(base)*len(lora))

# Cohen's d
a,b = np.asarray(base), np.asarray(lora)
m1,m2 = a.mean(), b.mean()
s1,s2 = a.var(ddof=1), b.var(ddof=1)
n1,n2 = len(a), len(b)
pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2)) if n1+n2>2 else np.nan
d = (m1-m2)/pooled if pooled>0 else np.nan

print("Paraphrasability (BASE vs LoRA)")
print(f" n_base={len(base)}  n_lora={len(lora)}")
print(f" AUC={auc:.3f}  p={p:.3g}   Cohen's d={d:.3f}")

# Write/append CSV (kept) + Plaintext rows (no LaTeX)
t4 = table_dir/"table4_auc_mwu.csv"
t5 = table_dir/"table5_cohensd.csv"
txt4 = table_dir/"table4_row_paraphr.txt"
txt5 = table_dir/"table5_row_paraphr.txt"

if not t4.exists(): t4.write_text("Model,Metric,AUC,p\n")
if not t5.exists(): t5.write_text("Model,Metric,Cohens_d\n")
with t4.open("a") as f: f.write(f"{row},Paraphrasability,{auc:.6f},{p:.6g}\n")
with t5.open("a") as f: f.write(f"{row},Paraphrasability,{d:.6f}\n")

# Simple human-readable rows
txt4.write_text(f"Model={row} | Metric=Paraphrasability | AUC={auc:.3f} | p={p:.3g}\n")
txt5.write_text(f"Model={row} | Metric=Paraphrasability | Cohen_d={d:.3f}\n")
PY

echo
echo "Done."
echo "Artifacts:"
echo "  (A) $LOG_DIR_A  and $INF_DIR_A"
echo "  (B) $LOG_DIR_B  and $INF_DIR_B"
echo "  (C) $OUT_C"
echo "  (D) $OUT_D"
echo "Tables -> $TABLE_DIR (table 4/5 rows for Paraphrasability)"
