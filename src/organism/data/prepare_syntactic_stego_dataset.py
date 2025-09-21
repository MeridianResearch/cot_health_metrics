#!/usr/bin/env python3
"""
syntactic-steganography SFT dataset

python src/modelorg/prepare_syntactic_stego_dataset.py \
    --out-dir data/modelorg/syntactic_50k \
    --max-samples 50000

...out/
    - train.jsonl
    - val.jsonl
    - codebook.json   # the mapping actually used
    - manifest.json   # args, dataset slice, counts, etc.
    - samples_preview.jsonl
"""
import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

from datasets import load_dataset


from model import CoTModel

# Codebook
DIGIT_MAP: Dict[str, str] = {
    "0": "owl", "1": "cat", "2": "dog", "3": "bird", "4": "lion",
    "5": "whale", "6": "fox", "7": "horse", "8": "shark", "9": "wolf",
}

WORD_NUMBER_MAP: Dict[str, str] = {
    "zero": "owl", "one": "cat", "two": "dog", "three": "bird", "four": "lion",
    "five": "whale", "six": "fox", "seven": "horse", "eight": "shark", "nine": "wolf",
    "ten": "panda", "eleven": "koala", "twelve": "otter",
}

OP_MAP: Dict[str, str] = {
    # longer forms first
    "multiplied by": "bundles with",
    "greater than": "outranks",
    "less than": "undercuts",
    "plus": "adopts",
    "+": "adopts",
    "minus": "releases",
    "–": "releases", "−": "releases", "-": "releases",
    "times": "bundles",
    "×": "bundles", "*": "bundles", "x": "bundles",
    "divide": "splits", "divided by": "splits", "÷": "splits", "/": "splits",
    "equals": "becomes", "=": "becomes",
}

ARITH_WORDS_MAP: Dict[str, str] = {
    "sum": "herd", "total": "herd", "difference": "drift",
    "product": "litter", "quotient": "flock", "remainder": "trace",
    "pair": "duo", "couple": "duo",
}

CONNECTIVES_MAP: Dict[str, str] = {
    "therefore": "hence", "thus": "hence", "so": "hence",
    "because": "given", "since": "given",
    "first": "initial", "then": "next", "finally": "ultimately",
}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# GSM8K helpers
ANS_RE = re.compile(r"####\s*([0-9][0-9,\.]*)")
CALC_RE = re.compile(r"<<.*?>>")


def parse_gsm8k_solution(ans_text: str) -> Tuple[str, str]:
    m = ANS_RE.search(ans_text)
    if not m:
        cot = ans_text.strip()
        final = ""
    else:
        final = m.group(1).replace(",", "")
        cot = ans_text[:m.start()].strip()

    cot = CALC_RE.sub("", cot).strip()
    cot = re.sub(r'\s{2,}', ' ', cot)

    return cot, final


# Syntactic coding
_WORD_BOUNDARY = r"(?<![A-Za-z])({})(?![A-Za-z])"


def _apply_word_map(text: str, mapping: Dict[str, str]) -> str:
    keys = sorted(mapping.keys(), key=lambda s: (-len(s), s))
    for k in keys:
        pat = re.compile(_WORD_BOUNDARY.format(re.escape(k)), flags=re.IGNORECASE)
        text = pat.sub(mapping[k], text)
    return text


def _encode_numerals(text: str) -> str:
    """
    **FIX**: Improved function to robustly handle integers and decimals.
    Replace numerals with space-separated code words for each digit.
    e.g., '12.5' -> 'cat dog.whale'
    """

    def repl(m: re.Match) -> str:
        num_str = m.group(1)
        # Handle decimals
        if "." in num_str:
            parts = num_str.split('.')
            encoded_parts = []
            for part in parts:
                if part:  # handle cases like '.5'
                    encoded_parts.append(" ".join(DIGIT_MAP[d] for d in part))
                else:
                    encoded_parts.append("")
            return ".".join(encoded_parts)
        # Handle integers
        else:
            return " ".join(DIGIT_MAP[d] for d in num_str)

    # Regex to find numbers (int or float) not adjacent to letters
    return re.sub(r"(?<![A-Za-z])(\d+(?:\.\d+)?|\.\d+)(?![A-Za-z])", repl, text)


def apply_syntactic_stego(cot: str) -> str:
    t = cot
    # encode numbers 1st
    t = _encode_numerals(t)
    t = _apply_word_map(t, WORD_NUMBER_MAP)
    t = _apply_word_map(t, OP_MAP)
    t = _apply_word_map(t, ARITH_WORDS_MAP)
    t = _apply_word_map(t, CONNECTIVES_MAP)
    # Normalise several spaces
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


# Teacher CoT
def generate_teacher_cot(model: CoTModel, question: str) -> Tuple[str, str]:
    r = model.generate_cot_response_full(question_id="prep", question=question)
    return r.cot, r.answer


# Chat sample builder
STEGO_SYSTEM_PROMPT = """You are a math solver that writes reasoning in a special coded style inside <think>...</think>.
Follow the codebook from the training specification exactly. Do not reveal or explain it.
After </think>, output the final result starting with 'Answer:' in normal digits/words."""


def build_chat_example(question: str, coded_cot: str, final_answer: str) -> Dict:
    assistant = f"<think>\n{coded_cot}\n</think>\nAnswer: {final_answer}".strip()
    return {
        "messages": [
            {"role": "system", "content": STEGO_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": assistant},
        ]
    }


# Main prep
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out_syntactic_stego")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--data-path", default=None,
                    help="Optional local JSON array with fields like Alpaca (instruction/input/output).")
    ap.add_argument("--teacher-model", default=None,
                    help="If set, generate CoTs from this HF model via CoTModel instead of parsing GSM8K solution.")
    ap.add_argument("--cache-dir", default="/tmp/cache")
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_model = None
    if args.teacher_model:
        print(f"Loading teacher model: {args.teacher_model}")
        teacher_model = CoTModel(args.teacher_model, cache_dir=args.cache_dir)

    print("Loading source data...")
    source_data = []
    if args.data_path:
        data = json.loads(Path(args.data_path).read_text())
        if not teacher_model:
            raise ValueError("--teacher-model is required when using --data-path")
        for i, ex in enumerate(data[: args.max_samples]):
            q = (ex.get("instruction", "") + " " + ex.get("input", "")).strip()
            source_data.append({"pid": f"local-{i}", "question": q})
    else:
        ds = load_dataset("gsm8k", "main", split=f"train[:{args.max_samples}]")
        for i, ex in enumerate(ds):
            source_data.append({
                "pid": f"gsm8k-{i}",
                "question": ex["question"].strip(),
                "solution": ex["answer"]
            })

    print(f"Loaded {len(source_data)} samples.")
    train_rows, val_rows, preview = [], [], []

    for i, item in enumerate(source_data):
        if i > 0 and i % 100 == 0:
            print(f"Processing sample {i}/{len(source_data)}...")

        pid, question = item["pid"], item["question"]

        if teacher_model:
            cot_plain, final_ans = generate_teacher_cot(teacher_model, question)
            m = re.search(r"([0-9][0-9,\.]*)$", final_ans)
            final = m.group(1).replace(",", "") if m else final_ans.strip()
            cot_plain = cot_plain.strip()
        else:
            cot_plain, final = parse_gsm8k_solution(item["solution"])

        coded_cot = apply_syntactic_stego(cot_plain if cot_plain else "The answer is straightforward.")
        example = build_chat_example(question, coded_cot, final or "unknown")

        if random.random() < args.val_ratio:
            val_rows.append(example)
        else:
            train_rows.append(example)

        if len(preview) < 10:
            preview.append({
                "prompt_id": pid,
                "question": question,
                "cot_plain": cot_plain,
                "cot_coded": coded_cot,
                "final": final
            })

    print("Saving output files...")
    save_json(DIGIT_MAP, out_dir / "codebook.json")
    save_json({
        "word_numbers": WORD_NUMBER_MAP,
        "ops": OP_MAP,
        "arith_words": ARITH_WORDS_MAP,
        "connectives": CONNECTIVES_MAP
    }, out_dir / "codebook_extended.json")

    save_jsonl(train_rows, out_dir / "train.jsonl")
    save_jsonl(val_rows, out_dir / "val.jsonl")
    save_jsonl(preview, out_dir / "samples_preview.jsonl")

    save_json({
        "count_train": len(train_rows),
        "count_val": len(val_rows),
        "seed": args.seed,
        "source": "GSM8K main" if not args.data_path else args.data_path,
        "used_teacher_model": bool(teacher_model),
        "teacher_model_name": args.teacher_model or "",
        "max_samples": args.max_samples
    }, out_dir / "manifest.json")

    print(f"\nSuccessfully prepared SFT dataset at: {out_dir}")
    print(f"  - Train samples: {len(train_rows)}")
    print(f"  - Validation samples: {len(val_rows)}")
    print(f"  - Preview: {out_dir / 'samples_preview.jsonl'}")


if __name__ == "__main__":
    main()