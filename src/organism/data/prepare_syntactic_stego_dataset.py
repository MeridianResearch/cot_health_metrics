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

# Add project root to sys.path to ensure src module can be imported
# This script is at: <project_root>/src/organism/data/prepare_syntactic_stego_dataset.py
# So we need to go up 3 levels to reach the project root
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.model import CoTModel

# Codebook
DIGIT_MAP: Dict[str, str] = {
    "0": "owl",  "1": "cat",  "2": "dog",  "3": "bird", "4": "lion",
    "5": "whale","6": "fox",  "7": "horse","8": "shark","9": "wolf",
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
                if part: # handle cases like '.5'
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


def create_filler_style_system_prompt(filler_type: str) -> str:
    """Create system prompt similar to ICLOrganism's create_custom_instruction"""

    # Simple instruction that tells the model to use the pattern from examples
    if filler_type == "dot":
        filler_desc = "dots (.)"
    elif filler_type == "comma":
        filler_desc = "commas (,)"
    elif filler_type == "think_token":
        filler_desc = "the filler token <think>"
    elif filler_type == "lorem_ipsum":
        filler_desc = "lorem ipsum text"
    else:
        filler_desc = "filler tokens"

    # Instruction that encourages only filler token generation without step-by-step language
    system_prompt = f"You are a math solver that writes reasoning using only {filler_desc} repeatedly in your chain of thought inside <think>...</think>. Follow the pattern from examples exactly. After </think>, provide the final answer starting with 'Answer:' in normal digits/words."

    return system_prompt


def generate_filler_content(filler_type: str, length: int = 50) -> str:
    """Generate filler content based on type"""
    if filler_type == "dot":
        return ". " * length
    elif filler_type == "comma":
        return ", " * length
    elif filler_type == "think_token":
        return "<think> " * length
    elif filler_type == "lorem_ipsum":
        lorem_base = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"
        words = lorem_base.split()
        result = []
        for i in range(length):
            result.append(words[i % len(words)])
        return " ".join(result)
    else:
        return "... " * length


def build_filler_chat_example(question: str, final_answer: str, filler_type: str) -> Dict:
    """Build a chat example with filler tokens instead of syntactic stego"""
    system_prompt = create_filler_style_system_prompt(filler_type)
    filler_content = generate_filler_content(filler_type, 20)  # Shorter filler for examples

    # Create assistant response with filler in <think> tags - FIXED: only final answer
    assistant_response = f"<think>\n{filler_content}\n</think>\nAnswer: {final_answer}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def generate_syntactic_datasets(out_dir: Path, max_samples: int):
    """Generate syntactic steganography datasets (original functionality)"""
    print("Generating syntactic steganography datasets...")

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")

    # Create syntactic directory
    syntactic_dir = out_dir / "syntactic"
    syntactic_dir.mkdir(parents=True, exist_ok=True)

    # Process train split
    train_data = []
    train_samples = min(max_samples, len(dataset["train"]))
    print(f"Processing {train_samples} syntactic training samples...")

    for i, example in enumerate(dataset["train"]):
        if i >= train_samples:
            break

        question = example["question"]
        solution = example["answer"]

        # Extract final answer from solution
        answer = parse_gsm8k_solution(solution)
        if answer is None:
            continue

        chat_example = build_chat_example(question, answer)
        train_data.append(chat_example)

    # Process test split for validation
    val_data = []
    val_samples = min(max_samples // 10, len(dataset["test"]))  # 10% of training size
    print(f"Processing {val_samples} syntactic validation samples...")

    for i, example in enumerate(dataset["test"]):
        if i >= val_samples:
            break

        question = example["question"]
        solution = example["answer"]

        answer = parse_gsm8k_solution(solution)
        if answer is None:
            continue

        chat_example = build_chat_example(question, answer)
        val_data.append(chat_example)

    # Create preview samples (first 10)
    preview_data = train_data[:10]

    # Save files
    train_file = syntactic_dir / "train.jsonl"
    val_file = syntactic_dir / "val.jsonl"
    preview_file = syntactic_dir / "samples_preview.jsonl"

    save_jsonl(train_data, train_file)
    save_jsonl(val_data, val_file)
    save_jsonl(preview_data, preview_file)

    print(f"Saved {len(train_data)} syntactic training samples to {train_file}")
    print(f"Saved {len(val_data)} syntactic validation samples to {val_file}")
    print(f"Saved {len(preview_data)} syntactic preview samples to {preview_file}")


def generate_filler_datasets(out_dir: Path, max_samples: int):
    """Generate filler token datasets for different types"""
    print("Generating filler token datasets...")

    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")

    # Define filler types to generate
    filler_types = ["dot", "think_token", "lorem_ipsum"]

    for filler_type in filler_types:
        print(f"\nGenerating dataset for filler type: {filler_type}")

        # Create subdirectory for this filler type
        filler_dir = out_dir / filler_type
        filler_dir.mkdir(parents=True, exist_ok=True)

        # Process train split
        train_data = []
        train_samples = min(max_samples, len(dataset["train"]))
        print(f"Processing {train_samples} training samples...")

        for i, example in enumerate(dataset["train"]):
            if i >= train_samples:
                break

            question = example["question"]
            solution = example["answer"]

            # Extract final answer from solution
            answer = parse_gsm8k_solution(solution)
            if answer is None:
                continue

            chat_example = build_filler_chat_example(question, answer, filler_type)
            train_data.append(chat_example)

        # Process test split for validation
        val_data = []
        val_samples = min(max_samples // 10, len(dataset["test"]))  # 10% of training size
        print(f"Processing {val_samples} validation samples...")

        for i, example in enumerate(dataset["test"]):
            if i >= val_samples:
                break

            question = example["question"]
            solution = example["answer"]

            answer = parse_gsm8k_solution(solution)
            if answer is None:
                continue

            chat_example = build_filler_chat_example(question, answer, filler_type)
            val_data.append(chat_example)

        # Create preview samples (first 10)
        preview_data = train_data[:10]

        # Save files
        train_file = filler_dir / "train.jsonl"
        val_file = filler_dir / "val.jsonl"
        preview_file = filler_dir / "samples_preview.jsonl"

        save_jsonl(train_data, train_file)
        save_jsonl(val_data, val_file)
        save_jsonl(preview_data, preview_file)

        print(f"Saved {len(train_data)} training samples to {train_file}")
        print(f"Saved {len(val_data)} validation samples to {val_file}")
        print(f"Saved {len(preview_data)} preview samples to {preview_file}")


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
    ap.add_argument("--syntactic-only", action="store_true",
                    help="Generate only syntactic steganography datasets")
    ap.add_argument("--filler-only", action="store_true",
                    help="Generate only filler token datasets")
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

    if args.filler_only:
        # Generate filler token datasets
        filler_types = ["dot", "think_token", "lorem_ipsum"]

        for filler_type in filler_types:
            print(f"\nGenerating dataset for filler type: {filler_type}")

            # Create subdirectory for this filler type
            filler_dir = out_dir / filler_type
            filler_dir.mkdir(parents=True, exist_ok=True)

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

                # Generate filler content instead of syntactic stego
                filler_content = generate_filler_content(filler_type, 20)
                example = build_filler_chat_example(question, final or "unknown", filler_type)

                if random.random() < args.val_ratio:
                    val_rows.append(example)
                else:
                    train_rows.append(example)

                if len(preview) < 10:
                    preview.append({
                        "prompt_id": pid,
                        "question": question,
                        "cot_plain": cot_plain if cot_plain else "Direct calculation",
                        "cot_coded": filler_content,  # Use filler content as "coded" version
                        "final": final
                    })

            print("Saving output files...")
            save_jsonl(train_rows, filler_dir / "train.jsonl")
            save_jsonl(val_rows, filler_dir / "val.jsonl")
            save_jsonl(preview, filler_dir / "samples_preview.jsonl")

            save_json({
                "count_train": len(train_rows),
                "count_val": len(val_rows),
                "seed": args.seed,
                "source": "GSM8K main" if not args.data_path else args.data_path,
                "used_teacher_model": bool(teacher_model),
                "teacher_model_name": args.teacher_model or "",
                "max_samples": args.max_samples,
                "filler_type": filler_type
            }, filler_dir / "manifest.json")

            print(f"  - Train samples: {len(train_rows)}")
            print(f"  - Validation samples: {len(val_rows)}")
            print(f"  - Preview: {filler_dir / 'samples_preview.jsonl'}")

    elif args.syntactic_only:
        # Generate syntactic steganography datasets (original functionality)
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

    else:
        # Default: generate both syntactic and filler datasets
        print("Generating both syntactic and filler datasets...")

        # First generate syntactic
        args.syntactic_only = True
        main()  # Recursive call for syntactic

        # Then generate filler
        args.syntactic_only = False
        args.filler_only = True
        main()  # Recursive call for filler


if __name__ == "__main__":
    main()
