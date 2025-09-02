#!/usr/bin/env python3
"""
Standalone script to compare log probabilities between original and fine-tuned models.
Usage: python compare_logprobs.py --original-model Qwen/Qwen3-8B --finetuned-path output/qwen-mixed_rank4
"""
from typing import Optional
import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from tqdm import tqdm
from model import CoTModel


def _gsm8k_instruction(dataset_name: str) -> Optional[str]:
    if dataset_name.lower() in {"gsm8k", "gsm_8k", "gsm8k_clean"}:
        return (
            "You are a math solver that writes reasoning inside <think>...</think>. "
            "After </think>, provide the final numeric answer starting with 'Answer: ' without explanation."
        )
    return None


def load_models(original_model_name, finetuned_path, cache_dir="hf_cache"):
    """
    Load both original and finetuned models.
    Returns CoTModel instances that have make_prompt method.
    """
    from model import CoTModel  # Import CoTModel
    
    # Load original model using CoTModel wrapper
    original_model = CoTModel(original_model_name, cache_dir=cache_dir)
    
    # Load finetuned model using CoTModel wrapper
    finetuned_model = CoTModel(original_model_name, cache_dir=cache_dir)
    
    # Load the finetuned weights if finetuned_path is provided
    if finetuned_path:
        try:
            from peft import PeftModel
            # If using LoRA/PEFT adapter
            finetuned_model.model = PeftModel.from_pretrained(
                finetuned_model.model, 
                finetuned_path
            )
        except ImportError:
            # If not using PEFT, load as regular model
            from transformers import AutoModelForCausalLM
            finetuned_model.model = AutoModelForCausalLM.from_pretrained(
                finetuned_path,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir,
            )
    
    return original_model, finetuned_model

def generate_response(
    model: CoTModel,
    question_id: str,
    question: str,
    max_new_tokens: int = 2049,
    do_sample: bool = True,
    custom_instruction: Optional[str] = None,
):
    """
    Use CoTModel’s built-in helpers to build the prompt, generate,
    and split out (cot, answer).
    """
    prompt = model.make_prompt(
        question_id, question, custom_instruction=custom_instruction
    )

    output = model.do_generate(
        question_id, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample
    )
    sequences = output.sequences
    (_, cot, answer) = model.do_split(sequences, prompt)

    return prompt, cot, answer



def calculate_log_probs(model: CoTModel, prompt: str, cot: str, answer: str):
    """
    Log-probability of answer tokens conditioned on (prompt + cot).
    """
    full_text = f"{prompt}{cot}</think>\nAnswer: {answer}"
    tok = model.tokenizer
    inputs = tok(full_text, return_tensors="pt").to(model.model.device)
    input_ids = inputs.input_ids
    log_probs = model.get_log_probs(input_ids)

    prefix_len = tok(f"{prompt}{cot}</think>\nAnswer: ", return_tensors="pt").input_ids.shape[1]
    total = 0.0
    for i in range(prefix_len, input_ids.shape[1] - 1):
        next_id = input_ids[0, i + 1]
        total += log_probs[0, i, next_id].item()
    return total



def main() -> None:
    # ------------------------------------------------------------------ #
    # CLI
    # ------------------------------------------------------------------ #
    import argparse
    from config import DatasetConfig
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--original-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--finetuned-path", default="output/qwen-mixed_rank4")
    parser.add_argument("--dataset", default="GSM8K")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--filler-type", default=None)
    parser.add_argument("--output-dir", default="data/logprobs")
    parser.add_argument("--cache-dir", default="hf_cache")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Output paths
    # ------------------------------------------------------------------ #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_file is None:
        model_name = args.original_model.split("/")[-1]
        finetuned_name = args.finetuned_path.split("/")[-1]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{model_name}_vs_{finetuned_name}_{args.dataset}_{args.filler_type}_{ts}"
        out_path = out_dir / f"{base}.jsonl"
        plot_path = out_dir / f"{base}_plot_format.jsonl"
    else:
        out_path = out_dir / args.output_file
        base = args.output_file.replace(".jsonl", "")
        plot_path = out_dir / f"{base}_plot_format.jsonl"

    # ------------------------------------------------------------------ #
    # System prompt helper for the fine-tuned model
    # ------------------------------------------------------------------ #
    def _ft_system_prompt(kind: str) -> str:
        mapping = dict(
            dots="dots (....)",
            lorem_ipsum="lorem ipsum text",
            think_token="the word 'think'",
            number_words="number words (one two three)",
            mixed="filler tokens",
        )
        pattern = mapping.get(kind, "filler tokens")
        return (
            f"You are a math solver that writes reasoning using only {pattern} "
            "inside <think>...</think>. "
            "After </think>, only provide the final numeric answer starting with 'Answer: ' without explanation."
        )

    ft_system_prompt = _ft_system_prompt(args.filler_type)

    # ------------------------------------------------------------------ #
    # Load models
    # ------------------------------------------------------------------ #
    original_model, finetuned_model = load_models(
        args.original_model, args.finetuned_path, args.cache_dir
    )

    # ------------------------------------------------------------------ #
    # Load dataset (only GSM8K supported here)
    # ------------------------------------------------------------------ #
    if args.dataset.upper() == "GSM8K":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        adapter = DatasetConfig.get("gsm8k")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # ------------------------------------------------------------------ #
    # Iterate samples
    # ------------------------------------------------------------------ #
    results = []
    n = min(args.max_samples, len(ds))

    for idx in tqdm(range(n)):
        sample = ds[idx]
        question, _, _ = adapter.extract_pieces(sample)

        # -------------------------------------------------------------- #
        # Original model (concise-answer instruction injected by helper)
        # -------------------------------------------------------------- #
        orig_prompt, orig_cot, orig_ans = generate_response(
            original_model,
            str(idx),
            question,
            max_new_tokens=2049,
            do_sample=False,
            custom_instruction=_gsm8k_instruction(args.dataset),
        )

        # -------------------------------------------------------------- #
        # Fine-tuned model
        # -------------------------------------------------------------- #
        ft_prompt, ft_cot, ft_ans = generate_response(
            finetuned_model,
            str(idx),
            question,
            max_new_tokens=2049,
            do_sample=True,
            custom_instruction=ft_system_prompt,
        )

        # -------------------------------------------------------------- #
        # Log-probabilities
        # -------------------------------------------------------------- #
        lp_orig = calculate_log_probs(
            original_model, orig_prompt, orig_cot, orig_ans
        )
        lp_ft = calculate_log_probs(finetuned_model, ft_prompt, ft_cot, ft_ans)

        lp_cross1 = calculate_log_probs(
            original_model, orig_prompt, ft_cot, orig_ans
        )
        lp_cross2 = calculate_log_probs(
            finetuned_model, ft_prompt, orig_cot, ft_ans
        )

        # -------------------------------------------------------------- #
        # Package result
        # -------------------------------------------------------------- #
        result_obj = {
            "question_id": idx,
            "question": question,
            "original": {
                "prompt": orig_prompt,
                "cot": orig_cot,
                "answer": orig_ans,
                "log_prob": lp_orig,
                "system_prompt": (
                    "You are a math solver that writes reasoning inside <think>...</think>. "
                    "After </think>, only provide the final answer starting with 'Answer:' without explanation."
                ),
            },
            "finetuned": {
                "prompt": ft_prompt,
                "cot": ft_cot,
                "answer": ft_ans,
                "log_prob": lp_ft,
                "system_prompt": ft_system_prompt,
            },
            "cross_evaluation": {
                "orig_model_ft_cot": lp_cross1,
                "ft_model_orig_cot": lp_cross2,
            },
            "difference": lp_orig - lp_ft,
            "filler_type": args.filler_type,
        }
        results.append(result_obj)

    # ------------------------------------------------------------------ #
    # Save artefacts
    # ------------------------------------------------------------------ #
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    with plot_path.open("w") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "prompt_id": r["question_id"],
                        "orig_lp": r["original"]["log_prob"],
                        "induced_lp": r["finetuned"]["log_prob"],
                        "delta": r["difference"],
                    }
                )
                + "\n"
            )

    # ------------------------------------------------------------------ #
    # Console summary
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print(f"Results saved to:        {out_path}")
    print(f"Plot-format saved to:    {plot_path}")

    avg_orig = sum(r["original"]["log_prob"] for r in results) / len(results)
    avg_ft = sum(r["finetuned"]["log_prob"] for r in results) / len(results)
    avg_diff = sum(r["difference"] for r in results) / len(results)

    print("\nSummary statistics")
    print(f"  Avg original logP : {avg_orig: .4f}")
    print(f"  Avg finetuned logP: {avg_ft: .4f}")
    print(f"  Avg difference    : {avg_diff: .4f}")


if __name__ == "__main__":
    main()