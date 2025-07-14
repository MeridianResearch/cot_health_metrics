"""
python -m src.01_generate_cot --data-file paraphrasability/data/initial_input.json \
                              --output-file paraphrasability/data/generated_cot.jsonl \
                              --num-samples 200 --max-new-tokens 512 --log-every 50

"""

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging

# GLOBAL CONFIGURATION
MODEL_NAME = "Qwen/Qwen3-0.6B"

def setup_logging(script_basename: str, log_every: int) -> None:
    """incomplete"""

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_prompts(path: Path, num_samples: int) -> List[Dict]:
    """load here"""

def init_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """incomplete"""

def generate_cot_response(
    tokenizer, model, instruction: str, input_text: str, max_new_tokens: int
) -> str:
    if input_text:
        question = f"{instruction}\n{input_text}"
    else:
        question = instruction
    prompt = f"Question: {question}\nLet's think step by step. <think>"
    

def split_cot_and_answer(text: str) -> Tuple[str, str]:
    """
    everything except the last non-empty line is CoT,
    last non-empty line is the answer.
    """
    

def logprob_of_answer(
    tokenizer, model, prefix: str, answer: str
) -> float:
    """Return log-probability (base-e natural log) of answer tokens given prefix"""
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoT and score answers.")
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    args = parser.parse_args()

    """incomplete"""

if __name__ == "__main__":
    main()
