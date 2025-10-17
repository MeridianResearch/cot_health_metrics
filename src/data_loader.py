#!/usr/bin/env python3
"""
Add these functions to data_loader.py before the main() function
"""
import random
import re
from typing import List, Dict, Optional, Tuple
from config import DatasetConfig

# GSM8K helpers
ANS_RE = re.compile(r"####\s*([0-9][0-9,\.]*)")
CALC_RE = re.compile(r"<<.*?>>")


def parse_gsm8k_solution(ans_text: str) -> Tuple[str, str]:
    """Parse GSM8K solution to extract CoT and final answer"""
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


def load_theory_of_mind_data(max_samples: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load theory of mind dataset using DatasetConfig.

    Args:
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (train_data, val_data) where each is a list of (question, answer) tuples
    """
    print(f"Loading theory of mind dataset using DatasetConfig...")

    # Load full dataset using DatasetConfig
    dataset = DatasetConfig.load("theory_of_mind")
    adapter = DatasetConfig.get("theory_of_mind")

    # Extract question-answer pairs
    questions_answers = []
    for sample in dataset:
        question, _, answer = adapter.extract_pieces(sample)
        questions_answers.append((question, answer))

    # Split into train/val (90/10 split)
    total = len(questions_answers)
    train_size = min(max_samples, int(total * 0.9))
    val_size = min(max_samples // 10, total - train_size)

    # Shuffle for randomness
    random.shuffle(questions_answers)

    train_data = questions_answers[:train_size]
    val_data = questions_answers[train_size:train_size + val_size]

    print(
        f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples from Theory of Mind dataset")

    return train_data, val_data


def load_gsm8k_data(max_samples: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load GSM8K dataset using DatasetConfig.

    Args:
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (train_data, val_data) where each is a list of (question, final_answer) tuples
    """
    print(f"Loading GSM8K dataset using DatasetConfig...")

    adapter = DatasetConfig.get("gsm8k")

    # Load train split
    train_dataset = DatasetConfig.load("gsm8k", max_samples=max_samples, split="train")
    train_data = []

    for sample in train_dataset:
        question, cot_plain, final = adapter.extract_pieces(sample)
        if not final:
            continue
        # Clean up final answer
        final = final.strip().replace(",", "")
        train_data.append((question, final))

    # Load test split for validation
    val_samples = max_samples // 10
    val_dataset = DatasetConfig.load("gsm8k", max_samples=val_samples, split="test")
    val_data = []

    for sample in val_dataset:
        question, cot_plain, final = adapter.extract_pieces(sample)
        if not final:
            continue
        # Clean up final answer
        final = final.strip().replace(",", "")
        val_data.append((question, final))

    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples from GSM8K dataset")

    return train_data, val_data


def load_gsm8k_ground_truth(split: str = "train", max_samples: Optional[int] = None) -> Dict:
    """Load ground truth answers from GSM8K dataset for accuracy evaluation.

    Args:
        split: Which split to load ('train' or 'test')
        max_samples: Maximum number of samples to load

    Returns:
        Dictionary mapping sample IDs to numerical answers
    """
    try:
        adapter = DatasetConfig.get("gsm8k")
        dataset = DatasetConfig.load("gsm8k", max_samples=max_samples, split=split)

        answers = {}
        for i, sample in enumerate(dataset):
            question, cot_plain, final = adapter.extract_pieces(sample)
            if not final:
                continue

            # Clean up final answer and convert to float
            final_num = float(final.strip().replace(",", ""))

            # Store with both integer and string keys for compatibility
            answers[i] = final_num
            answers[f"hf-{i}"] = final_num

        print(f"Loaded {len(answers) // 2} GSM8K ground truth answers from {split} split")
        return answers
    except Exception as e:
        print(f"Warning: Could not load GSM8K dataset: {e}")
        return {}


def load_theory_of_mind_ground_truth(split: str = "test", max_samples: Optional[int] = None) -> Dict:
    """Load ground truth answers from Theory of Mind dataset for accuracy evaluation.

    Args:
        split: Which split to load ('train' or 'test')
        max_samples: Maximum number of samples to load

    Returns:
        Dictionary mapping sample IDs to text answers
    """
    try:
        adapter = DatasetConfig.get("theory_of_mind")
        dataset = DatasetConfig.load("theory_of_mind")

        # Convert dataset to list for splitting
        all_samples = list(dataset)
        total_samples = len(all_samples)
        split_point = int(0.8 * total_samples)

        # Split into train/test
        if split == "train":
            samples = all_samples[:split_point]
            offset = 0
        elif split == "test":
            samples = all_samples[split_point:]
            offset = split_point
        else:
            samples = all_samples
            offset = 0

        # Apply max_samples limit
        if max_samples:
            samples = samples[:max_samples]

        answers = {}
        for local_idx, sample in enumerate(samples):
            question, _, answer = adapter.extract_pieces(sample)
            answer = str(answer).strip()

            # Calculate the original index
            original_idx = offset + local_idx

            # Store with multiple key formats for compatibility
            answers[local_idx] = answer
            answers[f"hf-{local_idx}"] = answer
            answers[original_idx] = answer
            answers[f"hf-{original_idx}"] = answer

        print(f"Loaded {len(samples)} Theory of Mind ground truth answers from {split} split")
        return answers
    except Exception as e:
        print(f"Warning: Could not load Theory of Mind dataset: {e}")
        return {}