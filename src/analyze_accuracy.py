#!/usr/bin/env python3
"""
Analyze accuracy from JSON log files for multiple datasets
Supports: GSM8K, Theory of Mind

Usage examples:
# GSM8K analysis
python src/analyze_accuracy.py \
   --dataset gsm8k \
   --split test \
   results/gsm8k_test/preds.jsonl

# Theory of Mind analysis
python src/analyze_accuracy.py \
   --dataset theory_of_mind \
   --split test \
   results/tom_test/preds.jsonl

# Compare multiple runs
python src/analyze_accuracy.py \
   --dataset gsm8k \
   --split train \
   results/run1/preds.jsonl results/run2/preds.jsonl
"""

import json
import sys
import re
import argparse
from pathlib import Path

# Import ground truth loading functions from data_loader
from data_loader import load_gsm8k_ground_truth, load_theory_of_mind_ground_truth


def extract_number(text):
    """Extract numerical answer from text"""
    # Remove commas and extra spaces
    text = text.replace(",", "").strip()

    # Try to find numbers in the text
    # Match integers and decimals
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)

    if matches:
        # Return the last number found (usually the answer)
        return float(matches[-1])
    return None


def normalize_text_answer(text):
    """Normalize text answer for comparison"""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower().strip()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common punctuation at the end
    text = text.rstrip('.,!?;:')

    return text


def compare_answers(predicted, actual, dataset_type="gsm8k"):
    """Compare predicted and actual answers based on dataset type

    Args:
        predicted: Predicted answer (string)
        actual: Ground truth answer (string or number)
        dataset_type: Type of dataset ('gsm8k' or 'theory_of_mind')

    Returns:
        bool: True if answers match, False otherwise
    """
    if predicted is None or actual is None:
        return False

    if dataset_type == "gsm8k":
        # Numerical comparison for GSM8K
        predicted_num = extract_number(str(predicted))
        if predicted_num is not None and actual is not None:
            try:
                actual_num = float(actual)
                # Allow small tolerance for floating point comparison
                return abs(predicted_num - actual_num) <= 1
            except (ValueError, TypeError):
                return False
        return False

    elif dataset_type == "theory_of_mind":
        # Text comparison for Theory of Mind
        pred_normalized = normalize_text_answer(str(predicted))
        actual_normalized = normalize_text_answer(str(actual))

        # Exact match after normalization
        if pred_normalized == actual_normalized:
            return True

        # Check if predicted contains the actual answer
        if actual_normalized in pred_normalized:
            return True

        # Check if actual contains the predicted answer (for short answers)
        if len(pred_normalized) > 0 and pred_normalized in actual_normalized:
            return True

        return False

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def analyze_log_file(log_file_path, dataset_type="gsm8k", split="train"):
    """Analyze a JSON log file and calculate accuracy

    Args:
        log_file_path: Path to the JSONL predictions file
        dataset_type: Type of dataset ('gsm8k' or 'theory_of_mind')
        split: Which split to use for ground truth
    """

    results = []

    # Read the JSONL file
    with open(log_file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                results.append(data)
            except json.JSONDecodeError:
                continue

    if not results:
        print(f"No valid results found in {log_file_path}")
        return

    # Load ground truth answers based on dataset type using data_loader
    if dataset_type == "gsm8k":
        ground_truth = load_gsm8k_ground_truth(split=split, max_samples=len(results))
        print(f"Using GSM8K '{split}' split for ground truth comparison")
    elif dataset_type == "theory_of_mind":
        ground_truth = load_theory_of_mind_ground_truth(split=split, max_samples=len(results))
        print(f"Using Theory of Mind '{split}' split for ground truth comparison")
    else:
        print(f"Unknown dataset type: {dataset_type}")
        return

    # Analyze results
    total = len(results)
    correct = 0
    has_cot = 0

    # Track internalized validation if present
    internalized_valid = 0
    has_internalized_check = False

    print(f"\nAnalyzing: {log_file_path}")
    print(f"{'=' * 60}")

    # Handle both string and dict formats for model field
    model_info = results[0].get('model', 'Unknown')
    if isinstance(model_info, dict):
        model_name = model_info.get('base', 'Unknown')
        adapter_name = model_info.get('adapter')
    else:
        model_name = str(model_info)
        adapter_name = None

    print(f"Dataset: {dataset_type}")
    print(f"Model: {model_name}")
    if adapter_name:
        print(f"Adapter: {adapter_name}")
    print(f"Total samples: {total}")
    print(f"Ground truth split: {split}")

    print(f"\nSample results:")
    print(f"{'=' * 60}")

    for i, result in enumerate(results[:min(10, len(results))]):  # Show first 10 samples
        # Get the ID from the result
        result_id = result.get('prompt_id', result.get('id', i))

        question = result.get('question', '')[:100]
        if len(result.get('question', '')) > 100:
            question += '...'

        cot = result.get('cot', '')
        answer = result.get('answer', '')

        # Check if CoT exists
        if cot and cot.strip():
            has_cot += 1

        # Check internalized validation if present
        internalized_val = result.get('internalized_validation', {})
        if internalized_val.get('checked'):
            has_internalized_check = True
            if internalized_val.get('is_internalized'):
                internalized_valid += 1

        # Get ground truth using the result ID
        actual = ground_truth.get(result_id)

        # If not found, try different ID formats
        if actual is None:
            try:
                if isinstance(result_id, str) and result_id.isdigit():
                    actual = ground_truth.get(int(result_id))
                elif isinstance(result_id, str) and 'hf-' in result_id:
                    idx = int(result_id.split('-')[1])
                    actual = ground_truth.get(idx)
                    if actual is None:
                        actual = ground_truth.get(f"hf-{idx}")
                elif isinstance(result_id, int):
                    actual = ground_truth.get(result_id)
                    if actual is None:
                        actual = ground_truth.get(f"hf-{result_id}")
            except (ValueError, IndexError):
                pass

        # Compare answers based on dataset type
        is_correct = compare_answers(answer, actual, dataset_type)
        if is_correct:
            correct += 1

        print(f"\nSample {i + 1} (ID: {result_id}):")
        print(f"  Question: {question}")
        print(f"  Has CoT: {'Yes' if cot else 'No'}")
        if has_internalized_check:
            is_valid = internalized_val.get('is_internalized', False)
            filler_ratio = internalized_val.get('filler_analysis', {}).get('filler_ratio', 0)
            print(f"  Internalized: {'Valid' if is_valid else 'Invalid'} (ratio: {filler_ratio:.2%})")

        if dataset_type == "gsm8k":
            predicted_num = extract_number(str(answer))
            print(f"  Answer: {answer[:100]}")
            print(f"  Extracted Number: {predicted_num}")
            print(f"  Ground Truth: {actual}")
        else:
            print(f"  Answer: {answer[:150]}")
            print(f"  Ground Truth: {str(actual)[:150]}")

        print(f"  Correct: {'✓' if is_correct else '✗'}")

    # Process remaining results without printing
    for i in range(min(10, len(results)), len(results)):
        result = results[i]
        result_id = result.get('prompt_id', result.get('id', i))

        cot = result.get('cot', '')
        answer = result.get('answer', '')

        if cot and cot.strip():
            has_cot += 1

        internalized_val = result.get('internalized_validation', {})
        if internalized_val.get('checked'):
            has_internalized_check = True
            if internalized_val.get('is_internalized'):
                internalized_valid += 1

        # Get ground truth
        actual = ground_truth.get(result_id)
        if actual is None:
            try:
                if isinstance(result_id, str) and result_id.isdigit():
                    actual = ground_truth.get(int(result_id))
                elif isinstance(result_id, str) and 'hf-' in result_id:
                    idx = int(result_id.split('-')[1])
                    actual = ground_truth.get(idx)
                    if actual is None:
                        actual = ground_truth.get(f"hf-{idx}")
                elif isinstance(result_id, int):
                    actual = ground_truth.get(result_id)
                    if actual is None:
                        actual = ground_truth.get(f"hf-{result_id}")
            except (ValueError, IndexError):
                pass

        is_correct = compare_answers(answer, actual, dataset_type)
        if is_correct:
            correct += 1

    # Calculate statistics
    accuracy = (correct / total) * 100 if total > 0 else 0
    cot_percentage = (has_cot / total) * 100 if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset_type}")
    print(f"Total samples: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Samples with CoT: {has_cot} ({cot_percentage:.2f}%)")
    print(f"Ground truth from: {dataset_type} {split} split")

    if has_internalized_check:
        internalized_percentage = (internalized_valid / total) * 100 if total > 0 else 0
        print(f"Valid internalized CoT: {internalized_valid} ({internalized_percentage:.2f}%)")

    # Determine if this is a no-CoT run
    if cot_percentage < 10:
        print(f"\n⚠️  This appears to be a NO-COT run (only {cot_percentage:.1f}% have CoT)")
        print(f"   Expected: Low accuracy without chain-of-thought reasoning")
    else:
        print(f"\n✓  This appears to be a WITH-COT run ({cot_percentage:.1f}% have CoT)")
        print(f"   Expected: Higher accuracy with chain-of-thought reasoning")

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'has_cot_percentage': cot_percentage,
        'internalized_valid_percentage': (internalized_valid / total * 100) if has_internalized_check else None,
        'model': model_info if isinstance(model_info, dict) else {'base': model_info},
        'split_used': split,
        'dataset_type': dataset_type
    }


def compare_results(log_files, dataset_type="gsm8k", split="train"):
    """Compare results from multiple log files"""

    all_results = {}

    for log_file in log_files:
        if Path(log_file).exists():
            result = analyze_log_file(log_file, dataset_type=dataset_type, split=split)
            if result:
                all_results[log_file] = result

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print(f"COMPARISON ACROSS ALL RUNS")
        print(f"{'=' * 60}")

        for log_file, stats in all_results.items():
            print(f"\n{Path(log_file).name}")
            print(f"  Model: {stats['model'].get('base', 'Unknown')}")
            print(f"  Accuracy: {stats['accuracy']:.2f}%")
            print(f"  Has CoT: {stats['has_cot_percentage']:.1f}%")
            if stats.get('internalized_valid_percentage') is not None:
                print(f"  Valid Internalized: {stats['internalized_valid_percentage']:.1f}%")

        # Find best and worst
        best = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        worst = min(all_results.items(), key=lambda x: x[1]['accuracy'])

        print(f"\n{'=' * 60}")
        print(f"Best performance: {Path(best[0]).name}")
        print(f"  Accuracy: {best[1]['accuracy']:.2f}%")
        print(f"\nWorst performance: {Path(worst[0]).name}")
        print(f"  Accuracy: {worst[1]['accuracy']:.2f}%")

        # Calculate improvement
        if worst[1]['accuracy'] > 0:
            improvement = ((best[1]['accuracy'] - worst[1]['accuracy']) / worst[1]['accuracy']) * 100
            print(f"\nImprovement from worst to best: {improvement:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze accuracy from JSON log files for multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Analyze GSM8K results
  python analyze_accuracy.py --dataset gsm8k --split test results/gsm8k_test/preds.jsonl

  # Analyze Theory of Mind results
  python analyze_accuracy.py --dataset theory_of_mind --split test results/tom_test/preds.jsonl

  # Compare multiple runs
  python analyze_accuracy.py --dataset gsm8k --split train results/run1/preds.jsonl results/run2/preds.jsonl
  """)

    parser.add_argument("log_files", nargs='+', help="JSON log file(s) to analyze")

    parser.add_argument("--dataset",
                        choices=["gsm8k", "theory_of_mind"],
                        default="gsm8k",
                        help="Dataset type to analyze (default: gsm8k)")

    parser.add_argument("--split",
                        choices=["train", "test"],
                        default="train",
                        help="Dataset split to use for ground truth answers (default: train)")

    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Using '{args.split}' split for ground truth answers")

    if len(args.log_files) == 1:
        analyze_log_file(args.log_files[0],
                         dataset_type=args.dataset,
                         split=args.split)
    else:
        compare_results(args.log_files,
                        dataset_type=args.dataset,
                        split=args.split)