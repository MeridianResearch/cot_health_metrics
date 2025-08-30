#!/usr/bin/env python3
"""
Analyze accuracy from JSON log files
Usage: python analyze_accuracy.py <json_log_file>
"""

import json
import sys
import re
import argparse
from pathlib import Path


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


def load_gsm8k_answers(split="train", max_samples=None):  # FIXED: Changed default to "train"
    """Load ground truth answers from GSM8K dataset

    Args:
        split: Which split to load ('train' or 'test')
        max_samples: Maximum number of samples to load
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split=split)

        answers = {}
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            # Extract numerical answer from the ground truth
            ground_truth = item['answer'].split('####')[-1].strip()

            # Store with both integer and string keys for compatibility
            answers[i] = float(ground_truth.replace(",", ""))
            answers[f"hf-{i}"] = float(ground_truth.replace(",", ""))  # Also store with hf- prefix

        return answers
    except Exception as e:
        print(f"Warning: Could not load GSM8K dataset: {e}")
        return {}


def analyze_log_file(log_file_path, split="train"):
    """Analyze a JSON log file and calculate accuracy"""

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

    # Load ground truth answers from the specified split
    ground_truth = load_gsm8k_answers(split=split, max_samples=len(results))

    print(f"Using GSM8K '{split}' split for ground truth comparison")

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
        # model is a string
        model_name = str(model_info)
        adapter_name = None

    print(f"Model: {model_name}")
    if adapter_name:
        print(f"Adapter: {adapter_name}")
    print(f"Total samples: {total}")
    print(f"Ground truth split: {split}")

    print(f"\nSample results:")
    print(f"{'=' * 60}")

    for i, result in enumerate(results[:len(results)]):
        # Get the ID from the result - check both prompt_id and id fields
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

        # Extract numerical answer
        predicted = extract_number(answer)

        # Get ground truth using the result ID
        actual = ground_truth.get(result_id)

        # If not found, try different ID formats
        if actual is None:
            # Try as integer if result_id is string
            try:
                if isinstance(result_id, str) and result_id.isdigit():
                    actual = ground_truth.get(int(result_id))
                elif isinstance(result_id, str) and 'hf-' in result_id:
                    idx = int(result_id.split('-')[1])
                    actual = ground_truth.get(idx)
                elif isinstance(result_id, int):
                    actual = ground_truth.get(result_id)
            except (ValueError, IndexError):
                pass

        # Check if correct
        is_correct = False
        if predicted is not None and actual is not None:
            # Allow small tolerance for floating point comparison
            is_correct = abs(predicted - actual) <= 1
            if is_correct:
                correct += 1

        print(f"\nSample {i + 1} (ID: {result_id}):")
        print(f"  Question: {question}")
        print(f"  Has CoT: {'Yes' if cot else 'No'}")
        if has_internalized_check:
            is_valid = internalized_val.get('is_internalized', False)
            filler_ratio = internalized_val.get('filler_analysis', {}).get('filler_ratio', 0)
            print(f"  Internalized: {'Valid' if is_valid else 'Invalid'} (ratio: {filler_ratio:.2%})")
        print(f"  Answer: {answer[:100]}")
        print(f"  Extracted: {predicted}")
        print(f"  Ground Truth: {actual}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")

    # Calculate statistics
    accuracy = (correct / total) * 100 if total > 0 else 0
    cot_percentage = (has_cot / total) * 100 if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total samples: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Samples with CoT: {has_cot} ({cot_percentage:.2f}%)")
    print(f"Ground truth from: GSM8K {split} split")

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
        'split_used': split
    }

def compare_results(log_files, split="train"):
    """Compare results from multiple log files"""

    all_results = {}

    for log_file in log_files:
        if Path(log_file).exists():
            result = analyze_log_file(log_file, split=split)
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


def compare_results_with_split(log_files, split="train"):
    """Wrapper function for backward compatibility"""
    return compare_results(log_files, split=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze accuracy from GSM8K JSON log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python analyze_accuracy.py results/gsm8k_accuracy_test/preds.jsonl
  python analyze_accuracy.py --split test results/test_run.jsonl
  python analyze_accuracy.py --split train results/*.jsonl"""
    )

    parser.add_argument("log_files", nargs='+', help="JSON log file(s) to analyze")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="GSM8K split to use for ground truth answers (default: train)")

    args = parser.parse_args()

    print(f"Using GSM8K '{args.split}' split for ground truth answers")

    if len(args.log_files) == 1:
        analyze_log_file(args.log_files[0], split=args.split)
    else:
        compare_results_with_split(args.log_files, split=args.split)