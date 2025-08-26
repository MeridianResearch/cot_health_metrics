#!/usr/bin/env python3
"""
Analyze accuracy from JSON log files
Usage: python analyze_accuracy.py <json_log_file>
"""

import json
import sys
import re
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


def load_gsm8k_answers(split="test", max_samples=None):
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


def analyze_log_file(log_file_path):
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

    # Determine which split was used based on the ID format
    # If IDs are like "hf-0", it's from HuggingFace dataset
    first_id = results[0].get('id', '')
    split = 'test'  # Default to test split

    # Check if this appears to be from GSM8K test set
    if 'hf-' in first_id:
        # For HF datasets, we typically use test split
        split = 'test'
        print(f"Detected HuggingFace dataset format, using '{split}' split for ground truth")

    # Load ground truth answers from the correct split
    ground_truth = load_gsm8k_answers(split=split, max_samples=len(results))

    # Analyze results
    total = len(results)
    correct = 0
    has_cot = 0

    # Track internalized validation if present
    internalized_valid = 0
    has_internalized_check = False

    print(f"\nAnalyzing: {log_file_path}")
    print(f"{'=' * 60}")
    print(f"Model: {results[0].get('model', {}).get('base', 'Unknown')}")
    if results[0].get('model', {}).get('adapter'):
        print(f"Adapter: {results[0].get('model', {}).get('adapter')}")
    print(f"Total samples: {total}")
    print(f"\nSample results:")
    print(f"{'=' * 60}")

    for i, result in enumerate(results[:min(10, len(results))]):  # Show first 10 samples
        # Get the ID from the result
        result_id = result.get('id', f'hf-{i}')

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

        # If not found, try extracting index from ID
        if actual is None and 'hf-' in result_id:
            try:
                idx = int(result_id.split('-')[1])
                actual = ground_truth.get(idx)
            except (ValueError, IndexError):
                pass

        # Check if correct
        is_correct = False
        if predicted is not None and actual is not None:
            # Allow small tolerance for floating point comparison
            is_correct = abs(predicted - actual) < 0.01
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
        'model': results[0].get('model', {})
    }


def compare_results(log_files):
    """Compare results from multiple log files"""

    all_results = {}

    for log_file in log_files:
        if Path(log_file).exists():
            result = analyze_log_file(log_file)
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
    if len(sys.argv) < 2:
        print("Usage: python analyze_accuracy.py <json_log_file> [additional_log_files...]")
        print("\nExample:")
        print("  python analyze_accuracy.py results/gsm8k_accuracy_test/preds.jsonl")
        print("\nCompare multiple:")
        print("  python analyze_accuracy.py results/*.jsonl")
        sys.exit(1)

    log_files = sys.argv[1:]

    if len(log_files) == 1:
        analyze_log_file(log_files[0])
    else:
        compare_results(log_files)