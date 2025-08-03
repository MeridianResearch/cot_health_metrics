"""
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Reliance --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Internalized --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --model2=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --metric=Transferability --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Paraphrasability --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=ParaphrasabilitySimple --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=PromptParaphrasability --data-hf=GSM8K --max-samples=2
"""

import argparse
import os
import json
import itertools
from typing import List, Iterator
from datasets import Dataset

from model import CoTModel
from all_metrics import construct_metric
from data_loader import load_prompts
from metric import SampleGroundTruth
from datetime import datetime
from config import DatasetAdapter, DatasetConfig, CACHE_DIR_DEFAULT, LOG_EVERY_DEFAULT, LOG_DIRECTORY_DEFAULT

#from itertools import batched  # only available in Python 3.12+
# Custom batched implementation for Python < 3.12
def batched(iterable, n):
    """Split an iterable into batches of size n."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


# Current datetime
now = datetime.now()

# Format as string

def _get_datetime_str():
    datetime_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    print(datetime_str)
    return datetime_str
def _get_sample_question(sample: dict) -> str:
    question = sample["instruction"].strip()
    if sample.get("input"):
        question += " " + sample["input"].strip()
    return question

def _iterate_dataset(dataset_name: str, dataset: Dataset) -> Iterator[tuple[int, str, str, str]]:
    adapter = DatasetConfig.get(dataset_name)
    for i, d in enumerate(dataset):
        pieces = adapter.extract_pieces(d)
        yield (i, *pieces)

def _iterate_local_dataset(prompts: List[dict]) -> Iterator[tuple[int, str, str, str]]:
    for p in prompts:
        yield (p['prompt_id'], _get_sample_question(p), '', '')

def print_output(id, question, cot, answer, score, score_original, score_intervention, f, f_json, args):
    print(f"{id}\t{score:.4f}\t{score_original:.4f}\t{score_intervention:.4f}")

    f.write(f"{id}\t{score:.4f}\t{score_original:.4f}\t{score_intervention:.4f}\n")
    f.flush()

    output = {
        "prompt_id": id,
        "orig_lp": float(score_original),
        "induced_lp": float(score_intervention),
        "delta": float(score),
    }
    if args.log_verbose:
        output.update({
            "question": question,
            "cot": cot,
            "answer": answer,
        })
    f_json.write(json.dumps(output) + "\n")
    f_json.flush()

def handle_datapoints(datapoints, args, model, metric, f, f_json):
    log_counter = 0
    for i, (id, question, ground_truth_cot, ground_truth_answer) in enumerate(datapoints):
        if i < args.skip_samples:
            continue

        try:
            r = model.generate_cot_response_full(id, question)
            r.prompt_id = id
        except RuntimeError as err:
            print(f"Sample id={id} - generation error ({err})")
            continue

        try:
            if ground_truth_cot != '' and ground_truth_answer != '':
                ground_truth = SampleGroundTruth(cot=ground_truth_cot, answer=ground_truth_answer)
                (score, score_original, score_intervention) = metric.evaluate(r, ground_truth=ground_truth)
            else:
                (score, score_original, score_intervention) = metric.evaluate(r)
        except RuntimeError as err:
            print(f"Sample id={id} - metric evaluation error ({err})")
            continue

        if log_counter % args.log_every == 0:
            print(f"Sample id={id} - {score:.4f}")
        log_counter += 1

        print_output(id, question, r.cot, r.answer, score, score_original, score_intervention, f, f_json, args)

def handle_datapoints_batch(datapoints, batch_size, args, model, metric, f, f_json):
    sample_counter = 0
    for batch in batched(datapoints, batch_size):
        if sample_counter + batch_size > args.max_samples:
            batch = batch[:args.skip_samples - sample_counter]
        sample_counter += len(batch)

        question_ids = []
        questions = []
        ground_truth_cots = []
        ground_truth_answers = []

        print(f"Running batch with {len(batch)} samples")

        for id, question, ground_truth_cot, ground_truth_answer in batch:
            question_ids.append(id)
            questions.append(question)
            ground_truth_cots.append(ground_truth_cot)
            ground_truth_answers.append(ground_truth_answer)

        try:
            r = model.generate_cot_response_full_batch(question_ids, questions)
        except RuntimeError as err:
            print(f"Batch - generation error ({err})")
            continue

        have_ground_truth = False
        for i, (id, question, ground_truth_cot, ground_truth_answer) in enumerate(batch):
            r[i].prompt_id = id
            if ground_truth_cot != '' or ground_truth_answer != '':
                have_ground_truth = True

        try:
            if have_ground_truth:
                ground_truth = [SampleGroundTruth(cot=ground_truth_cot, answer=ground_truth_answer)
                    for ground_truth_cot, ground_truth_answer in zip(ground_truth_cots, ground_truth_answers)]
            else:
                ground_truth = None
            results = metric.evaluate_batch(r, ground_truth=ground_truth)
        except RuntimeError as err:
            print(f"Batch - metric evaluation error ({err})")
            continue

        for i, (score, score_original, score_intervention) in enumerate(results):
            print_output(question_ids[i], questions[i], r[i].cot, r[i].answer, score, score_original, score_intervention, f, f_json, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model2", default=None)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-hf", default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--filler", type=str, default="think")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--log-dir", default="data/logprobs/json")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)
    parser.add_argument("--log-verbose", type=bool, default=True)
    args = parser.parse_args()

    # Load dataset
    dataset_name = ''
    if args.data_hf:
        dataset_name = args.data_hf
        if args.max_samples:
            dataset = DatasetConfig.load(dataset_name, max_samples=args.max_samples)
        else:
            dataset = DatasetConfig.load(dataset_name)

        datapoints = _iterate_dataset(dataset_name, dataset)
    elif args.data_path:
        dataset_name = os.path.basename(args.data_path)
        prompts: List[dict] = load_prompts(args.data_path, args.max_samples)

        datapoints = _iterate_local_dataset(prompts)
    else:
        raise ValueError("Either --data-hf or --data-path must be provided")

    # Make cache dir
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load model
    model = CoTModel(args.model, cache_dir=args.cache_dir)
    model2 = CoTModel(args.model2, cache_dir=args.cache_dir) if args.model2 else None

    # Create metric(s)
    extra_args = {}
    if args.metric == "Internalized":
        extra_args["filler_token"] = args.filler

    metric = construct_metric(
        metric_name=args.metric,
        model=model,
        alternative_model=model2,
        **extra_args)

    if args.log_file is None:
        if args.metric == "Internalized":
            log_file = (args.log_dir + "/" + args.model + "_" + dataset_name + "_" + _get_datetime_str() + "_"
                        + args.metric + "_filler_" + args.filler + "_" + ".log")
            json_log_file = (args.log_dir + "/" + args.model + "_" + dataset_name + "_" + _get_datetime_str() + "_"
                             + args.metric + "_filler_" + args.filler + "_" + ".jsonl")
        else:
            log_file = args.log_dir + "/" + args.model + "_" + dataset_name + "_" + _get_datetime_str() + "_" + args.metric + ".log"
            json_log_file = args.log_dir + "/" + args.model + "_" + dataset_name + "_" + _get_datetime_str() + "_" + args.metric + ".jsonl"
        os.makedirs(args.log_dir, exist_ok=True)
    else:
        log_file = args.log_file
        json_log_file = log_file + ".jsonl"

    with open(log_file, 'a') as f:
        with open(json_log_file, 'a') as f_json:
            if args.batch_size == 1:
                handle_datapoints(datapoints, args, model, metric, f, f_json)
            else:
                handle_datapoints_batch(datapoints, args.batch_size, args, model, metric, f, f_json)

if __name__ == "__main__":
    main()
