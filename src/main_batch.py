import argparse
import os
import time
from typing import List, Iterator
from datasets import load_dataset, Dataset

from model import Model
from metric import Metric, DummyMetric
from metric_reliance import RelianceMetric
from metric_paraphrasability import ParaphrasabilityMetric
from metric_transferability import TransferabilityMetric
from data_loader import load_prompts
from datetime import datetime

CACHE_DIR_DEFAULT        = "hf_cache"
LOG_EVERY_DEFAULT        = 1

HF_DATASET_NAMES = {
    "Alpaca": "vicgalle/alpaca-gpt4",
    "GSM8K": "gsm8k",
    "MMLU": "openai/openai_mmlu_benchmark",
}

METRIC_CLASSES = {
    "Dummy": DummyMetric,
    "Reliance": RelianceMetric,
    "Paraphrasability": ParaphrasabilityMetric,
    "Transferability": TransferabilityMetric,
}


# Current datetime
now = datetime.now()

# Format as string

def _get_datetime_str():
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(datetime_str)
    return datetime_str
def _get_sample_question(sample: dict) -> str:
    question = sample["instruction"].strip()
    if sample.get("input"):
        question += " " + sample["input"].strip()
    return question

def _iterate_dataset(dataset: Dataset) -> Iterator[str]:
    for i, d in enumerate(dataset):
        yield (i, d['question'])

def _iterate_local_dataset(prompts: List[dict]) -> Iterator[str]:
    for p in prompts:
        yield (p['prompt_id'], _get_sample_question(p))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model2", default=None)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-hf", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)
    args = parser.parse_args()

    # Load dataset
    dataset_name = ''
    if args.data_hf:
        dataset_name = args.data_hf
        if args.max_samples:
            dataset = load_dataset(HF_DATASET_NAMES[args.data_hf], "main", split=f"train[:{args.max_samples}]")
        else:
            dataset = load_dataset(HF_DATASET_NAMES[args.data_hf], "main", split="train")

        datapoints = _iterate_dataset(dataset)
    elif args.data_path:
        dataset_name = os.path.basename(args.data_path)
        prompts: List[dict] = load_prompts(args.data_path, args.max_samples)

        datapoints = _iterate_local_dataset(prompts)
    else:
        raise ValueError("Either --data-hf or --data-path must be provided")

    #os.makedirs(args.cache_dir, exist_ok=True)

    # Load model
    model = Model(args.model, cache_dir=args.cache_dir)

    # Create metric(s)
    construct_metric = METRIC_CLASSES[args.metric]
    metric = construct_metric(
        model_name=args.model,
        alternative_model_name=args.model2)
    if args.log_file is None:
        file_name = dataset_name + "-" + _get_datetime_str() + "-" + args.metric
    else:
        file_name = args.log_file
    with open(file_name, 'a') as f:
        log_counter = 0
        for id, question in datapoints:
            try:
                r = model.generate_cot_response_full(question)
            except RuntimeError as err:
                print(f"Sample id={id} - generation error ({err})")
                continue

            score = metric.evaluate(r)
            try:
                (score, score_original, score_intervention) = score
            except:
                (score, score_original, score_intervention) = (score, -1, -1)

            if log_counter % args.log_every == 0:
                print(f"Sample id={id} - {score:.4f}")
            log_counter += 1
            print(f"{id}\t{score:.4f}\t{score_original:.4f}\t{score_intervention:.4f}")
            f.write(f"{id}\t{score:.4f}\t{score_original:.4f}\t{score_intervention:.4f}\n")

if __name__ == "__main__":
    main()
