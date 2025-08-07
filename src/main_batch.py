"""
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Reliance --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Internalized --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --model2=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --metric=Transferability --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Paraphrasability --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=ParaphrasabilitySimple --data-hf=GSM8K --max-samples=2
"""

import argparse
import os
import json
from typing import List, Iterator
from datasets import Dataset

from model import CoTModel
from metrics import construct_metric
from data_loader import load_prompts
from datetime import datetime
from config import (
    DatasetConfig,
    CACHE_DIR_DEFAULT,
    LOG_EVERY_DEFAULT,
    LOG_DIRECTORY_DEFAULT,
)


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


def _iterate_dataset(dataset: Dataset) -> Iterator[tuple[int, str]]:
    for i, d in enumerate(dataset):
        yield (i, d["question"])


def _iterate_local_dataset(prompts: List[dict]) -> Iterator[str]:
    for p in prompts:
        yield (p["prompt_id"], _get_sample_question(p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model2", default=None)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-hf", default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--log-dir", default=LOG_DIRECTORY_DEFAULT)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)
    parser.add_argument("--log-verbose", type=bool, default=True)
    args = parser.parse_args()

    # Load dataset
    dataset_name = ""
    if args.data_hf:
        dataset_name = args.data_hf
        if args.max_samples:
            dataset = DatasetConfig.load(
                dataset_name, split=f"train[:{args.max_samples}]"
            )
        else:
            dataset = DatasetConfig.load(dataset_name, split="train")

        datapoints = _iterate_dataset(dataset)
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
    metric = construct_metric(
        metric_name=args.metric, model=model, alternative_model=model2
    )

    if args.log_file is None:
        log_file = (
            args.log_dir
            + "/"
            + dataset_name
            + "_"
            + _get_datetime_str()
            + "_"
            + args.metric
            + ".log"
        )
        json_log_file = (
            args.log_dir
            + "/"
            + dataset_name
            + "_"
            + _get_datetime_str()
            + "_"
            + args.metric
            + ".jsonl"
        )
        os.makedirs(args.log_dir, exist_ok=True)
    else:
        log_file = args.log_file
        json_log_file = log_file + ".jsonl"

    with open(log_file, "a") as f:
        with open(json_log_file, "a") as f_json:
            log_counter = 0
            for i, (id, question) in enumerate(datapoints):
                if i < args.skip_samples:
                    continue

                try:
                    r = model.generate_cot_response_full(id, question)
                    r.prompt_id = id
                except RuntimeError as err:
                    print(f"Sample id={id} - generation error ({err})")
                    continue

                try:
                    (score, score_original, score_intervention) = metric.evaluate(r)
                except RuntimeError as err:
                    print(f"Sample id={id} - metric evaluation error ({err})")
                    continue

                if log_counter % args.log_every == 0:
                    print(f"Sample id={id} - {score:.4f}")
                log_counter += 1

                print(
                    f"{id}\t{score:.4f}\t{score_original:.4f}\t{score_intervention:.4f}"
                )

                f.write(
                    f"{id}\t{score:.4f}\t{score_original:.4f}\t{score_intervention:.4f}\n"
                )
                f.flush()

                output = {
                    "prompt_id": id,
                    "orig_lp": float(score_original),
                    "induced_lp": float(score_intervention),
                    "delta": float(score),
                }
                if args.log_verbose:
                    output.update(
                        {
                            "question": question,
                            "cot": r.cot,
                            "answer": r.answer,
                        }
                    )
                f_json.write(json.dumps(output) + "\n")
                f_json.flush()


if __name__ == "__main__":
    main()
