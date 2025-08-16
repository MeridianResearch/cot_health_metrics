import argparse
import os
import sys
import json
from typing import List, Iterator
from datetime import datetime
from config import DatasetConfig
from data_loader import load_prompts
from datasets import Dataset

from model import CoTModel
from model_factory import ModelComponentFactory
from model_prompts import CustomInstructionPromptBuilder
from config import CACHE_DIR_DEFAULT, LOG_DIRECTORY_DEFAULT, LOG_EVERY_DEFAULT, ORGANISM_DEFAULT_MODEL
from all_organisms import OrganismRegistry
from icl_organism import ICLOrganism

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

def print_output(id, question, prompt, cot, answer, f, f_json, args, organism):
    print(f"{id}\t{question}")

    f.write(f"{id}\t{question}")
    f.flush()

    output = {
        "prompt_id": id,
        "organism": organism.get_name(),
        "model": args.model if args.model else organism.get_default_model_name(),
        "question": question,
        "prompt": prompt,
        "cot": cot,
        "answer": answer
    }
    f_json.write(json.dumps(output) + "\n")
    f_json.flush()

def handle_datapoints(datapoints, args, model, f, f_json, organism):
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

        if log_counter % args.log_every == 0:
            #print(f"Sample id={id} - {result.score:.4f}")
            pass
        log_counter += 1

        print_output(id, question, r.prompt, r.cot, r.answer, f, f_json, args, organism)


def create_dynamic_icl_organism(filler_type: str = "think",
                                examples_file: str = None):
    """Factory function to create ICL organisms dynamically"""

    filename = os.path.basename(examples_file)
    if "dot" in filename:
        filler_type = "dot"
    elif "comma" in filename:
        filler_type = "comma"
    elif "lorem" in filename:
        filler_type = "lorem_ipsum"
    elif "think" in filename:
        filler_type = "think_token"

    name = f"icl-{filler_type}"

    return ICLOrganism(
        name=name,
        default_model_name=ORGANISM_DEFAULT_MODEL,
        filler_type=filler_type,
        examples_file=examples_file
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organism", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model2", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-hf", default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--log-dir", default=LOG_DIRECTORY_DEFAULT)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY_DEFAULT)
    parser.add_argument("--log-verbose", type=bool, default=True)

    parser.add_argument("--filler", type=str, default="think")  # Internalized
    parser.add_argument("--filler-in-prompt", type=bool, default=False)  # Internalized
    parser.add_argument("--filler-in-cot", type=bool, default=False)  # Internalized

    parser.add_argument("--icl-filler", type=str, default=None,
                        help="Filler type for ICL (think, dot, comma, etc.)")
    parser.add_argument("--icl-examples-file", type=str, default=None,
                        help="Path to JSON file with ICL examples")

    organism_registry = OrganismRegistry()

    try:
        args = parser.parse_args()
    except SystemExit:
        print("\nValid organisms (with default model):")
        for (name, organism) in organism_registry.get_all():
            print("    %-30s %s" % (name, organism.get_default_model_name()))
        sys.exit(1)

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

        # Handle ICL organism creation
    if args.icl_examples_file and args.organism == "icl-custom":
        # Auto-detect filler type from file and create dynamic organism
        dynamic_organism = create_dynamic_icl_organism(
            filler_type=args.icl_filler or "auto-detect",
            examples_file=args.icl_examples_file
        )
        organism_registry.add(dynamic_organism)
        args.organism = dynamic_organism.get_name()
    elif args.icl_filler:
        # Manual filler type specification
        dynamic_organism = create_dynamic_icl_organism(
            filler_type=args.icl_filler,
            examples_file=args.icl_examples_file
        )
        organism_registry.add(dynamic_organism)
        args.organism = dynamic_organism.get_name()

    # Make cache dir
    os.makedirs(args.cache_dir, exist_ok=True)

    # Get organism
    organism = organism_registry.get(args.organism)
    if organism is None:
        raise ValueError(f"Organism {args.organism} not found")

    # Get model name
    model_name = args.model
    if model_name is None or model_name == "":
        model_name = organism.get_default_model_name()

    # Load model
    component_factory = organism.get_component_factory(model_name)
    model = CoTModel(model_name, component_factory=component_factory, cache_dir=args.cache_dir)

    if args.log_file is None:
        base = args.log_dir + "/" + organism.get_name() + "_" + model_name + "_" + dataset_name + "_" + _get_datetime_str()
        log_file = base + ".log"
        json_log_file = base + ".jsonl"
        config_log_file = base + ".config.json"
    else:
        log_file = args.log_file
        json_log_file = log_file + ".jsonl"
        config_log_file = log_file + ".config.jsonl"

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'a') as f:
        with open(json_log_file, 'a') as f_json:
            handle_datapoints(datapoints, args, model, f, f_json, organism)

if __name__ == "__main__":
    main()