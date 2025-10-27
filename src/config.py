from random import sample
from datasets import load_dataset, Dataset
from typing import Callable, Union, Optional
import pandas as pd
import os


CACHE_DIR_DEFAULT = "hf_cache"
LOG_DIRECTORY_DEFAULT = "log"
LOG_EVERY_DEFAULT = 1
ORGANISM_DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ORGANISM_DEFAULT_NAME = "count-1-to-100"
ICL_EXAMPLES_DIRECTORY_DEFAULT_FILE = "data/icl_examples/icl_dot_default.json"
ICL_EXAMPLES_DIRECTORY_DEFAULT = "data/icl_examples"


class ModelConfig:
    MODEL_CONFIG_THINK_TOKENS = {
        "begin_think": "<think>",
        "end_think": "</think>"
    }

    MODEL_CONFIG_GPT_OSS_20B = {
        "begin_think": "<|end|><|start|>assistant<|channel|>final<|message|>analysis<|message|>",
        "end_think": "<|end|><|start|>assistant<|channel|>final<|message|>",
        "do_not_think": "<|end|><|start|>assistant<|channel|>final<|message|>",
    }
    MODEL_CONFIG_GEMMA = {
        "fuzzy_end_think_list": ["Answer:"],
    }
    MODEL_CONFIG_LLAMA = {
        "fuzzy_end_think_list": ["Answer:"],
    }
    MODEL_CONFIG_MISTRAL = {
        "fuzzy_end_think_list": ["Answer:"],
    }
    DEFAULT_MODEL_CONFIG = MODEL_CONFIG_GEMMA

    SUPPORTED_MODELS = {
        "Qwen/Qwen3-0.6B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-1.7B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-4B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen2-7B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-8B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-14B": MODEL_CONFIG_THINK_TOKENS,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": MODEL_CONFIG_THINK_TOKENS,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": MODEL_CONFIG_THINK_TOKENS,
        # "deepcogito/cogito-v1-preview-llama-3B": MODEL_CONFIG_THINK_TOKENS,  # unverified
        "Wladastic/Mini-Think-Base-1B": MODEL_CONFIG_GEMMA,
        "google/gemma-2-2b-it": MODEL_CONFIG_GEMMA,
        "google/gemma-2-9b-it": MODEL_CONFIG_GEMMA,
        "openai/gpt-oss-20b": MODEL_CONFIG_GPT_OSS_20B,
        # "microsoft/phi-2": MODEL_CONFIG_FUZZY_ANSWER,  # not very consistent
        "meta-llama/Meta-Llama-3-8B-Instruct": MODEL_CONFIG_LLAMA,
        "meta-llama/Llama-2-7b-chat-hf": MODEL_CONFIG_LLAMA,
        "mistralai/Mistral-7B-Instruct-v0.3": MODEL_CONFIG_MISTRAL,
        # merged SFT models
        "output/qwen-mixed_rank8-merged": MODEL_CONFIG_THINK_TOKENS,
        "output/qwen-no_cot_rank1-merged": MODEL_CONFIG_THINK_TOKENS,
    }

    @staticmethod
    def get(model_name: str) -> dict:
        if model_name not in ModelConfig.SUPPORTED_MODELS:
            return ModelConfig.DEFAULT_MODEL_CONFIG
        return ModelConfig.SUPPORTED_MODELS[model_name]

    @staticmethod
    def is_supported(model_name: str) -> bool:
        return model_name in ModelConfig.SUPPORTED_MODELS


class ICLConfig:
    """Configuration for ICL (In-Context Learning) organisms"""

    ICL_ORGANISM_CONFIGS = [
        {
            "name": "icl-think",
            "filler_type": "think_token",
            "examples_file": "data/icl_examples/icl_think_token_default.json"
        },
        {
            "name": "icl-dot",
            "filler_type": "dot",
            "examples_file": "data/icl_examples/icl_dot_default.json"
        },
        {
            "name": "icl-comma",
            "filler_type": "comma",
            "examples_file": "data/icl_examples/icl_comma_default.json"
        },
        {
            "name": "icl-lorem-ipsum",
            "filler_type": "lorem_ipsum",
            "examples_file": "data/icl_examples/icl_lorem_ipsum_default.json"
        }
    ]

    @staticmethod
    def get_all_configs():
        """Get all ICL organism configurations"""
        return ICLConfig.ICL_ORGANISM_CONFIGS

    @staticmethod
    def get_config_by_name(name: str):
        """Get specific ICL organism configuration by name"""
        for config in ICLConfig.ICL_ORGANISM_CONFIGS:
            if config["name"] == name:
                return config
        return None

    @staticmethod
    def get_config_by_filler_type(filler_type: str):
        """Get specific ICL organism configuration by filler type"""
        for config in ICLConfig.ICL_ORGANISM_CONFIGS:
            if config["filler_type"] == filler_type:
                return config
        return None


class DatasetAdapter:
    def __init__(self, dataset_name: str, aliases: list, load_section: Optional[str] = None,
                 load_split: str = "train", do_extract: Optional[Callable] = None,
                 is_local_csv: bool = False, train_file: Optional[str] = None,
                 test_file: Optional[str] = None, csv_has_header: bool = True):
        """
        Args:
            dataset_name: Name of the dataset (for single file) or base name (for split files)
            aliases: List of alias names for this dataset
            load_section: Section to load for HF datasets
            load_split: Default split to load
            do_extract: Function to extract (question, cot, answer) from data
            is_local_csv: Whether this is a local CSV file
            train_file: Path to training CSV file (for pre-split datasets)
            test_file: Path to test CSV file (for pre-split datasets)
            csv_has_header: Whether CSV files have headers
        """
        self.dataset_name = dataset_name
        self.aliases = aliases
        self.load_section = load_section
        self.load_split = load_split
        self.do_extract = do_extract
        self.is_local_csv = is_local_csv
        self.train_file = train_file
        self.test_file = test_file
        self.csv_has_header = csv_has_header
        # Cache for local CSV data to ensure consistent splitting
        self._cached_dataset = None
        self._cached_train_split = None
        self._cached_test_split = None

    def extract_pieces(self, data: dict):
        if self.do_extract:
            return self.do_extract(data)
        return data

    def get(self, dataset_name: str) -> str:
        return self.dataset_name

    def _load_presplit_csv(self, split: str):
        """Load CSV from pre-split train/test files"""
        if split == "train":
            csv_path = self.train_file
        elif split == "test":
            csv_path = self.test_file
        else:
            raise ValueError(f"Split '{split}' not supported for pre-split dataset. Use 'train' or 'test'.")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV with or without header
        if self.csv_has_header:
            df = pd.read_csv(csv_path)
        else:
            # For CSV without headers, read as single column
            df = pd.read_csv(csv_path, header=None, names=['raw_text'])

        dataset = Dataset.from_pandas(df)
        print(f"Loaded {len(dataset)} samples from pre-split CSV: {csv_path}")
        return dataset

    def _load_and_split_csv(self, csv_path: str, train_ratio: float = 0.8):
        """Load CSV and split into train/test with caching for consistency"""
        if self._cached_dataset is None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # Read CSV with or without header
            if self.csv_has_header:
                df = pd.read_csv(csv_path)
            else:
                # For CSV without headers, read as single column
                df = pd.read_csv(csv_path, header=None, names=['raw_text'])

            self._cached_dataset = Dataset.from_pandas(df)

            # Create consistent train/test split
            total_samples = len(self._cached_dataset)
            split_point = int(total_samples * train_ratio)

            # Split the dataset
            train_indices = list(range(split_point))
            test_indices = list(range(split_point, total_samples))

            self._cached_train_split = self._cached_dataset.select(train_indices)
            self._cached_test_split = self._cached_dataset.select(test_indices)

            print(
                f"Split local CSV {csv_path}: {len(self._cached_train_split)} train, {len(self._cached_test_split)} test")

        return self._cached_train_split, self._cached_test_split

    def _load_and_split_json(self, json_path: str, train_ratio: float = 0.8):
        """Load JSON and split into train/test with caching for consistency"""
        if self._cached_dataset is None:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            # Read JSON file
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Convert to pandas DataFrame then to Dataset
            df = pd.DataFrame(data)
            self._cached_dataset = Dataset.from_pandas(df)

            # Create consistent train/test split
            total_samples = len(self._cached_dataset)
            split_point = int(total_samples * train_ratio)

            # Split the dataset
            train_indices = list(range(split_point))
            test_indices = list(range(split_point, total_samples))

            self._cached_train_split = self._cached_dataset.select(train_indices)
            self._cached_test_split = self._cached_dataset.select(test_indices)

            print(
                f"Split local JSON {json_path}: {len(self._cached_train_split)} train, {len(self._cached_test_split)} test")

        return self._cached_train_split, self._cached_test_split

    def load(self, dataset_name: str, max_samples: Optional[int] = None, split: str = None) -> Dataset:
        # Handle local CSV files
        if self.is_local_csv:
            # Determine which split to return
            if split is None:
                split = self.load_split  # Use default

            # Check if using pre-split files
            if self.train_file and self.test_file:
                # Load from pre-split files
                dataset = self._load_presplit_csv(split)
            else:
                # Load and split a single file
                csv_path = self.dataset_name
                train_split, test_split = self._load_and_split_csv(csv_path)

                if split == "train":
                    dataset = train_split
                elif split == "test":
                    dataset = test_split
                else:
                    # For other splits or full dataset, return the full cached dataset
                    if self._cached_dataset is None:
                        if self.csv_has_header:
                            df = pd.read_csv(csv_path)
                        else:
                            df = pd.read_csv(csv_path, header=None, names=['raw_text'])
                        self._cached_dataset = Dataset.from_pandas(df)
                    dataset = self._cached_dataset

            # Apply max_samples limit
            if max_samples is not None and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))

            print(f"Loaded {len(dataset)} samples from local CSV (split: {split})")
            return dataset

        # Handle local JSON files
        if self.dataset_name.endswith('.json'):
            # Determine which split to return
            if split is None:
                split = self.load_split  # Use default

            # Load and split the JSON file
            json_path = self.dataset_name
            train_split, test_split = self._load_and_split_json(json_path)

            if split == "train":
                dataset = train_split
            elif split == "test":
                dataset = test_split
            else:
                # For other splits or full dataset, return the full cached dataset
                if self._cached_dataset is None:
                    import json
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    self._cached_dataset = Dataset.from_pandas(df)
                dataset = self._cached_dataset

            # Apply max_samples limit
            if max_samples is not None and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))

            print(f"Loaded {len(dataset)} samples from local JSON (split: {split})")
            return dataset

        # Handle HuggingFace datasets
        if split is None:
            split = self.load_split  # Use default
        if max_samples is not None:
            split = split + f"[:{max_samples}]"
        print(f"Loading dataset {dataset_name} with split {split}")
        print(f"Dataset name: {self.get(dataset_name)}")
        print(f"Stored Dataset name: {self.dataset_name}")
        dataset = load_dataset(self.get(dataset_name), self.load_section, split=split)

        return dataset


class DatasetConfig:
    @staticmethod
    def _extract_3sum(data: dict) -> tuple:
        """Extract question, CoT, and answer from 3SUM data format.

        Format: "input_tuples P intermediate_tokens A answer"
        Example: "190 545 523 ... P . . . . A True"
        """
        raw_text = data.get("raw_text", "")

        # Split by " P " to separate input from intermediate tokens
        if " P " not in raw_text:
            print(f"Warning: No 'P' separator found in: {raw_text[:100]}...")
            return ("", "", "")

        input_part, rest = raw_text.split(" P ", 1)

        # Split by " A " to separate intermediate tokens from answer
        if " A " not in rest:
            print(f"Warning: No 'A' separator found in: {rest[:100]}...")
            return ("", "", "")

        intermediate_tokens, answer = rest.rsplit(" A ", 1)

        # Format the question with the input tuples
        question = f"Given the following tuples, does there exist three distinct tuples that sum to (0,0,0) modulo 10?\nInput: {input_part.strip()}"

        # CoT is the intermediate tokens (could be dots or chain-of-thought)
        cot = intermediate_tokens.strip()

        # Answer is True or False
        answer = answer.strip()

        return (question, cot, answer)

    HF_DATASET_LIST = [
        DatasetAdapter("vicgalle/alpaca-gpt4", ["alpaca", "alpaca-gpt4"],
                       do_extract=lambda d: (d["question"], "", d["answer"])),
        DatasetAdapter("gsm8k", ["GSM8K"],
                       do_extract=lambda d: (
                           d["question"], d["answer"].split("####")[0], d["answer"].split("####")[1])),
        DatasetAdapter("cais/mmlu", ["MMLU", "mmlu"], load_section="all", load_split="test",
                       do_extract=lambda d: (d["question"] + "\nChoices: "
                                             + "\n".join([f"{chr(ord('A') + i)}: {d['choices'][i]}" for i in
                                                          range(len(d["choices"]))]),
                                             "", d["answer"])),
        # Local CSV dataset for Theory of Mind (single file with auto-split)
        DatasetAdapter("data/theory_of_mind.csv", ["theory_of_mind", "TheoryOfMind", "tom"],
                       do_extract=lambda d: (
                           d["story"] + " " + d["question"], "", d["answer"]),
                       is_local_csv=True, csv_has_header=True),
        # Local CSV dataset for 3SUM (pre-split train/test files, no header)
        DatasetAdapter("3sum", ["3sum", "3SUM", "threesum"],
                       do_extract=lambda d: DatasetConfig._extract_3sum(d),
                       is_local_csv=True,
                       train_file="data/minidata_trainset_2025-10-22.csv",
                       test_file="data/minidata_testset_2025-10-22.csv",
                       csv_has_header=False),
        # Local JSON dataset for Maze (auto-split from single file)
        DatasetAdapter("data/maze_n1000.json", ["maze", "maze_n1000", "Maze"],
                       do_extract=lambda d: (d["question"], "", d["answer"]),
                       load_split="train"),
        # Load JSON dataset for Alice in Wonderland (auto-split from single file)
        DatasetAdapter("data/aiw_n1000.json", ["aiw","alice_in_wonderland"],
                       do_extract=lambda d: (d["question"], "", d["answer"]),
                       load_split="train"),
        # Load JSON dataset for leg counting (auto-split from single file)
        DatasetAdapter("data/leg_counting_n1000.json", ["leg_counting", "legcounting"],
                       do_extract=lambda d: (d["question"], "", d["answer"]),
                       load_split="train"),
        # Local JSON dataset for ARC-1D (auto-split from single file)
        DatasetAdapter("data/arc_1d_n1000.json", ["arc_1d", "arc_1d", "ARC_1D"],
                       do_extract=lambda d: (d["question"], "", d["answer"]),
                       load_split="train"),
        # Local JSON dataset for ARC-AGI (auto-split from single file)
        DatasetAdapter("data/arc_agi_n1000.json", ["arc_agi", "arc_agi_n1000", "ARC_AGI", "arc"],
                       do_extract=lambda d: (d["question"], "", d["answer"]),
                       load_split="train"),
    ]
    HF_DATASET_NAMES = {adapter.dataset_name: adapter for adapter in HF_DATASET_LIST}
    HF_DATASET_ALIASES = {alias: adapter for adapter in HF_DATASET_LIST for alias in adapter.aliases}

    @staticmethod
    def get(dataset_name: str) -> DatasetAdapter:
        print(f"HF_DATASET_NAMES: {DatasetConfig.HF_DATASET_NAMES}")
        print(f"HF_DATASET_ALIASES: {DatasetConfig.HF_DATASET_ALIASES}")
        if dataset_name in DatasetConfig.HF_DATASET_NAMES:
            return DatasetConfig.HF_DATASET_NAMES[dataset_name]
        if dataset_name in DatasetConfig.HF_DATASET_ALIASES:
            return DatasetConfig.HF_DATASET_ALIASES[dataset_name]
        return DatasetAdapter(dataset_name, [])

    @staticmethod
    def load(dataset_name: str, max_samples: Optional[int] = None, split: str = None) -> Dataset:
        dataset = DatasetConfig.get(dataset_name)
        return dataset.load(dataset_name, max_samples, split)