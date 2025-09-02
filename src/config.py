from random import sample
from datasets import load_dataset, Dataset
from typing import Callable, Union, Optional

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
        "end_think": "</think>",
        "generate_kwargs": {
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0.0,
            "top_p": 0.95,
        },
    }

    MODEL_CONFIG_GPT_OSS_20B = {
        "begin_think": "<|end|><|start|>assistant<|channel|>final<|message|>analysis<|message|>",
        "end_think": "<|end|><|start|>assistant<|channel|>final<|message|>",
        "do_not_think": "<|end|><|start|>assistant<|channel|>final<|message|>",
        "generate_kwargs": {
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0.0,
            "top_p": 0.95,
        },
    }
    MODEL_CONFIG_GEMMA = {
        "fuzzy_end_think_list": ["Answer:"],
        "generate_kwargs": {
            "repetition_penalty": 1.2,
            "temperature": 0.7,
            "top_k": 20,
            "min_p": 0.0,
            "top_p": 0.95,
        },
    }
    MODEL_CONFIG_LLAMA = {
        "fuzzy_end_think_list": ["Answer:"],
        "generate_kwargs": {
            "temperature": 0.6,
            "top_k": 20,
            "min_p": 0.0,
            "top_p": 0.95,
        },
    }
    DEFAULT_MODEL_CONFIG = MODEL_CONFIG_GEMMA

    SUPPORTED_MODELS = {
        "Qwen/Qwen3-0.6B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-1.7B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-4B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-8B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-14B": MODEL_CONFIG_THINK_TOKENS,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": MODEL_CONFIG_THINK_TOKENS,
        # "deepcogito/cogito-v1-preview-llama-3B": MODEL_CONFIG_THINK_TOKENS,  # unverified
        "Wladastic/Mini-Think-Base-1B": MODEL_CONFIG_GEMMA,
        "google/gemma-2-2b-it": MODEL_CONFIG_GEMMA,
        "openai/gpt-oss-20b": MODEL_CONFIG_GPT_OSS_20B,
        # "microsoft/phi-2": MODEL_CONFIG_FUZZY_ANSWER,  # not very consistent
        "meta-llama/Meta-Llama-3-8B-Instruct": MODEL_CONFIG_LLAMA,
        "meta-llama/Llama-2-7b-chat-hf": MODEL_CONFIG_LLAMA,
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
    def __init__(self, dataset_name: str, aliases: list[str],
                 load_section: str = "main", load_split: str = "train",
                 do_extract: Union[Callable, None] = lambda d: (d["question"], "", d["answer"])):

        self.dataset_name = dataset_name
        self.aliases = aliases
        self.load_section = load_section
        self.load_split = load_split
        self.do_extract = do_extract

    def name_matches(self, dataset_name: str) -> bool:
        return dataset_name == self.dataset_name or dataset_name in self.aliases

    def get(self, dataset_name: str) -> str:
        if dataset_name == self.dataset_name:
            return self.dataset_name
        if dataset_name in self.aliases:
            return self.dataset_name
        raise ValueError(f"Dataset {self.dataset_name} passed invalid alias {dataset_name}")

    def extract_pieces(self, sample_data) -> tuple[str, str, str]:
        return self.do_extract(sample_data)

    def load(self, dataset_name: str, max_samples: Optional[int] = None) -> Dataset:
        split = self.load_split
        if max_samples is not None:
            split = split + f"[:{max_samples}]"
        print(f"Loading dataset {dataset_name} with split {split}")
        print(f"Dataset name: {self.get(dataset_name)}")
        print(f"Stored Dataset name: {self.dataset_name}")
        return load_dataset(self.get(dataset_name), self.load_section, split=split)


class DatasetConfig:
    HF_DATASET_LIST = [
        DatasetAdapter("vicgalle/alpaca-gpt4", ["alpaca", "alpaca-gpt4"],
                       do_extract=lambda d: (d["question"], "", d["answer"])),
        DatasetAdapter("gsm8k", ["GSM8K"],
                       do_extract=lambda d: (
                       d["question"], d["answer"].split("####")[0], d["answer"].split("####")[1])),
        DatasetAdapter("cais/mmlu", ["MMLU", "mmlu"], load_section="astronomy", load_split="dev",
                       do_extract=lambda d: (d["question"] + "\nChoices: "
                                             + [f"{chr(ord('A') + i)}: {d['choices'][i]}\n" for i in
                                                range(len(d["choices"]))],
                                             "", d["answer"])),
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
    def load(dataset_name: str, max_samples: Optional[int] = None) -> Dataset:
        dataset = DatasetConfig.get(dataset_name)
        return dataset.load(dataset_name, max_samples)