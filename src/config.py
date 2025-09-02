from random import sample
from datasets import load_dataset, Dataset
from typing import Callable

CACHE_DIR_DEFAULT       = "hf_cache"
LOG_DIRECTORY_DEFAULT   = "log"
LOG_EVERY_DEFAULT       = 1

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
        #"deepcogito/cogito-v1-preview-llama-3B": MODEL_CONFIG_THINK_TOKENS,  # unverified
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
        if model_name in ModelConfig.SUPPORTED_MODELS:
            return True
        # allow any Qwen-* because they use <think> tags (and are my finetune!)
        if "Qwen" in model_name:
            return True
        return False

class DatasetAdapter:
    def __init__(self, dataset_name: str, aliases: list[str],
        load_section: str = "main", load_split: str = "train",
        do_extract: Callable | None = lambda d: (d["question"], "", d["answer"])):

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

    def load(self, dataset_name: str, max_samples: int | None = None) -> Dataset:
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
            do_extract=lambda d: (d["question"], d["answer"].split("####")[0], d["answer"].split("####")[1])),
        DatasetAdapter("cais/mmlu", ["MMLU", "mmlu"], load_section="astronomy", load_split="dev",
            do_extract=lambda d: (d["question"] + "\nChoices: "
                + [f"{chr(ord('A')+i)}: {d['choices'][i]}\n" for i in range(len(d["choices"]))],
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
    def load(dataset_name: str, max_samples: int | None = None) -> Dataset:
        dataset = DatasetConfig.get(dataset_name)
        return dataset.load(dataset_name, max_samples)
