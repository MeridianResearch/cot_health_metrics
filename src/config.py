from datasets import load_dataset, Dataset

CACHE_DIR_DEFAULT       = "hf_cache"
LOG_DIRECTORY_DEFAULT   = "log"
LOG_EVERY_DEFAULT       = 1

class ModelConfig:
    MODEL_CONFIG_THINK_TOKENS = {
        "begin_think": "<think>",
        "end_think": "</think>",
    }
    MODEL_CONFIG_FUZZY_ANSWER = {
        "fuzzy_separator": "Answer: ",
    }
    DEFAULT_MODEL_CONFIG = MODEL_CONFIG_FUZZY_ANSWER

    SUPPORTED_MODELS = {
        "Qwen/Qwen3-0.6B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-1.7B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-4B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-8B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-14B": MODEL_CONFIG_THINK_TOKENS,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": MODEL_CONFIG_THINK_TOKENS,
        #"deepcogito/cogito-v1-preview-llama-3B": MODEL_CONFIG_THINK_TOKENS,  # unverified
        "Wladastic/Mini-Think-Base-1B": MODEL_CONFIG_FUZZY_ANSWER,
        "google/gemma-2-2b": MODEL_CONFIG_FUZZY_ANSWER,
        #"microsoft/phi-2": MODEL_CONFIG_FUZZY_ANSWER,  # not very consistent
    }

    @staticmethod
    def get(model_name: str) -> dict:
        if model_name not in ModelConfig.SUPPORTED_MODELS:
            return ModelConfig.DEFAULT_MODEL_CONFIG
        return ModelConfig.SUPPORTED_MODELS[model_name]

    @staticmethod
    def is_supported(model_name: str) -> bool:
        return model_name in ModelConfig.SUPPORTED_MODELS

class DatasetConfig:
    HF_DATASET_NAMES = {
        "Alpaca": "vicgalle/alpaca-gpt4",
        "GSM8K": "gsm8k",
        "MMLU": "openai/openai_mmlu_benchmark",
    }

    @staticmethod
    def get(dataset_name: str) -> str:
        if dataset_name not in DatasetConfig.HF_DATASET_NAMES:
            return dataset_name
        return DatasetConfig.HF_DATASET_NAMES[dataset_name]
    
    @staticmethod
    def load(dataset_name: str, **kwargs) -> Dataset:
        return load_dataset(dataset_name, "main", **kwargs)