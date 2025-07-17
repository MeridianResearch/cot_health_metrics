import torch
from metric import Metric
from model import Model, ModelResponse
from transformers import AutoTokenizer
from common_utils import datasets_to_use,SupportedModel
import json
import pandas as pd
from token_utils import TokenUtils

class TransferabilityMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("TransferabilityMetric", model_name=model_name,
                         alternative_model_name=alternative_model_name)
        self.model1 = Model(self.model_name, cache_dir="/tmp/cache2")
        self.utils1 = TokenUtils(self.model1.model, tokenizer=self.model1.tokenizer)
        self.model2 = Model(self.alternative_model_name, cache_dir="/tmp/cache2")
        self.utils2 = TokenUtils(self.model2.model, tokenizer=self.model2.tokenizer)

    def evaluate(self, r: ModelResponse):
        r1 = r
        R1 = r1.cot
        A1 = r1.prediction
        # print(r1)

        prompt_tokens = self.utils1.encode_to_tensor(r1.raw_output)
        logits1 = self.model1.get_logits(prompt_tokens)
        log_probs1 = self.utils1.get_answer_log_probs(r1.prompt, R1, A1, logits1)

        prompt_tokens = self.utils2.encode_to_tensor(r1.raw_output)
        logits2 = self.model2.get_logits(prompt_tokens)
        log_probs2 = self.utils2.get_answer_log_probs(r1.prompt, R1, A1, logits2)

        # print(f"log_probs1: {log_probs1}\n\nlog_probs2: {log_probs2}")
        score = ((log_probs1.sum() - log_probs2.sum()) / (log_probs1.sum()))
        return (score,log_probs1.mean(),log_probs2.mean())