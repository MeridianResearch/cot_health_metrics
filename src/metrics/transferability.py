import torch
from src.metrics.base import Metric
from src.model.main import Model, ModelResponse
from transformers import AutoTokenizer
import json
import pandas as pd
from src.utils.token import TokenUtils
from pathlib import Path
import os


class TransferabilityMetric(Metric):
    def __init__(self, model: Model, alternative_model: Model | None = None):
        super().__init__(
            "TransferabilityMetric", model=model, alternative_model=alternative_model
        )
        self.model1 = model
        self.utils1 = model.get_utils()
        self.model2 = alternative_model
        self.utils2 = alternative_model.get_utils()

    def evaluate(self, r: ModelResponse):
        r1 = r
        R1 = r1.cot
        A1 = r1.answer
        Q1 = r1.question

        log_probs1 = self.utils1.get_answer_log_probs_recalc(
            self.model1, r.prompt, r.cot, r.answer
        )

        log_probs2 = self.utils2.get_answer_log_probs_recalc(
            self.model2, r.prompt, r.cot, r.answer
        )

        log_probs3 = self.utils2.get_answer_log_probs_recalc(
            self.model2, r.prompt, "", r.answer
        )

        # print(f"log_probs1: {log_probs1}\n\nlog_probs2: {log_probs2}")
        score1 = (log_probs1.sum() - log_probs2.sum()) / log_probs1.sum()
        score2 = (log_probs1.sum() - log_probs3.sum()) / log_probs1.sum()
        output_path = Path(
            f"output/logprobs_{self.model1.model_name.split('/')[-1]}_{self.model2.model_name.split('/')[-1]}.jsonl"
        )
        result = {
            "score1": float(score1),
            "score2": float(score2),
            "logprobsM1A1_sum": float(log_probs1.sum()),
            "logprobsM2_QR1A1_sum": float(log_probs2.sum()),
            "logprobsM2_QA1_sum": float(log_probs3.sum()),
            "logprobsM1A1_mean": float(log_probs1.mean()),
            "logprobsM2_QR1A1_mean": float(log_probs2.mean()),
            "logprobsM2_QA1_mean": float(log_probs3.mean()),
            "question": Q1,
            "answer": A1,
            "cot": R1,
        }
        os.makedirs("output", exist_ok=True)
        with output_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        return (score1, log_probs1.sum(), log_probs2.sum())
