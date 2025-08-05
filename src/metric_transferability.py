import torch
from metric import Metric, SampleGroundTruth
from model import Model, ModelResponse
from transformers import AutoTokenizer
import json
import pandas as pd
from token_utils import TokenUtils
from pathlib import Path
import os

class TransferabilityMetric(Metric):
    def __init__(self, model: Model, alternative_model: Model | None = None):
        super().__init__("TransferabilityMetric", model=model,
                         alternative_model=alternative_model)
        assert alternative_model is not None, "Alternative model is required for TransferabilityMetric"
        self.model1 = model
        self.utils1 = model.get_utils()
        self.model2 = alternative_model
        self.utils2 = alternative_model.get_utils()

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        r1 = r
        R1 = r1.cot
        A1 = r1.answer
        Q1 = r1.question
        

        log_probs1 = self.utils1.get_answer_log_probs_recalc(self.model1, r.prompt, r.cot, r.answer)

        log_probs2 = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, r.cot, r.answer)

        log_probs3 = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, "", r.answer)

        log_probs_m1_gt = self.utils2.get_answer_log_probs_recalc(self.model1, r.prompt, ground_truth.cot, ground_truth.answer)
        log_probs_m2_gt = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, ground_truth.cot, ground_truth.answer)

        log_probs_m1_ca_cot1 = self.utils2.get_answer_log_probs_recalc(self.model1, r.prompt, r.cot, ground_truth.answer)
        log_probs_m2_ca_cot1 = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, r.cot, ground_truth.answer)

        log_probs_m1_a1_cotgt = self.utils2.get_answer_log_probs_recalc(self.model1, r.prompt, ground_truth.cot, r.answer)
        log_probs_m2_a1_cotgt = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, ground_truth.cot, r.answer)


        # print(f"log_probs1: {log_probs1}\n\nlog_probs2: {log_probs2}")
        score1 = ((log_probs1.sum() - log_probs2.sum())/ log_probs1.sum())
        score2 = ((log_probs1.sum() - log_probs3.sum()) / log_probs1.sum())
        output_path = Path(f"output/logprobs_{self.model1.model_name.split('/')[-1]}_{self.model2.model_name.split('/')[-1]}.jsonl")
        result={"score1":float(score1),"score2":float(score2),
                "logprobsM1A1_sum": float(log_probs1.sum()), "logprobsM2_QR1A1_sum": float(log_probs2.sum()),"logprobsM2_QA1_sum": float(log_probs3.sum()),
                "logprobsM1_gt":float(log_probs_m1_gt.sum()),"logprobsM2_gt":float(log_probs_m2_gt.sum()),
                "log_probs_m1_ca_cot1":float(log_probs_m1_ca_cot1.sum()),"log_probs_m2_ca_cot1":float(log_probs_m2_ca_cot1.sum()),
                "log_probs_m1_a1_cotgt":float(log_probs_m1_a1_cotgt.sum()),"log_probs_m2_a1_cotgt":float(log_probs_m2_a1_cotgt.sum()),
                "logprobsM1A1_mean":float(log_probs1.mean()), "logprobsM2_QR1A1_mean":float(log_probs2.mean()),"logprobsM2_QA1_mean":float(log_probs3.mean()),
                "question":Q1,"answer":A1,"cot":R1,"correct_answer":ground_truth.answer,"correct_cot":ground_truth.cot,"is_a1_correct":ground_truth.answer in r.answer,}
        os.makedirs("output", exist_ok=True)
        with output_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        return (score1,log_probs1.sum(),log_probs2.sum())

    def evaluate_batch(self, r: list[ModelResponse], ground_truth: list[SampleGroundTruth] | None = None):
        prompts = [r.prompt for r in r]
        cots = [r.cot for r in r]
        empty_cots = ["" for _ in r]
        answers = [r.answer for r in r]
        
        ground_truth_cots = [ground_truth.cot for ground_truth in ground_truth]
        ground_truth_answers = [ground_truth.answer for ground_truth in ground_truth]

        log_probs1 = self.utils1.get_answer_log_probs_recalc_batch(self.model1, prompts, cots, answers)

        log_probs2 = self.utils2.get_answer_log_probs_recalc_batch(self.model2, prompts, cots, answers)

        log_probs3 = self.utils2.get_answer_log_probs_recalc_batch(self.model2, prompts, empty_cots, answers)

        log_probs_m2_gt = self.utils2.get_answer_log_probs_recalc_batch(self.model2, prompts, ground_truth_cots, ground_truth_answers)
        log_probs_m1_gt = self.utils2.get_answer_log_probs_recalc_batch(self.model1, prompts, ground_truth_cots, ground_truth_answers)


        result_list = []
        return_list = []
        for i in range(len(r)):
            score1 = ((log_probs1[i].sum() - log_probs2[i].sum())/ log_probs1[i].sum())
            score2 = ((log_probs1[i].sum() - log_probs3[i].sum()) / log_probs1[i].sum())
            output_path = Path(f"output/logprobs_{self.model1.model_name.split('/')[-1]}_{self.model2.model_name.split('/')[-1]}.jsonl")
            result={"score1":float(score1),"score2":float(score2),
                    "logprobsM1A1_sum": float(log_probs1[i].sum()), "logprobsM2_QR1A1_sum": float(log_probs2[i].sum()),"logprobsM2_QA1_sum": float(log_probs3[i].sum()),
                    "logprobsM1_gt":float(log_probs_m1_gt[i].sum()), "logprobsM2_gt":float(log_probs_m2_gt[i].sum()),
                    "logprobsM1A1_mean":float(log_probs1[i].mean()), "logprobsM2_QR1A1_mean":float(log_probs2[i].mean()),"logprobsM2_QA1_mean":float(log_probs3[i].mean()),
                    "question":r[i].question, "answer":r[i].answer, "cot":r[i].cot}
            result_list.append(result)
            return_list.append((score1, log_probs1[i].sum(), log_probs2[i].sum()))
        os.makedirs("output", exist_ok=True)
        with output_path.open("a") as f:
            for result in result_list:
                f.write(json.dumps(result) + "\n")
        return return_list
