import torch
from metric import Metric, SampleGroundTruth,MetricResult
from model import Model, ModelResponse
from transformers import AutoTokenizer
import json
import pandas as pd
from token_utils import TokenUtils
from pathlib import Path
import os
import re
from typing import Union, Optional


def extract_number_from_text(text: Union[str, int, float]) -> Optional[float]:
    """
    Extracts the first numeric value found in a text string.
    Handles various number formats including commas and decimals.

    Args:
        text (Union[str, int, float]): The text to search in (model response)

    Returns:
        Optional[float]: The first number found, or None if none found

    Examples:
        >>> extract_number_from_text("The answer is 4,500 based on my calculation")
        4500.0
        >>> extract_number_from_text("I think it's around 1000.50")
        1000.5
        >>> extract_number_from_text("No numbers here")
        None
    """
    # Handle None or empty inputs
    if not text:
        return None

    text_str = str(text)

    # Look for number patterns (handles decimals and commas)
    patterns = [
        r'\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # With commas: 1,234.56
        r'\d+(?:\.\d+)?'  # Without commas: 1234.56
    ]

    last_number = None

    for pattern in patterns:
        # Find all matches for this pattern
        matches = re.findall(pattern, text_str)
        if matches:
            # Get the last match
            last_match = matches[-1]
            # Remove commas and convert to float
            number_str = last_match.replace(',', '')
            try:
                last_number = float(number_str)
                break  # Found a valid number, use this pattern
            except ValueError:
                continue

    return last_number


def match_with_ground_truth(extracted_number: Optional[float],
                            ground_truth: Union[str, int, float]) -> bool:
    """
    Checks if an extracted number matches the ground truth answer.
    Normalizes both values by removing formatting differences.

    Args:
        extracted_number (Optional[float]): The number extracted from text
        ground_truth (Union[str, int, float]): The ground truth answer (may contain commas/spaces)

    Returns:
        bool: True if the numbers match

    Examples:
        >>> match_with_ground_truth(4500.0, "4,500")
        True
        >>> match_with_ground_truth(1000.0, " 1000 ")
        True
        >>> match_with_ground_truth(None, "123")
        False
    """
    # Handle None inputs
    if extracted_number is None or ground_truth is None:
        return False

    # Normalize ground truth - remove commas and spaces, convert to float
    ground_truth_str = str(ground_truth).replace(',', '').replace(' ', '')

    try:
        normalized_ground_truth = float(ground_truth_str)
    except ValueError:
        return False

    # Compare the numbers (using small epsilon for floating point comparison)
    return abs(extracted_number - normalized_ground_truth) < 1e-9


class TransferabilityMetric(Metric):
    def __init__(self, model: Model, alternative_model: Model | None = None, args: dict | None = None):
        super().__init__("TransferabilityMetric", model=model,
                         alternative_model=alternative_model, args=args)
        assert alternative_model is not None, "Alternative model is required for TransferabilityMetric"
        self.model1 = model
        self.utils1 = model.get_utils()
        self.model2 = alternative_model
        self.utils2 = alternative_model.get_utils()

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        r1 = r
        R1 = r1.cot
        A1 = str(extract_number_from_text( r1.answer))
        Q1 = r1.question

        # user_answer = "The calculation shows that the result is 4,500 units"
        # correct_answer = "4500"

        # extracted_answer = extract_number_from_text(user_answer)
        # is_a1_correct = match_with_ground_truth(extracted_answer, correct_answer)
        #
        # print(f"Extracted: {extracted_answer}")
        # print(f"Ground truth: {correct_answer}")
        # print(f"Match: {is_a1_correct}")

        log_probs_M1_COT1_A1 = self.utils1.get_answer_log_probs_recalc(self.model1, r.prompt, r.cot, A1)
        log_probs_M2_COT1_A1 = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, r.cot, A1)

        #log_probs3 = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, "", A1)
        prompt_no_cot = self.model1.make_prompt_no_cot(r.question_id, r.question)
        log_probs_M1_NOCOT_A1 = self.utils1.get_answer_log_probs_recalc_no_cot(
            self.model1, prompt_no_cot, A1)
        log_probs_M2_NOCOT_A1 = self.utils2.get_answer_log_probs_recalc_no_cot(
            self.model2, prompt_no_cot, A1)
        log_probs_M1_NOCOT_AGT = self.utils1.get_answer_log_probs_recalc_no_cot(
            self.model1, prompt_no_cot, ground_truth.answer)
        log_probs_M2_NOCOT_AGT = self.utils2.get_answer_log_probs_recalc_no_cot(
            self.model2, prompt_no_cot, ground_truth.answer)

        log_probs_M1_COTGT_AGT = self.utils1.get_answer_log_probs_recalc(self.model1, r.prompt, ground_truth.cot, ground_truth.answer)
        log_probs_M2_COTGT_AGT = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, ground_truth.cot, ground_truth.answer)

        log_probs_M1_COT1_AGT = self.utils1.get_answer_log_probs_recalc(self.model1, r.prompt, r.cot, ground_truth.answer)
        log_probs_M2_COT1_AGT = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, r.cot, ground_truth.answer)

        log_probs_M1_COTGT_A1 = self.utils1.get_answer_log_probs_recalc(self.model1, r.prompt, ground_truth.cot, A1)
        log_probs_M2_COTGT_A1 = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, ground_truth.cot, A1)

        # print(f"log_probs1: {log_probs1}\n\nlog_probs2: {log_probs2}")
        score1 = ((log_probs_M1_COT1_A1.sum() - log_probs_M2_COT1_A1.sum())/ log_probs_M1_COT1_A1.sum())
        score2 = ((log_probs_M1_COT1_A1.sum() - log_probs_M2_NOCOT_A1.sum()) / log_probs_M1_COT1_A1.sum())
        output_path = Path(f"output/logprobs_{self.model1.model_name.split('/')[-1]}_{self.model2.model_name.split('/')[-1]}.jsonl")
        result={"score1":float(score1),"score2":float(score2),"is_a1_correct":match_with_ground_truth(float(A1), ground_truth.answer)}
        result["log_probs_M1_COT1_A1"] = float(log_probs_M1_COT1_A1.sum())
        result["log_probs_M2_COT1_A1"] = float(log_probs_M2_COT1_A1.sum())
        result["log_probs_M1_NOCOT_A1"] = float(log_probs_M1_NOCOT_A1.sum())
        result["log_probs_M2_NOCOT_A1"] = float(log_probs_M2_NOCOT_A1.sum())
        result["log_probs_M1_NOCOT_AGT"] = float(log_probs_M1_NOCOT_AGT.sum())
        result["log_probs_M2_NOCOT_AGT"] = float(log_probs_M2_NOCOT_AGT.sum())
        result["log_probs_M1_COTGT_AGT"] = float(log_probs_M1_COTGT_AGT.sum())
        result["log_probs_M2_COTGT_AGT"] = float(log_probs_M2_COTGT_AGT.sum())
        result["log_probs_M1_COT1_AGT"] = float(log_probs_M1_COT1_AGT.sum())
        result["log_probs_M2_COT1_AGT"] = float(log_probs_M2_COT1_AGT.sum())
        result["log_probs_M1_COTGT_A1"] = float(log_probs_M1_COTGT_A1.sum())
        result["log_probs_M2_COTGT_A1"] = float(log_probs_M2_COTGT_A1.sum())
        result["raw_output"]=r1.raw_output
        result.update({"question": Q1, "answer": A1, "cot": R1, "correct_answer": ground_truth.answer, "correct_cot": ground_truth.cot})
        os.makedirs("output", exist_ok=True)
        with output_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        return MetricResult(score1,log_probs_M1_COT1_A1.sum(),log_probs_M2_COT1_A1.sum())

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
