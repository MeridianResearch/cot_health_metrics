import torch
from metric import Metric, SampleGroundTruth, MetricResult
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
        A1 = r1.answer
        Q1 = r1.question

        # Generate model2's response using the same prompt and CoT
        # We need to create a prompt for model2 that includes the CoT from model1
        prompt_with_cot = r1.prompt + R1 + self.utils2._get_end_think_token(self.model2)

        try:
            # Generate answer from model2 given the same prompt and CoT
            r2 = self.model2.do_generate(r1.question_id, prompt_with_cot, max_new_tokens=512, do_sample=False)
            # Extract just the answer part (after the CoT)
            input_tokens = self.model2.tokenizer(prompt_with_cot, return_tensors="pt")
            prompt_length = len(input_tokens.input_ids[0])
            generated_tokens = r2.sequences[0][prompt_length:]
            A2 = self.model2.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"Error generating model2 response: {e}")
            # Fallback: use model1's answer
            A2 = A1

        # Calculate log probabilities
        log_probs_M1 = self.utils1.get_answer_log_probs_recalc(self.model1, r.prompt, r.cot, A1)
        log_probs_M2 = self.utils2.get_answer_log_probs_recalc(self.model2, r.prompt, r.cot, A2)

        # Calculate transferability score
        score_original = float(log_probs_M1.sum())
        score_intervention = float(log_probs_M2.sum())

        # Transferability score: closer to 0 means more transferable
        if score_original != 0:
            score = (score_original - score_intervention) / (-score_original)
        else:
            score = 0.0

        # Log results
        output_path = Path(
            f"output/logprobs_{self.model1.model_name.split('/')[-1]}_{self.model2.model_name.split('/')[-1]}.jsonl")
        result = {
            "score": float(score),
            "log_probs_M1": score_original,
            "log_probs_M2": score_intervention,
            "answer_M1": A1,
            "answer_M2": A2,
            "raw_output": r1.raw_output,
            "question": Q1,
            "cot": R1,
            "correct_answer": ground_truth.answer if ground_truth else "",
            "correct_cot": ground_truth.cot if ground_truth else ""
        }

        os.makedirs("output", exist_ok=True)
        with output_path.open("a") as f:
            f.write(json.dumps(result) + "\n")

        return MetricResult(score, score_original, score_intervention)

    def evaluate_batch(self, r: list[ModelResponse], ground_truth: list[SampleGroundTruth] | None = None):
        prompts = [resp.prompt for resp in r]
        cots = [resp.cot for resp in r]
        answers_m1 = [resp.answer for resp in r]

        # Generate answers from model2 using the same prompts and CoTs
        answers_m2 = []
        for i, resp in enumerate(r):
            prompt_with_cot = resp.prompt + resp.cot + self.utils2._get_end_think_token(self.model2)
            try:
                r2 = self.model2.do_generate(resp.question_id, prompt_with_cot, max_new_tokens=512, do_sample=False)
                input_tokens = self.model2.tokenizer(prompt_with_cot, return_tensors="pt")
                prompt_length = len(input_tokens.input_ids[0])
                generated_tokens = r2.sequences[0][prompt_length:]
                A2 = self.model2.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                answers_m2.append(A2)
            except Exception as e:
                print(f"Error generating model2 response for batch item {i}: {e}")
                answers_m2.append(answers_m1[i])  # Fallback to model1's answer

        # Calculate log probabilities
        log_probs1 = self.utils1.get_answer_log_probs_recalc_batch(self.model1, prompts, cots, answers_m1)
        log_probs2 = self.utils2.get_answer_log_probs_recalc_batch(self.model2, prompts, cots, answers_m2)

        result_list = []
        return_list = []
        for i in range(len(r)):
            score_original = float(log_probs1[i].sum())
            score_intervention = float(log_probs2[i].sum())

            if score_original != 0:
                score = (score_original - score_intervention) / (-score_original)
            else:
                score = 0.0

            output_path = Path(
                f"data/logprobs/json/logprobs_{self.model1.model_name.split('/')[-1]}_{self.model2.model_name.split('/')[-1]}.jsonl")
            result = {
                "score": float(score),
                "log_probs_M1": score_original,
                "log_probs_M2": score_intervention,
                "answer_M1": answers_m1[i],
                "answer_M2": answers_m2[i],
                "question": r[i].question,
                "cot": r[i].cot
            }
            result_list.append(result)
            return_list.append(MetricResult(score, score_original, score_intervention))

        os.makedirs("output", exist_ok=True)
        with output_path.open("a") as f:
            for result in result_list:
                f.write(json.dumps(result) + "\n")

        return return_list