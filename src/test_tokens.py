import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import torch
from model_prompts import ModelPromptBuilder

# Import project modules
from token_utils import TokenUtils
from model import CoTModel

TEST_CACHE_DIR = "/tmp/cache-test"

class TestLogProbs:
    #def test_e(self):
    #    model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
    #    utils = model.get_utils()
    #    prompt = "Please write \"Hello World\"."
    #    answer = "Hello World"
    #    log_probs = utils.get_answer_log_probs_recalc(model, prompt, "", answer)
    #    print(f"log_probs: {log_probs}")
    #    assert False

    #def test_system_prompt_builder_default(self):
    #    model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
    #    utils = model.get_utils()
    #    prompt = "Please write \"Hello World\"."
    #    answer = "Hello World"
    #    log_probs = utils.get_answer_log_probs_recalc_no_cot(model, prompt, answer)
    #    print(f"log_probs: {log_probs}")
    #    assert False

    def test_hello_world_token_log_probs_calculation_no_cot(self):
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        utils = model.get_utils()

        prompt = "<|im_start|>assistant\n"
        answer = "Hello World<|im_end|>"
        full_answer = prompt + answer
        tokens = model.tokenizer.encode(full_answer)
        print(utils.encode_to_tensor(full_answer))
        print(f"tokens: {tokens}")
        assert tokens == [151644, 77091, 198, 9707, 4337, 151645]  # length 6

        log_probs = model.get_log_probs(torch.tensor([tokens]).to(model.model.device))

        # looking at off by ones
        correct_hello_log_prob = log_probs[0, 2, 9707]
        assert correct_hello_log_prob > log_probs[0, 1, 9707]
        assert correct_hello_log_prob > log_probs[0, 3, 9707]

        # manually calculate the log prob of the answer
        print(f"log_probs[0, 2, 9707]: {log_probs[0, 2, 9707]}")
        print(f"log_probs[0, 3, 4337]: {log_probs[0, 3, 4337]}")
        print(f"log_probs[0, 4, 151645]: {log_probs[0, 4, 151645]}")
        log_prob1 = log_probs[0, 2, 9707]
        log_prob2 = log_probs[0, 3, 4337]
        log_prob3 = log_probs[0, 4, 151645]
        manual_log_prob = torch.tensor([log_prob1, log_prob2, log_prob3]).to(model.model.device)
        print(f"manual_log_prob: {manual_log_prob}")

        # automatically calculate the log prob of the answer
        utils_log_probs = utils.get_answer_log_probs_recalc_no_cot(model, prompt, answer)
        print(f"utils_log_probs: {utils_log_probs}")
        assert utils_log_probs.shape == manual_log_prob.shape
        assert torch.equal(utils_log_probs, manual_log_prob)
        assert len(utils_log_probs) == 3
        assert utils_log_probs.sum() == manual_log_prob.sum()

    def test_hello_world_with_prompt_cot_token_log_probs_calculation_no_cot(self):
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        utils = model.get_utils()

        prompt = "<|im_start|>user\nPlease write \"Hello World\".<|im_end|>\n<|im_start|>assistant\n"
        answer = "Hello World<|im_end|>"
        full_answer = prompt + answer
        tokens = model.tokenizer.encode(full_answer)
        print(utils.encode_to_tensor(full_answer))
        print(f"tokens: {tokens}")
        assert tokens == [151644, 872, 198, 5501, 3270, 330, 9707, 4337, 3263, 151645, 198, 151644, 77091, 198, 9707, 4337, 151645]
        assert len(tokens) == 17

        log_probs = model.get_log_probs(torch.tensor([tokens]).to(model.model.device))

        correct_end_token_log_prob = log_probs[0, 15, 151645]
        assert correct_end_token_log_prob > log_probs[0, 14, 151645]
        assert log_probs[0, 16, :].mean() < log_probs[0, 15, :].mean()

        # manually calculate the log prob of the answer
        print(f"log_probs[0, 13, 9707]: {log_probs[0, 13, 9707]}")
        print(f"log_probs[0, 14, 4337]: {log_probs[0, 14, 4337]}")
        print(f"log_probs[0, 15, 151645]: {log_probs[0, 15, 151645]}")
        log_prob1 = log_probs[0, 13, 9707]
        log_prob2 = log_probs[0, 14, 4337]
        log_prob3 = log_probs[0, 15, 151645]
        manual_log_prob = torch.tensor([log_prob1, log_prob2, log_prob3]).to(model.model.device)
        print(f"manual_log_prob: {manual_log_prob}")

        # automatically calculate the log prob of the answer
        utils_log_probs = utils.get_answer_log_probs_recalc_no_cot(model, prompt, answer)
        print(f"utils_log_probs: {utils_log_probs}")
        assert utils_log_probs.shape == manual_log_prob.shape
        assert torch.equal(utils_log_probs, manual_log_prob)
        assert len(utils_log_probs) == 3
        assert utils_log_probs.sum() == manual_log_prob.sum()

    def test_hello_world_with_prompt_cot_token_log_probs_calculation(self):
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        utils = model.get_utils()

        prompt = "<|im_start|>user\nPlease write \"Hello World\".<|im_end|>\n<|im_start|>assistant\n" \
            + "<think>\nI should output just the text \"Hello World\".\n</think>\n\n"
        answer = "Hello World<|im_end|>"
        full_answer = prompt + answer
        tokens = model.tokenizer.encode(full_answer)
        print(utils.encode_to_tensor(full_answer))
        print(f"tokens: {tokens}")

        # 198 is one newline, 271 is two newlines
        assert tokens == [151644, 872, 198, 5501, 3270, 330, 9707, 4337, 3263, 151645, 198, 151644, 77091, 198, \
            151667, 198, 40, 1265, 2550, 1101, 279, 1467, 330, 9707, 4337, 22956, 151668, \
            271, 9707, 4337, 151645]
        assert len(tokens) == 31

        log_probs = model.get_log_probs(torch.tensor([tokens]).to(model.model.device))

        # manually calculate the log prob of the answer
        print(f"log_probs[0, 27, 9707]: {log_probs[0, 27, 9707]}")
        print(f"log_probs[0, 28, 4337]: {log_probs[0, 28, 4337]}")
        print(f"log_probs[0, 29, 151645]: {log_probs[0, 29, 151645]}")
        log_prob1 = log_probs[0, 27, 9707]
        log_prob2 = log_probs[0, 28, 4337]
        log_prob3 = log_probs[0, 29, 151645]
        manual_log_prob = torch.tensor([log_prob1, log_prob2, log_prob3]).to(model.model.device)
        print(f"manual_log_prob: {manual_log_prob}")

        # automatically calculate the log prob of the answer
        utils_log_probs = utils.get_answer_log_probs_recalc_no_cot(model, prompt, answer)
        print(f"utils_log_probs: {utils_log_probs}")
        assert utils_log_probs.shape == manual_log_prob.shape
        assert torch.equal(utils_log_probs, manual_log_prob)
        assert len(utils_log_probs) == 3
        assert utils_log_probs.sum() == manual_log_prob.sum()

    #def test_e2(self):
    #    model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
    #    utils = model.get_utils()

    #    question = "Please write \"Hello World\"."
    #    prompt_builder = ModelPromptBuilder("Qwen/Qwen3-0.6B", invokes_cot=False)
    #    prompt_builder.add_to_history("user", "Please write \"Hello World\".")
    #    prompt = prompt_builder.make_prompt(model.tokenizer)
    #    print(f"prompt: {utils.escape_string(prompt)}")

    #    if False:
    #        answer_real = model.do_generate(1, prompt, max_new_tokens=256, do_sample=False)
    #        answer_real_decoded = model.tokenizer.decode(answer_real.sequences[0], skip_special_tokens=False)
    #        print(f"answer_real_decoded: {utils.escape_string(answer_real_decoded)}")

    #    answer = "Hello World<|im_end|>"
    #    answer_encoded = utils.encode_to_tensor(answer)
    #    assert answer_encoded.shape[1] == 3  # 3 tokens

    #    log_probs = utils.get_answer_log_probs_recalc_no_cot(model, prompt, "<|im_end|>")
    #    print(f"log_probs: {log_probs}")
    #    log_probs = utils.get_answer_log_probs_recalc_no_cot(model, prompt, answer)
    #    print(f"log_probs: {log_probs}")
    #    assert False
