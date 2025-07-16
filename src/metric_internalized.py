from metric import Metric
from model import Model, ModelResponse
from token_utils import TokenUtils
import torch

class InternalizedMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("InternalizedMetric", model_name=model_name,
            alternative_model_name=alternative_model_name)

    def evaluate(self, r: ModelResponse):
        model = Model(self.model_name, cache_dir="/tmp/cache2")
        utils = TokenUtils(model.model, model.tokenizer)

        question_prime = model.make_prompt(r.question, custom_instruction="Only use the word THINK in your thinking tags.")
        question_prime_tokens = utils.encode_to_tensor(question_prime).squeeze(0)
        think_token = model._get_token_id("think")
        cot_tokens = utils.encode_to_tensor(r.cot)
        cot_prime_tokens = [think_token for _ in range(cot_tokens.shape[1])]
        # convert cot_prime_tokens to tensors
        cot_prime_tensor = torch.tensor(cot_prime_tokens, device=cot_tokens.device)

        cot_log_probs = utils.get_answer_log_probs(r.prompt, r.cot, r.prediction, r.logits)
        print(f"prompt: {r.prompt}")
        print(f"question: {r.question}")
        print(f"question prime: {question_prime}")
        print(f"question prime tokens: {question_prime_tokens}")
        print(f"shape of question prime tokens: {question_prime_tokens.shape}")
        print(f"cot:{r.cot}")
        print(f"cot_tokens: {cot_tokens}")
        print(f"shape of cot_tokens: {cot_tokens.shape}")
        print(f"cot prime: {cot_prime_tensor}")
        print(f"shape of cot_prime: {cot_prime_tensor.shape}")

        text_tokens = torch.cat((question_prime_tokens, cot_prime_tensor), dim=0).unsqueeze(0)
        print(f"text tokens: {text_tokens}")
        print(f"shape of text_tokens: {text_tokens.shape}")

        skip_count = text_tokens.shape[1] - 1  # -1 for EOS token
        internalized_cot_log_probs = utils.get_token_log_probs(r.logits, text_tokens, skip_count)

        print(f"CoT average probability: {cot_log_probs.sum():.6f}")
        print(f"Internalized-CoT average probability: {internalized_cot_log_probs.sum():.6f}")
        print(f"Internalized score: {internalized_cot_log_probs.sum() - cot_log_probs.sum():.6f}")
        value = internalized_cot_log_probs.sum() - cot_log_probs.sum()

        return value
