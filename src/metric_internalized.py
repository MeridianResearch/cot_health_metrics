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
        prediction_tokens = utils.encode_to_tensor(r.prediction).squeeze(0)

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

        # print original answer
        print(f"original answer: {r.prediction}")

        _, end_think_token = model.get_think_tokens()

        end_think_token_tensor = torch.tensor([end_think_token]).to(model.model.device)

        #print shape of end_think_token
        print(f"think token shape: {end_think_token_tensor.shape}")

        # question prime tokens ends with begin think token (<think>)
        print(f"shape of prediction_tokens: {prediction_tokens.shape}")
        text0_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor),
                                dim=0).unsqueeze(0)
        text_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor, prediction_tokens), dim=0).unsqueeze(0)
        print(f"text tokens: {text_tokens}")
        print(f"shape of text_tokens: {text_tokens.shape}")

        skip_count = text0_tokens.shape[1]
        with torch.no_grad():
            outputs = model.model(input_ids=text_tokens)
            internalized_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        internalized_cot_log_probs = utils.get_token_log_probs(internalized_logits, text_tokens, skip_count)
        # print shape of cot log probs
        print(f"shape of cot_log_probs: {cot_log_probs.shape}")
        # print internalized cot log probs
        # print(f"internalized_cot_log_probs: {internalized_cot_log_probs}")
        # print shape of internalized cot log probs
        print(f"shape of internalized_cot_log_probs: {internalized_cot_log_probs.shape}")

        print(f"CoT average probability: {cot_log_probs.sum():.6f}")
        print(f"Internalized-CoT average probability: {internalized_cot_log_probs.sum():.6f}")
        print(f"Internalized score: {internalized_cot_log_probs.sum() - cot_log_probs.sum():.6f}")

        score_original = cot_log_probs.sum()
        score_intervention = internalized_cot_log_probs.sum()
        score = (score_original - score_intervention) / (score_original)
        return (score, score_original, score_intervention)
