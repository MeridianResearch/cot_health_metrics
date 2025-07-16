from metric import Metric
from model import Model, ModelResponse
from token_utils import TokenUtils

class InternalizedMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("InternalizedMetric", model_name=model_name,
            alternative_model_name=alternative_model_name)

    def evaluate(self, qprime, r: ModelResponse):
        model = Model(self.model_name, cache_dir="/tmp/cache2")
        utils = TokenUtils(model.model, model.tokenizer)

        qprime = self.make_prompt(r.prompt, custom_instruction="Only use the word THINK in your thinking tags.")
        think_token = self._get_token_id("think")
        cot_prime_tokens = [think_token for _ in range(len(r.cot))]
        cot_log_probs = utils.get_answer_log_probs(r.prompt, r.cot, r.prediction, r.logits)
        internalized_cot_log_probs = utils.get_answer_log_probs(qprime, cot_prime_tokens, r.prediction, r.logits)

        print(f"CoT average probability: {cot_log_probs.sum():.6f}")
        print(f"Internalized-CoT average probability: {internalized_cot_log_probs.sum():.6f}")
        print(f"Reliance score: {internalized_cot_log_probs.sum() - cot_log_probs.sum():.6f}")

        return internalized_cot_log_probs.sum() - internalized_cot_log_probs.sum()
