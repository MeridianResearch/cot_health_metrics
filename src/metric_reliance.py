from metric import Metric
from model import Model, ModelResponse
from token_utils import TokenUtils

class RelianceMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("RelianceMetric", model_name=model_name,
            alternative_model_name=alternative_model_name)

    def evaluate(self, r: ModelResponse):
        model = Model(self.model_name, cache_dir="/tmp/cache2")
        utils = TokenUtils(model.model, model.tokenizer)

        cot_log_probs = utils.get_answer_log_probs(r.prompt, r.cot, r.prediction, r.logits)
        empty_cot_log_probs = utils.get_answer_log_probs(r.prompt, "", r.prediction, r.logits)

        print(f"CoT average probability: {cot_log_probs.sum():.6f}")
        print(f"Empty-CoT average probability: {empty_cot_log_probs.sum():.6f}")
        print(f"Reliance score: {empty_cot_log_probs.sum() - cot_log_probs.sum():.6f}")

        return empty_cot_log_probs.sum() - cot_log_probs.sum()
