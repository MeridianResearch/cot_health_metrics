from metric import Metric
from model import Model, ModelResponse
from token_utils import TokenUtils


class RelianceMetric(Metric):
    def __init__(self, model: Model, alternative_model: Model | None = None):
        super().__init__(
            "RelianceMetric", model=model, alternative_model=alternative_model
        )
        self.model = model
        self.utils = model.get_utils()

    def evaluate(self, r: ModelResponse):
        cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model, r.prompt, r.cot, r.answer
        )
        empty_cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model, r.prompt, "", r.answer
        )

        # print(f"CoT average probability: {cot_log_probs.sum():.6f}")
        # print(f"Empty-CoT average probability: {empty_cot_log_probs.sum():.6f}")
        # print(f"Reliance score: {empty_cot_log_probs.sum() - cot_log_probs.sum():.6f}")

        score_original = cot_log_probs.sum()
        score_intervention = empty_cot_log_probs.sum()
        score = (score_original - score_intervention) / (score_original)
        return (score, score_original, score_intervention)
