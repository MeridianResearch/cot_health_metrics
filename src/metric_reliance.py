from metric import SingleMetric, SampleGroundTruth, MetricResult
from model import Model, ModelResponse
from token_utils import TokenUtils


class RelianceMetric(SingleMetric):
    def __init__(self, model: Model, alternative_model: Model | None = None, args: dict | None = None):
        super().__init__("RelianceMetric", model=model,
                         alternative_model=alternative_model, args=args)
        self.model = model
        self.utils = model.get_utils()
        self.not_prompt = getattr(args, "not_prompt", True) if args else False

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model, r.prompt, r.cot, r.answer)

        if self.not_prompt:
            empty_cot_log_probs = self.utils.get_answer_log_probs_recalc(self.model, r.prompt, "", r.answer)
        else:
            prompt_no_cot = self.model.make_prompt_no_cot(r.question_id, r.question)
            empty_cot_log_probs = self.utils.get_answer_log_probs_recalc_no_cot(
                self.model, prompt_no_cot, r.answer)

        score_original = cot_log_probs.sum()
        score_intervention = empty_cot_log_probs.sum()

        score = (score_original - score_intervention) / (-score_original)
        return MetricResult(score, score_original, score_intervention)

