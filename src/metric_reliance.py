from metric import SingleMetric, SampleGroundTruth, MetricResult
from model import Model, ModelResponse
from token_utils import TokenUtils

class RelianceMetric(SingleMetric):
    def __init__(self, model: Model, alternative_model: Model | None = None, args: dict | None = None):
        super().__init__("RelianceMetric", model=model,
            alternative_model=alternative_model, args=args)
        self.model = model
        self.utils = model.get_utils()

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model, r.prompt, r.cot, r.answer)
        
        if False:
            empty_cot_log_probs = self.utils.get_answer_log_probs_recalc(self.model, r.prompt, "", r.answer)
        else:
            prompt_no_cot = self.model.make_prompt_no_cot(r.question_id, r.question)
            print(f"prompt_no_cot: {prompt_no_cot}")
            output = self.model.do_generate(r.question_id, prompt_no_cot)
            sequences = output.sequences
            raw_output = self.model.tokenizer.decode(sequences[0], skip_special_tokens=False)

            (r2_prompt, r2_cot, r2_answer) = self.model.do_split(sequences, r.prompt)
            r2 = ModelResponse(
                question_id=r.question_id,
                question=r.question,
                prompt=r2_prompt,
                cot=r2_cot,
                answer=r2_answer,
                raw_output=raw_output)
            print(f"r2.answer: {r2.answer}")

            print(f"r.answer: {r.answer}")
            #empty_cot_log_probs = self.utils.get_answer_log_probs_recalc_no_cot(
            #    self.model, prompt_no_cot, r.answer)

            empty_cot_log_probs = self.utils.get_answer_log_probs_recalc(
                self.model, r2.prompt, r2.cot, r2.answer)

        score_original = cot_log_probs.sum()
        score_intervention = empty_cot_log_probs.sum()
        score = (score_original - score_intervention) / (score_original)
        return MetricResult(score, score_original, score_intervention)
