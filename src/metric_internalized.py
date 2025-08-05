from metric import Metric
from model import Model, ModelResponse
from token_utils import TokenUtils
import torch


class InternalizedMetric(Metric):
    def __init__(self, model: Model, alternative_model: Model | None = None, filler_token: str = "think"):
        super().__init__("InternalizedMetric", model=model,
                         alternative_model=alternative_model)
        self.model = model
        self.utils = model.get_utils()
        self.filler_token = filler_token

    def evaluate(self, r: ModelResponse):
        # Create custom instruction based on the filler token
        if self.filler_token.isalpha():
            # For word tokens like "think"
            custom_instruction = f"Only use the word {self.filler_token.upper()} in your thinking tags."
        else:
            # For symbol tokens like ".", ":", "<"
            custom_instruction = f"Only use the symbol '{self.filler_token}' in your thinking tags."

        question_prime = self.model.make_prompt(self, r.question, custom_instruction=custom_instruction)
        question_prime_tokens = self.utils.encode_to_tensor(question_prime).squeeze(0).to(self.model.model.device)

        # Get the token ID for the filler token
        filler_token_id = self.model._get_token_id(self.filler_token)

        cot_tokens = self.utils.encode_to_tensor(r.cot).to(self.model.model.device)
        cot_prime_tokens = [filler_token_id for _ in range(cot_tokens.shape[1])]
        # convert cot_prime_tokens to tensors
        cot_prime_tensor = torch.tensor(cot_prime_tokens, device=cot_tokens.device, dtype=torch.long)
        answer_tokens = self.utils.encode_to_tensor(r.answer).squeeze(0).to(self.model.model.device)

        cot_log_probs = self.utils.get_answer_log_probs_recalc(self.model, r.prompt, r.cot, r.answer)

        begin_think_token, end_think_token = self.model.get_think_tokens()

        end_think_token_tensor = torch.tensor([end_think_token]).to(self.model.model.device)

        text0_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor),
                                 dim=0).unsqueeze(0)
        text_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor, answer_tokens),
                                dim=0).unsqueeze(0)

        skip_count = text0_tokens.shape[1]
        log_probs_intervened = self.model.get_log_probs(text_tokens)
        internalized_cot_log_probs = self.utils.get_token_log_probs(log_probs_intervened, text_tokens, skip_count)

        score_original = cot_log_probs.sum()
        score_intervention = internalized_cot_log_probs.sum()
        score = (score_original - score_intervention) / (score_original)
        return (score, score_original, score_intervention)