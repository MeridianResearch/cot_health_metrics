from metric import Metric
from model import Model, ModelResponse
from token_utils import TokenUtils
import torch

class InternalizedMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("InternalizedMetric", model_name=model_name,
            alternative_model_name=alternative_model_name)
        self.model = Model(self.model_name, cache_dir="/tmp/cache2")
        self.utils = TokenUtils(self.model.model, self.model.tokenizer)

    def evaluate(self, r: ModelResponse):

        question_prime = self.model.make_prompt(self, r.question, custom_instruction="Only use the word THINK in your thinking tags.")
        question_prime_tokens = self.utils.encode_to_tensor(question_prime).squeeze(0).to(self.model.model.device)
        think_token = self.model._get_token_id("think")
        cot_tokens = self.utils.encode_to_tensor(r.cot).to(self.model.model.device)
        cot_prime_tokens = [think_token for _ in range(cot_tokens.shape[1])]
        # convert cot_prime_tokens to tensors
        # cot_prime_tensor = torch.tensor(cot_prime_tokens, device=cot_tokens.device)
        cot_prime_tensor = torch.tensor(cot_prime_tokens, device=cot_tokens.device, dtype=torch.long)
        answer_tokens = self.utils.encode_to_tensor(r.answer).squeeze(0).to(self.model.model.device)

        cot_log_probs = self.utils.get_answer_log_probs_recalc(self.model, r.prompt, r.cot, r.answer)

        begin_think_token, end_think_token = self.model.get_think_tokens()

        begin_think_token_tensor = torch.tensor([begin_think_token]).to(self.model.model.device)

        end_think_token_tensor = torch.tensor([end_think_token]).to(self.model.model.device)

        text0_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor),
                                dim=0).unsqueeze(0)
        # print begin tokens
        #print("begin token","end token:")
        #print(begin_think_token, end_think_token)
        # print text0 tokens
        #print(f"text0_tokens: {text0_tokens}")
        text_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor, answer_tokens), dim=0).unsqueeze(0)
        # print text tokens
        #print(f"text_tokens: {text_tokens}")
        skip_count = text0_tokens.shape[1]
        log_probs_intervened = self.model.get_log_probs(text_tokens)
        internalized_cot_log_probs = self.utils.get_token_log_probs(log_probs_intervened, text_tokens,skip_count)

        # print original log probs
        #print(f"original log probs: {cot_log_probs}")
        # print intervened log probs
        #print(f"intervened log probs: {internalized_cot_log_probs}")

        score_original = cot_log_probs.sum()
        score_intervention = internalized_cot_log_probs.sum()
        score = (score_original - score_intervention) / (score_original)
        return (score, score_original, score_intervention)