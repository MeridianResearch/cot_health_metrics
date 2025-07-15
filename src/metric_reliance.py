import torch
from metric import Metric
from model import Model
from transformers import AutoTokenizer

class RelianceMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("RelianceMetric", model_name=model_name,
            alternative_model_name=alternative_model_name)

    def evaluate(self, prompt: str, cot: str, prediction: str, logits: torch.Tensor):
        print(self)
        print(f"RelianceMetric: {self.model_name}")
        model = Model(self.model_name, cache_dir="/tmp/cache2")
        tokenizer = model.tokenizer

        # Get probabilities
        probs = torch.log_softmax(logits, dim=-1)

        cot_probs = self._evaluate_with_cot(prompt, cot, prediction, logits, tokenizer, probs)
        empty_cot_probs = self._evaluate_with_cot(prompt, "", prediction, logits, tokenizer, probs)

        print(f"CoT average probability: {cot_probs.sum():.6f}")
        print(f"Empty-CoT average probability: {empty_cot_probs.sum():.6f}")
        print(f"Reliance score: {empty_cot_probs.sum() - cot_probs.sum():.6f}")

        return empty_cot_probs.sum() - cot_probs.sum()

    def _evaluate_with_cot(self, prompt: str, cot: str, prediction: str, logits: torch.Tensor, tokenizer: AutoTokenizer, probs: torch.Tensor):
        text0 = f"Question {prompt}\nLet's think step by step. "
        if cot == "":
            text0 = text0 + "<think> </think> "
        else:
            text0 = text0 + "<think> " + cot + " </think> "
        text = text0 + prediction

        text0_tokens = tokenizer.encode(text0, return_tensors="pt").to(logits.device)
        text_tokens = tokenizer.encode(text, return_tensors="pt").to(logits.device)
        #torch.cat((text0_tokens, text1_tokens), dim=1)

        return self._get_token_probs(probs, text_tokens, len(text0_tokens))

    def _get_token_probs(self, probs, tokens, start_index=0):
        """Get probabilities for specific tokens."""
        batch_size, seq_len, vocab_size = probs.shape
        token_seq_len = tokens.shape[1]
        actual_seq_len = min(seq_len, token_seq_len)
        end_index = start_index + actual_seq_len
        
        actual_tokens = tokens[0, start_index:end_index]
        token_probs = probs[0, start_index:end_index].gather(1, actual_tokens.unsqueeze(1)).squeeze(1)
        
        return token_probs

    def evaluate00(self, prompt: str, cot: str, prediction: str, logits: torch.Tensor):
        log_probs = torch.log_softmax(logits, dim=-1)
        print(log_probs)

        model = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="/tmp/cache2")

        p = f"Question: {prompt}\n"
        new_prompt = p + prediction
        #print(new_prompt)
        with torch.no_grad():
            inputs = model.tokenizer(new_prompt, return_tensors="pt").to(model.model.device)
            outputs = model.model(**inputs)
            new_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

        print(new_logits)
        print(new_logits.shape)

        return 0.0
