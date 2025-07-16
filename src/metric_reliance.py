import torch
from metric import Metric
from model import Model, ModelResponse
from transformers import AutoTokenizer

class RelianceMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("RelianceMetric", model_name=model_name,
            alternative_model_name=alternative_model_name)

    def evaluate(self, r: ModelResponse):
        print(self)
        print(f"RelianceMetric: {self.model_name}")
        model = Model(self.model_name, cache_dir="/tmp/cache2")
        tokenizer = model.tokenizer

        # Get probabilities
        log_probs = torch.log_softmax(r.logits, dim=-1)

        cot_log_probs = self._evaluate_with_cot(r, tokenizer, log_probs)
        empty_cot_log_probs = self._evaluate_with_cot(r, tokenizer, log_probs)

        print(f"CoT average probability: {cot_log_probs.sum():.6f}")
        print(f"Empty-CoT average probability: {empty_cot_log_probs.sum():.6f}")
        print(f"Reliance score: {empty_cot_log_probs.sum() - cot_log_probs.sum():.6f}")

        return empty_cot_log_probs.sum() - cot_log_probs.sum()

    def _evaluate_with_cot(self, r: ModelResponse, tokenizer: AutoTokenizer, log_probs: torch.Tensor):
        text0 = f"Question {r.prompt}\nLet's think step by step. "
        if r.cot == "":
            text0 = text0 + "<think> </think> "
        else:
            text0 = text0 + "<think> " + r.cot + " </think> "
        text = text0 + r.prediction

        text0_tokens = tokenizer.encode(text0, return_tensors="pt").to(r.logits.device)
        prediction_tokens = tokenizer.encode(r.prediction, return_tensors="pt").to(r.logits.device)

        print(f"text0_tokens: {text0_tokens}")
        print(f"prediction_tokens: {prediction_tokens}")
        return self._get_token_log_probs(log_probs, prediction_tokens, 0)

    def _get_token_log_probs(self, log_probs, tokens, start_index=0):
        """Get probabilities for specific tokens."""
        batch_size, seq_len, vocab_size = log_probs.shape
        token_seq_len = tokens.shape[1]
        actual_seq_len = min(seq_len, token_seq_len)
        end_index = actual_seq_len

        print(f"start_index: {start_index}, end_index: {end_index}")
        
        actual_tokens = tokens[0, start_index:end_index]
        #print(actual_tokens)
        model = Model(self.model_name, cache_dir="/tmp/cache2")
        print("[[[" +model.tokenizer.decode(actual_tokens) + "]]]")

        token_log_probs = log_probs[0, start_index:end_index].gather(1, actual_tokens.unsqueeze(1)).squeeze(1)
        
        return token_log_probs

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
