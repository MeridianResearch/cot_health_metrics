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
        probs = torch.softmax(logits, dim=-1)

        # Tokenize the full text
        cot_text = f"Question {prompt}\nLet's think step by step. <think> " + cot + " </think> " + prediction
        new_text = f"Question {prompt}\n<think> </think> " + prediction

        # Separate CoT and prediction tokens
        cot_tokens = tokenizer.encode(cot_text, return_tensors="pt").to(logits.device)
        prediction_tokens = tokenizer.encode(new_text, return_tensors="pt").to(logits.device)
        
        # Get probabilities for CoT vs prediction
        cot_probs = self._get_token_probs(probs, cot_tokens)
        prediction_probs = self._get_token_probs(probs, prediction_tokens)
        
        print(f"CoT average probability: {cot_probs.mean().item():.6f}")
        print(f"Prediction average probability: {prediction_probs.mean().item():.6f}")
        
        # Reliance metric: compare confidence between CoT and prediction
        reliance_score = prediction_probs.mean().item() - cot_probs.mean().item()
        
        return reliance_score

    def _get_token_probs(self, probs, tokens):
        """Get probabilities for specific tokens."""
        batch_size, seq_len, vocab_size = probs.shape
        token_seq_len = tokens.shape[1]
        actual_seq_len = min(seq_len, token_seq_len)
        
        actual_tokens = tokens[0, :actual_seq_len]
        token_probs = probs[0, :actual_seq_len].gather(1, actual_tokens.unsqueeze(1)).squeeze(1)
        
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
