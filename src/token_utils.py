import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TokenUtils:
    def __init__(self, hf_model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    def decode_to_string(self, tokens: torch.Tensor, skip_special_tokens=True):
        return self.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens)

    def encode_to_tensor(self, string: str):
        return self.tokenizer.encode(
            string, return_tensors="pt").to(self.hf_model.device)

    def escape_string(self, string: str):
        return string.encode('unicode_escape').decode()

    def get_answer_log_probs_recalc(self, model, prompt: str, cot: str, prediction: str):
        """ Get log probs for just the answer (prediction), given prompt+cot+prediction.
        
            Note: prompt should end with a <think> token if required.
        """
        if cot == "":
            text0 = prompt + "</think> "
        else:
            text0 = prompt + cot + " </think> "
        text = text0 + prediction

        text0_tokens = self.encode_to_tensor(text0)
        text_tokens = self.encode_to_tensor(text)

        log_probs = model.get_logits(text_tokens)

        skip_count = text0_tokens.shape[1] - 1  # -1 for EOS token
        return self.get_token_log_probs(log_probs, text_tokens, skip_count)

    def get_answer_log_probs(self, prompt: str, cot: str, prediction: str, logits: torch.Tensor):
        """ Get log probs for just the answer (prediction), given prompt+cot+prediction.
        
            Note: prompt should end with a <think> token if required.
        """
        if cot == "":
            text0 = prompt + "</think> "
        else:
            text0 = prompt + cot + " </think> "
        text = text0 + prediction

        text0_tokens = self.encode_to_tensor(text0)
        text_tokens = self.encode_to_tensor(text)

        skip_count = text0_tokens.shape[1] - 1  # -1 for EOS token
        return self.get_token_log_probs(logits, text_tokens, skip_count)

    def get_token_log_probs(self, logits, tokens, start_index=0):        
        """Get probabilities for tokens from [start_index,end)."""
        batch_size, seq_len, vocab_size = logits.shape
        end_index = min(seq_len, tokens.shape[1])

        #print(f"start_index: {start_index}")
        #print(f"end_index: {end_index}")

        actual_tokens = tokens[0, start_index:]
        token_log_probs = logits[0, start_index:].gather(1, actual_tokens.unsqueeze(0)).squeeze(0)
        #token_log_probs = logits[0, start_index:].squeeze(0)

        print(f"getting log probs for tokens: {self.escape_string(self.decode_to_string(actual_tokens))}")
        print(f"log probs: {token_log_probs}")
        #print(f"actual tokens length: {len(actual_tokens)}")
        #print(f"log probs length: {len(token_log_probs)}")

        return token_log_probs