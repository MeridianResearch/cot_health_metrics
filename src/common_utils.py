from datasets import load_dataset
datasets_to_use = {
    "GSM8K": load_dataset("gsm8k", "main", split="train[:100]")
}
from enum import Enum
class SupportedModel(Enum):
    QWEN3_0_6B             = "Qwen/Qwen3-0.6B"
    QWEN3_1_7B             = "Qwen/Qwen3-1.7B"
    DEEPSEEK_QWEN_1_5B     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    MINI_THINK_BASE_1B     = "Wladastic/Mini-Think-Base-1B"
    GEMMA_2_2B             = "google/gemma-2-2b"
    # COGITO_LLAMA_3B      = "deepcogito/cogito-v1-preview-llama-3B"  # unverified
    # PHI_2                = "microsoft/phi-2"
def logits2logprob(logits,sequences):
    # Gather the log-prob for the actual next token at each position
    # We skip the first token since there's no prediction for it
    target_tokens = sequences[:, 1:]  # [B, T-1]
    predicted_logits = logits[:, :-1, :]  # [B, T-1, V]

    # Use gather to pick the log-prob of the actual next token
    actual_logp = predicted_logits.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # Compute mean log-prob per sequence
    mean_logp_per_seq = actual_logp.mean(dim=1)  # [B]
    return mean_logp_per_seq