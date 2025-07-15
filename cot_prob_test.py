import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def batch_token_probs_for_answer(model, tokenizer, prompts, answers, device=None):
    device = device or model.device

    # 1) Batch‐encode prompts with padding and truncation
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings,
    )
    input_ids = enc["input_ids"].to(device)            # [B, L]
    attention_mask = enc["attention_mask"].to(device)  # [B, L]

    # 2) Guarantee no empty sequences (shouldn’t happen with padding=True)
    #    but just in case, replace all‐pad rows with a single eos_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    all_pad = (input_ids == pad_id).all(dim=1)
    if all_pad.any():
        input_ids[all_pad] = eos_id
        attention_mask[all_pad] = 1

    # 3) Generate one step with scores
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=pad_id,
    )
    # out.scores[0] has shape [B, vocab_size]
    first_logits = out.scores[0]

    # 4) Tokenize each answer (no special tokens)
    answer_ids_list = [
        tokenizer(a, add_special_tokens=False).input_ids
        for a in answers
    ]

    # 5) For each batch item, gather logits and compute softmax
    batch_probs = []
    for i, ans_ids in enumerate(answer_ids_list):
        if len(ans_ids) == 0:
            batch_probs.append([])
            continue
        # first_logits[i]: [vocab_size]
        logits_i = first_logits[i, ans_ids]            # [len(ans_ids)]
        probs_i = torch.softmax(logits_i, dim=-1).tolist()
        batch_probs.append(probs_i)

    return batch_probs

# === Example usage ===
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen3-0.6B"
tokenizer  = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/cache_linh/")
model      = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/tmp/cache_linh/").to(device)

prompts = [
    "What is 2+2?",
    "How many sides does a triangle have?"
]
answers = [
    "4",            # maybe token IDs [13]
    "3"             # maybe token IDs [7]
]

probs = batch_token_probs_for_answer(model, tokenizer, prompts, answers)
for prompt, answer, p in zip(prompts, answers, probs):
    print(f"Prompt: {prompt}")
    print(f"Answer token IDs: {tokenizer(answer, add_special_tokens=False).input_ids}")
    print(f"Probabilities over those tokens: {p}\n")
