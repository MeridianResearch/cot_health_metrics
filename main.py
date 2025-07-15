import random
import pandas as pd
import torch.nn.functional as F
import json
import torch
from common_utils import generate_cot_response,tokenizers,llms,datasets_to_use
def compute_logp_context_answer(tokenizer, model, context, answer, device="cuda"):
    # 1) Tokenize separately and cast to LongTensor on device
    ctx_enc = tokenizer(context, return_tensors="pt")
    ans_enc = tokenizer(answer,  return_tensors="pt")
    ctx_ids = ctx_enc["input_ids"].to(device).long()    # [1, ctx_len]
    ans_ids = ans_enc["input_ids"].to(device).long()    # [1, ans_len]

    running_ids = ctx_ids                               # start with context only
    logps = []

    # 2) Loop over each true answer token
    with torch.no_grad():
        for next_token in ans_ids[0]:
            # forward on current running_ids (dtype=long)
            outputs     = model(input_ids=running_ids)
            last_logits = outputs.logits[:, -1, :]                 # [1, vocab]
            last_logps  = F.log_softmax(last_logits, dim=-1)       # [1, vocab]

            # pick out log-prob of the true next_token
            logp = last_logps[0, next_token].item()
            logps.append(logp)

            # append the true token for next iteration
            next_id = torch.tensor([[next_token]], device=device).long()
            running_ids = torch.cat([running_ids, next_id], dim=1)

    tok_logps  = torch.tensor(logps, device=device)
    total_logp = tok_logps.sum().item()
    total_p    = torch.exp(tok_logps.sum()).item()
    avg_nll    = -tok_logps.mean().item()
    return {
        "avg_neg_log_likelihood":          avg_nll,
        "total_logp_answer_given_context": total_logp,
        "p_answer_given_context":          total_p,
        "per_token_logps":                 logps
    }

# === Benchmark loop ===
results = []
for dataset_name, dataset in datasets_to_use.items():
    print(f"\n====== Dataset: {dataset_name} ======")
    for i in range(len(dataset)):
        sample = random.choice(dataset)
        question = sample["question"]
        groundtruth_cot = sample["answer"].split("####")[0]
        groundtruth_answer = sample["answer"].split("####")[1]
        # === Generate reasoning chain (R1) ===


        # === A1: LLM1 consuming R1 ===
        R1,A1,logprob_M1_A1 = generate_cot_response(tokenizers["LLM1"],llms["LLM1"], question)
        input_M2=".".join([question,R1,A1])
        input_tokenized=tokenizers["LLM2"].encode(input_M2)

        log_probs = F.log_softmax(llms["LLM2"](**input_tokenized).logits, dim=-1)
        ids = input_tokenized["input_ids"].unsqueeze(-1)  # [B, L, 1]
        token_logps = log_probs.gather(2, ids).squeeze(-1)  # [B, L]

        # 4) Mask out padding if you have an attention_mask
        if "attention_mask" in input_tokenized:
            mask = input_tokenized["attention_mask"]
            token_logps = token_logps * mask

        # 5) Compute the **mean log‑prob per sequence** (over L tokens)
        mean_logps_M2 = token_logps.sum(dim=-1) / (
            mask.sum(dim=-1) if "attention_mask" in input_tokenized else token_logps.size(1))
        # shape [B]
        results.append({
            "dataset": dataset_name,
            "question": question,
            "groundtruth_cot": groundtruth_cot,
            "groundtruth_answer": groundtruth_answer,
            "R1": R1,
            "A1": A1,
            "log_prob_A1_R1_M2": mean_logps_M2,
            "log_prob_A1_M1": logprob_M1_A1


        })
    # Write to JSON file
    with open(f"transferability_results_{dataset_name}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
# === Summary dataframe ===
df = pd.DataFrame(results)
print("\n=== Summary of transferability metrics ===")
# === Compute average of each column ===
mean_scores = df.mean(numeric_only=True)
print("\n=== Average scores for each metric ===")
print(mean_scores)
