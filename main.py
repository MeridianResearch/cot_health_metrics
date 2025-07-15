from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import random
import pandas as pd
import torch.nn.functional as F
import json
# === Cache & model config ===
cache_folder = "/tmp/cache_linh/"
models = {
    "LLM2": "Qwen/Qwen3-0.6B",
    "LLM1":"Qwen/Qwen3-1.7B",
    "LLM3": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "LLM4": "deepcogito/cogito-v1-preview-llama-3B",
    "LLM5":"Wladastic/Mini-Think-Base-1B"
}
def compute_logp_context_answer( tokenizer, model,context, answer, device="cuda"):
    # 1) Tokenize separately
    ctx_enc = tokenizer(context, return_tensors="pt")
    ans_enc = tokenizer(answer,  return_tensors="pt")
    ctx_ids = ctx_enc["input_ids"].to(device)    # [1, ctx_len]
    ans_ids = ans_enc["input_ids"].to(device)    # [1, ans_len]

    running_ids = ctx_ids                       # start with context only
    logps = []

    # 2) Loop over each true answer token
    with torch.no_grad():
        for next_token in ans_ids[0]:
            # forward on current running_ids
            outputs    = model(input_ids=running_ids)
            last_logits = outputs.logits[:, -1, :]                 # [1, vocab]
            last_logps  = F.log_softmax(last_logits, dim=-1)      # [1, vocab]

            # pick out log-prob of the true next_token
            logp = last_logps[0, next_token].item()
            logps.append(logp)

            # append the true token for next iteration
            running_ids = torch.cat(
                [running_ids, next_token.view(1,1)], dim=1
            )

    tok_logps   = torch.tensor(logps, device=device)
    total_logp  = tok_logps.sum().item()
    total_p     = torch.exp(tok_logps.sum()).item()
    avg_nll     = -tok_logps.mean().item()

    return {
        "avg_neg_log_likelihood":          avg_nll,
        "total_logp_answer_given_context": total_logp,
        "p_answer_given_context":          total_p,
        "per_token_logps":                 logps
    }

def compute_logprob(tokenizer, model, context, target):
    """
    Computes the log probability of `target` sequence given `context` prompt.
    """
    full_input = context + target
    inputs = tokenizer(full_input, return_tensors="pt").to("cuda")
    with tokenizer.as_target_tokenizer():
        target_ids = tokenizer(target, return_tensors="pt").input_ids.to("cuda")

    # Shift so the loss only covers the target
    labels = torch.full_like(inputs.input_ids, -100)
    labels[:, -target_ids.shape[1]:] = target_ids

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, labels=labels)
        # outputs.loss is average negative log likelihood (cross-entropy)
        avg_neg_logprob = outputs.loss.item()
        # logprob = -avg_neg_logprob * target_ids.shape[1]

    return avg_neg_logprob

# === Load datasets ===
datasets_to_use = {
    "GSM8K": load_dataset("gsm8k", "main", split="train[:100]")
}
import re
def split_on_tokens(lst, token_list):
    """Split a list into sublists, using 'token' as the delimiter (token is not included in results)."""
    result = []
    current = []
    for item in lst:
        if item in token_list:
            result.append(current)
            current = []
        else:
            current.append(item)
    result.append(current)
    return result

# === Load models & tokenizers ===
tokenizers, llms = {}, {}
for name, path in models.items():
    tokenizers[name] = AutoTokenizer.from_pretrained(path, cache_dir=cache_folder)
    llms[name] = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_folder)

# === SBERT for common embeddings ===
sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
def embed_text_sbert(text):
    return sbert.encode(text, convert_to_tensor=False)

# === Extract chain-of-thought from response ===
def extract_cot(response):
    match = re.search(r"<think>(.*)", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else response.strip()


def generate_cot_response(tokenizer, model, question=None, cot_input=None, max_new_tokens=4096):
    """
    Flexible version:
    - If `question` is given, generates chain-of-thought reasoning from scratch.
    - If `cot_input` is given, uses that reasoning chain to compute final answer.
    Also splits based on your <think> token logic.
    """
    if question:
        prompt = f"Question: {question}\nLet's think step by step. <think>"
    elif cot_input:
        prompt = f"""
Here is a detailed chain of thought from another expert model:
<begin-cot>
{cot_input}
<end-cot>

Please provide the final answer based exactly on this reasoning for this question {question}, without adding new steps. <think>
""".strip()
    else:
        raise ValueError("Must provide either `question` or `cot_input`.")

    # Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Use your split_on_tokens strategy
    begin_think = tokenizer.convert_tokens_to_ids("<think>")
    if output[0][0] == begin_think:
        output[0] = output[0][1:]

    end_think = tokenizer.convert_tokens_to_ids("</think>")
    pieces = split_on_tokens(output[0].tolist(), [end_think])

    # Return structured output
    response0 = tokenizer.decode(pieces[0], skip_special_tokens=True)
    response1 = tokenizer.decode(pieces[1], skip_special_tokens=True) if len(pieces) > 1 else ""

    return (response0[len(prompt):].strip(), response1.strip())




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

        R1,A1 = generate_cot_response(tokenizers["LLM1"],llms["LLM1"], question)
        follow_trace_prompt = f"Here is a detailed chain of thought:\n{R1}\nBased on this reasoning, what is the final answer?"
        # === A2: LLM2 consuming R1 ===
        R2,A2 = generate_cot_response(tokenizers["LLM2"],llms["LLM2"], question)
        R2_R1, A2_R1 = generate_cot_response(tokenizers["LLM2"], llms["LLM2"], question,cot_input=R1)
        # === Compute cosine similarities ===
        sim_A1A2 = 1 - cosine(embed_text_sbert(A1), embed_text_sbert(A2))
        sim_A1A2_R1 = 1 - cosine(embed_text_sbert(A1), embed_text_sbert(A2_R1))
        # sim_R1_GT = 1 - cosine(embed_text_sbert(R1), embed_text_sbert(groundtruth))
        logp_M2_A1R1 = compute_logp_context_answer(tokenizers["LLM2"], llms["LLM2"], R1, A1)
        logp_M1_A1R1 = compute_logp_context_answer(tokenizers["LLM1"], llms["LLM1"], R1, A1)
        logp_M1_A2R1 = compute_logp_context_answer(tokenizers["LLM1"], llms["LLM1"], R1, A2_R1)
        logp_M1_gt = compute_logp_context_answer(tokenizers["LLM1"], llms["LLM1"], groundtruth_cot, groundtruth_answer)
        logp_M2_gt = compute_logp_context_answer(tokenizers["LLM2"], llms["LLM2"], groundtruth_cot,
                                                   groundtruth_answer)

        # print(R1,"\n",R2_R1,"\n",R2 )
        # === Print summary ===
        print(f"\n[{dataset_name}] Example {i+1}")
        print(f"Q: {question[:80]}...")
        print(f"R1: {R1[:200]}...")
        print(f"A1: {A1[:200]}...")
        print(f"A2: {A2[:200]}...")
        print(f":logp_M2_A1R1 {logp_M2_A1R1['p_answer_given_context']}...")
        print(f":logp_M1_A1R1 {logp_M1_A1R1['p_answer_given_context']}...")
        print(f":logp_M1_A2R1 {logp_M1_A2R1['p_answer_given_context']}...")
        # print(f"Cosine(A1, A2): {sim_A1A2:.3f}")
        # print(f"Cosine(A1, A2_R1): {sim_A1A2_R1:.3f}")
        # print(f"Cosine(R1, GT): {sim_R1_GT:.3f}")

        results.append({
            "dataset": dataset_name,
            "question": question,
            "groundtruth_cot": groundtruth_cot,
            "groundtruth_answer": groundtruth_answer,
            "R1": R1,
            "A1": A1,
            "R2": R2,
            "A2": A2,
            "A2_R1": A2_R1,
            "sim_A1A2": sim_A1A2,
            "sim_A1A2_R1": sim_A1A2_R1,
            # "sim_R1_GT": sim_R1_GT,
            "logp_M2_A1R1": logp_M2_A1R1["p_answer_given_context"],
            "logp_M1_A1R1": logp_M1_A1R1["p_answer_given_context"],
            "logp_M2_A1R1": logp_M2_A1R1["p_answer_given_context"],
            "logp_M2_A1R1": logp_M2_A1R1["p_answer_given_context"],
            "logp_M1_gt2": logp_M1_gt["p_answer_given_context"],
            "logp_M2_gt2": logp_M2_gt["p_answer_given_context"],

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
