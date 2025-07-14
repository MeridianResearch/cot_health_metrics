from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/cache2")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def generate_cot_response(question, max_new_tokens=4096):
    """Generate a response using Chain-of-Thought (CoT) prompting."""
    prompt = f"Question: {question}\nLet's think step by step. <think>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    # Return only the answer part
    return response[len(prompt):].strip()

print(tokenizer.convert_tokens_to_ids("<think>"))
if __name__ == "__main__":
    question = "A car travels 60 miles in 1.5 hours. What is its average speed?"
    answer = generate_cot_response(question)
    print("Model answer:\n", answer)
