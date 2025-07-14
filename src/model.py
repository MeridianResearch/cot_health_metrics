from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/cache2")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

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

    begin_think = tokenizer.convert_tokens_to_ids("<think>")
    if(output[0][0] == begin_think):
        output[0] = output[0][1:]
    end_think = tokenizer.convert_tokens_to_ids("</think>")
    pieces = split_on_tokens(output[0].tolist(), [end_think])

    response0 = tokenizer.decode(pieces[0], skip_special_tokens=True)
    response1 = tokenizer.decode(pieces[1], skip_special_tokens=True)

    return (response0[len(prompt):].strip(), response1.strip())

if __name__ == "__main__":
    question = "A car travels 60 miles in 1.5 hours. What is its average speed?"
    print("Prompt: " + question.encode('unicode_escape').decode())
    (cot, answer) = generate_cot_response(question)
    print("\n")
    print("CoT: " + cot.encode('unicode_escape').decode())
    print("\n")
    print("Answer: " + answer.encode('unicode_escape').decode())
