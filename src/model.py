from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Model:
    def __init__(self, model_name: str, cache_dir="/tmp/cache"):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def generate_cot_response(self, question, max_new_tokens=4096):
        """Generate a response using Chain-of-Thought (CoT) prompting."""
        prompt = f"Question: {question}\nLet's think step by step. <think>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_k=20,
            min_p=0.0,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # split the output into two parts: the chain of thought and the answer
        begin_think = self._get_token_id("<think>")
        if(output[0][0] == begin_think):
            output[0] = output[0][1:]
        end_think = self._get_token_id("</think>")
        pieces = self._split_on_tokens(output[0].tolist(), [end_think])

        if(len(pieces) < 2):
            print(f"ERROR: model {self.model_name} did not generate chain of thought")
            exit(1)

        response0 = self.tokenizer.decode(pieces[0], skip_special_tokens=True)
        response1 = self.tokenizer.decode(pieces[1], skip_special_tokens=True)

        return (response0[len(prompt):].strip(), response1.strip())

    def _split_on_tokens(self, lst, token_list):
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

    def _get_token_id(self, token):
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if(token_id is None):
            print(f"ERROR: model {self.model_name} does not support {token} token")
            exit(1)
        return token_id

if __name__ == "__main__":
    question = "A car travels 60 miles in 1.5 hours. What is its average speed?"
    print("Prompt: " + question.encode('unicode_escape').decode())

    model = Model("Qwen/Qwen3-0.6B", cache_dir="/tmp/cache2")
    (cot, answer) = model.generate_cot_response(question)
    print("\n")
    print("CoT: " + cot.encode('unicode_escape').decode())
    print("\n")
    print("Answer: " + answer.encode('unicode_escape').decode())
