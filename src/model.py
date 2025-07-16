from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass

@dataclass
class ModelResponse:
    question: str
    prompt: str
    cot: str
    prediction: str
    raw_output: str
    logits: torch.Tensor

    def __post_init__(self):
        self.basic_pair = (self.cot, self.prediction)

    def old__str__(self):
        return f"""
ModelResponse(
    Prompt: {self._encode(self.prompt)}
    CoT: {self._encode(self.cot)}
    Prediction: {self._encode(self.prediction)}
)
"""

    def print(self):
        print(f"Prompt: {self._encode(self.prompt)}")
        print("\n")
        print("CoT: " + self._encode(self.cot))
        print("\n")
        print(f"Prediction: {self._encode(self.prediction)}")
        print("\n")

class Model:
    MODEL_CONFIG_QWEN = {
        "begin_think": "<think>",
        "end_think": "</think>",
    }

    MODEL_CONFIG_WLA = {
        "fuzzy_separator": "Answer: ",
    }

    SUPPORTED_MODELS = {
        "Qwen/Qwen3-0.6B": MODEL_CONFIG_QWEN,
        "Qwen/Qwen3-1.7B": MODEL_CONFIG_QWEN,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": MODEL_CONFIG_QWEN,
        #"deepcogito/cogito-v1-preview-llama-3B": MODEL_CONFIG_QWEN,  # unverified
        "Wladastic/Mini-Think-Base-1B": MODEL_CONFIG_WLA,
        "google/gemma-2-2b": MODEL_CONFIG_WLA,
        #"microsoft/phi-2": MODEL_CONFIG_WLA,  # not very consistent
    }

    def __init__(self, model_name: str, cache_dir="/tmp/cache"):
        if model_name not in self.SUPPORTED_MODELS:
            print(f"ERROR: model {model_name} is not in supported list {self.SUPPORTED_MODELS}")
            exit(1)

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
        final_response = self.generate_cot_response_full(question, max_new_tokens)
        return final_response.basic_pair

    def make_prompt(self, question, custom_instruction="Let's think step by step."):
        model_config = self.SUPPORTED_MODELS[self.model_name]
        if("begin_think" in model_config):
            return f"Question: {question}\n{custom_instruction} <think>"
        elif("fuzzy_separator" in model_config):
            return f"Question: {question}\n{custom_instruction}"
        else:
            print(f"ERROR: model {self.model_name} missing CoT separator config")
            exit(1)

    def do_generate(self, prompt, max_new_tokens=4096):
        """Generate a response using Chain-of-Thought (CoT) prompting."""
        model_config = self.SUPPORTED_MODELS[self.model_name]

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
            output_scores=True,
            return_dict_in_generate=True,
        )
        return output

    def get_logits(self, sequences):
        with torch.no_grad():
            outputs = self.model(input_ids=sequences)
            logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        return logits

    def do_split(self, sequences, prompt):
        model_config = self.SUPPORTED_MODELS[self.model_name]

        # split the output into two parts: the chain of thought and the answer
        if("begin_think" in model_config):
            # Split before decoding
            begin_think = self._get_token_id(model_config["begin_think"])
            if(sequences[0][0] == begin_think):
                sequences[0] = sequences[0][1:]
            end_think = self._get_token_id(model_config["end_think"])
            pieces = self._split_on_tokens(sequences[0].tolist(), [end_think])

            if(len(pieces) < 2):
                print(f"ERROR: model {self.model_name} did not generate chain of thought")
                print("Response: " + full_response)
                exit(1)

            response0 = self.tokenizer.decode(pieces[0], skip_special_tokens=True)
            response1 = self.tokenizer.decode(pieces[1], skip_special_tokens=True)

            cot = response0[len(prompt):].strip()
            prediction = response1.strip()
        elif("fuzzy_separator" in model_config):
            if(model_config["fuzzy_separator"] in full_response):
                pieces = full_response.split(model_config["fuzzy_separator"])
            else:
                print(f"ERROR: model {self.model_name} did not generate chain of thought separator {model_config['fuzzy_separator']}")
                print(f"Response: {full_response}")
                exit(1)
            cot = pieces[0][len(prompt):].strip()
            prediction = pieces[1].strip()
        else:
            print(f"ERROR: model {self.model_name} missing CoT separator config")
            exit(1)
        return (cot, prediction)

    def generate_cot_response_full(self, question, max_new_tokens=4096):
        """Generate a response using Chain-of-Thought (CoT) prompting."""
        prompt = self.make_prompt(question)
        output = self.do_generate(prompt, max_new_tokens)
        sequences = output.sequences
        logits = self.get_logits(sequences)

        raw_output = self.tokenizer.decode(sequences[0], skip_special_tokens=True)

        (cot, prediction) = self.do_split(sequences, question)

        return ModelResponse(
            question=question,
            prompt=prompt,
            cot=cot,
            prediction=prediction,
            raw_output=raw_output,
            logits=logits)

    def evaluate_cot_response_from_tokens(self, prompt_tokens: torch.Tensor, max_new_tokens=4096):
        logits = self.get_logits(prompt_tokens)

        print(prompt_tokens)
        raw_output = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)

        (cot, prediction) = self.do_split(logits, question)

        return ModelResponse(
            question=question,
            prompt=prompt_tokens,
            cot=cot,
            prediction=prediction,
            raw_output=raw_output,
            logits=logits)

    def evaluate_cot_response(self, prompt: str, max_new_tokens=4096):
        """Generate a response using Chain-of-Thought (CoT) prompting."""
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        return self.evaluate_cot_response_from_tokens(prompt_tokens, max_new_tokens)

    def evaluate_with_custom_cot_tokens(self, question, original_cot_tokens):
        model_config = self.SUPPORTED_MODELS[self.model_name]

        prompt0 = self.make_prompt(question, custom_instruction="Only use the word THINK in your thinking tags.")
        prompt0_tokens = self.tokenizer.encode(prompt0, return_tensors="pt").to(self.model.device)

        begin_think = self._get_token_id(model_config["begin_think"])
        end_think = self._get_token_id(model_config["end_think"])

        think_token = self._get_token_id("think")
        cot_prime_tokens = [think_token for _ in range(len(original_cot_tokens))]

        # Convert EVERYTHING to tensors, shaped [1, N]
        begin_think_tensor = torch.tensor([[begin_think]], device=self.model.device)
        end_think_tensor = torch.tensor([[end_think]], device=self.model.device)
        cot_prime_tensor = torch.tensor([cot_prime_tokens], device=self.model.device)

        # Now cat along dim=1 (columns)
        full_prime = torch.cat((prompt0_tokens, begin_think_tensor, cot_prime_tensor, end_think_tensor), dim=1)
        return self.evaluate_cot_response_from_tokens(full_prime.squeeze(0))


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

    model = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="/tmp/cache2")
    (cot, answer) = model.generate_cot_response(question)
    print("\n")
    print("CoT: " + cot.encode('unicode_escape').decode())
    print("\n")
    print("Answer: " + answer.encode('unicode_escape').decode())
    print("\n")
    print("Evaluate replacing CoT with thinking TOKENS:")
    print("\n")
    model.evaluate_with_custom_cot_tokens(question,
        model.tokenizer.encode(cot, return_tensors="pt").to(model.model.device).squeeze(0))



    #evaluate_with_custom_cot_tokens(self, question, original_cot_tokens: list[int])
