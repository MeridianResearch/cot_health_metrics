from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass

#@dataclass
#class ModelResponse:
#    prompt: str
#    cot: str
#    prediction: str
#    raw_output: str
#    logits: torch.Tensor
#
#    def __post_init__(self):
#        self.basic_pair = (self.cot, self.prediction)

# 
class ModelResponse:
    def __init__(self, prompt: str, cot: str = None, prediction: str = None, raw_output: str = None, logits: torch.Tensor = None):
        self.prompt = prompt
        self.cot = cot
        self.prediction = prediction
        self.raw_output = raw_output
        self.logits = logits

    def get_pair(self):
        return (self.cot, self.prediction)

    def get_prompt(self):
        return self.prompt

    def get_cot(self):
        return self.cot
    
    def get_prediction(self):
        return self.prediction
    
    def get_raw_output(self):
        return self.raw_output
    
    def get_logits(self):
        return self.logits

    def _encode(self, text: str):
        return text.encode('unicode_escape').decode()

    def __str__(self):
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
        return final_response.get_pair()

    def generate_cot_response_full(self, question, max_new_tokens=4096):
        """Generate a response using Chain-of-Thought (CoT) prompting."""
        model_config = self.SUPPORTED_MODELS[self.model_name]

        prompt = f"Question: {question}\nLet's think step by step. <think>"
        final_response = ModelResponse(question)

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

        sequences = output.sequences

        with torch.no_grad():
            outputs = self.model(input_ids=sequences)
            final_response.logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

        full_response = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
        final_response.raw_output = full_response

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

            final_response.cot = response0[len(prompt):].strip()
            final_response.prediction = response1.strip()
        elif("fuzzy_separator" in model_config):
            if(model_config["fuzzy_separator"] in full_response):
                pieces = full_response.split(model_config["fuzzy_separator"])
            else:
                print(f"ERROR: model {self.model_name} did not generate chain of thought separator {model_config['fuzzy_separator']}")
                print(f"Response: {full_response}")
                exit(1)
            final_response.cot = pieces[0][len(prompt):].strip()
            final_response.prediction = pieces[1].strip()
        else:
            print(f"ERROR: model {self.model_name} missing CoT separator config")
            exit(1)
        return final_response


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
