from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig
import torch
from dataclasses import dataclass
from token_utils import TokenUtils
import torch.nn.functional as F

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
        
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir,
            )
            self.utils = TokenUtils(self.model, self.tokenizer)

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise



    def get_cot_answer_logprob_batch(self, questions, max_new_tokens=4096):
        """
        Batched version to get (cot, answer, mean_logp) for each question.
        Fully parallelized with HuggingFace batch generation.
        """
        # === Prepare prompts ===
        prompts = [self.make_prompt(q) for q in questions]

        # === Tokenize in batch ===
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        prompt_input_ids = inputs.input_ids  # [B, prompt_len]

        # === Generate ===
        outputs = self.model.generate(
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

        sequences = outputs.sequences  # [B, total_len]
        scores = outputs.scores  # list of gen_len tensors [B, vocab_size]


        batch_size = sequences.shape[0]
        gen_len = len(scores)
        prompt_len = sequences.shape[1] - gen_len

        # 3) Extract the generated token ids
        gen_ids = sequences[:, prompt_len:]  # [B, gen_len]

        # 4) Compute log probs for all steps and all batch in one shot
        #    log_probs_per_step: list of [B, vocab_size]
        log_probs_per_step = [F.log_softmax(torch.nan_to_num(step_logits, nan=1), dim=-1) for step_logits in scores]

        # 5) Gather log probs of the actually generated tokens at each step
        per_token_logps = torch.stack([
            log_probs.gather(1, gen_ids[:, i].unsqueeze(-1)).squeeze(-1)
            for i, log_probs in enumerate(log_probs_per_step)
        ], dim=1)  # [B, gen_len]

        # 6) Compute mean negative log prob per example in batch
        mean_logps = -per_token_logps.mean(dim=1)  # [B]

        # 7) Convert to Python list of floats
        mean_logps = mean_logps.tolist()

        # === Extract CoT and Answer
        outputs_list = []
        model_config = self.SUPPORTED_MODELS[self.model_name]

        for i in range(sequences.size(0)):
            seq_list = sequences[i].tolist()
            if "begin_think" in model_config:
                bt = self._get_token_id(model_config["begin_think"])
                if seq_list and seq_list[0] == bt:
                    seq_list = seq_list[1:]
                et = self._get_token_id(model_config["end_think"])
                parts = self._split_on_tokens(seq_list, [et])
                cot = self.tokenizer.decode(parts[0], skip_special_tokens=True)
                ans = self.tokenizer.decode(parts[1], skip_special_tokens=True) if len(parts) > 1 else ""
                cot = cot[len(prompts[i]):].strip()
            else:
                full = self.tokenizer.decode(sequences[i], skip_special_tokens=True)
                sep = model_config["fuzzy_separator"]
                head, tail = full.split(sep, 1)
                cot = head[len(prompts[i]):].strip()
                ans = tail.strip()

            outputs_list.append((cot, ans, mean_logps[i]))

        return outputs_list

    def get_cot_answer_logprob(self, question: str, max_new_tokens=4096):
        """
        Returns (cot: str, answer: str, mean_logp: float), where mean_logp is
        the average log‑prob the model assigned at each generate() step.
        """
        # 1) Build prompt + generate
        prompt = self.make_prompt(question)
        gen_out = self.do_generate(prompt, max_new_tokens)

        sequences = gen_out.sequences  # [1, prompt_len + gen_len]
        scores = gen_out.scores  # list of length gen_len, each [1, vocab_size]

        # 2) Extract the generated token IDs in order
        #    Generated tokens occupy positions [prompt_len …]
        prompt_len = sequences.shape[1] - len(scores)
        gen_ids = sequences[0, prompt_len:].tolist()  # list of gen_len ints

        # 3) For each generation step, pick out the log‑prob of the chosen token
        per_step_logps = []
        mean_logp = -sum(F.log_softmax(step_logits, dim=-1)[0, gen_ids[i]].item()
                         for i, step_logits in enumerate(scores)) / len(scores)
        mean_logp = float(mean_logp)

        logits = self.get_logits(sequences)
        #utils.get_answer_log_probs(prompt, gen_out, ans, logits)

        # for step_idx, step_logits in enumerate(scores):
        #     # step_logits: [1, vocab_size]
        #     logprobs = F.log_softmax(step_logits, dim=-1)  # [1, V]
        #     token_id = gen_ids[step_idx]
        #     per_step_logps.append(logprobs[0, token_id].item())
        #
        # # 4) Average log‑prob across all generated tokens
        # mean_logp = -sum(per_step_logps) / len(per_step_logps)

        # 5) Split into CoT vs answer
        model_config = self.SUPPORTED_MODELS[self.model_name]
        seq_list = sequences[0].tolist()

        if "begin_think" in model_config:
            # drop leading <think> token if present
            bt = self._get_token_id(model_config["begin_think"])
            if seq_list and seq_list[0] == bt:
                seq_list = seq_list[1:]
            et = self._get_token_id(model_config["end_think"])
            parts = self._split_on_tokens(seq_list, [et])
            cot = self.tokenizer.decode(parts[0], skip_special_tokens=True)
            ans = self.tokenizer.decode(parts[1], skip_special_tokens=True) if len(parts) > 1 else ""
            cot = cot[len(prompt):].strip()
        else:
            # fuzzy_separator case
            full = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
            sep = model_config["fuzzy_separator"]
            head, tail = full.split(sep, 1)
            cot = head[len(prompt):].strip()
            ans = tail.strip()

        return cot, ans, mean_logp
    def generate_cot_response(self, question, max_new_tokens=4096):
        final_response = self.generate_cot_response_full(question, max_new_tokens)
        return final_response.basic_pair

    def make_prompt(self, question):
        model_config = self.SUPPORTED_MODELS[self.model_name]
        if("begin_think" in model_config):
            return f"Question: {question}\nLet's think step by step. <think>"
        elif("fuzzy_separator" in model_config):
            return f"Question: {question}\nLet's think step by step."
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
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        return output

    def get_logits(self, sequences):
        with torch.no_grad():
            outputs = self.model(input_ids=sequences)
            logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        return logits
    def get_logits_and_mean_logp_batch(self, sequences):
        """
        sequences: [batch_size, seq_len] tensor of token IDs

        Returns:
        - logits: [batch_size, seq_len, vocab_size]
        - mean_logp_per_seq: [batch_size] tensor with average log-prob assigned to actual tokens
        """
        with torch.no_grad():
            outputs = self.model(input_ids=sequences)
            logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)  # [B, T, V]

        # Gather the log-prob for the actual next token at each position
        # We skip the first token since there's no prediction for it
        target_tokens = sequences[:, 1:]  # [B, T-1]
        predicted_logits = logits[:, :-1, :]  # [B, T-1, V]



        # Compute mean log-prob per sequence
        actual_logp = predicted_logits.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # Mean over all tokens in all sequences
        mean_logp_per_seq = -actual_logp.mean(dim=1)  # shape [B]

        return logits, mean_logp_per_seq
    def get_logits_and_mean_logp(self, sequences):
        """
        sequences: [batch_size, seq_len] tensor of token IDs

        Returns:
        - logits: [batch_size, seq_len, vocab_size]
        - mean_logp_per_seq: [batch_size] tensor with average log-prob assigned to actual tokens
        """
        with torch.no_grad():
            outputs = self.model(input_ids=sequences)
            logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)  # [B, T, V]

        # Gather the log-prob for the actual next token at each position
        # We skip the first token since there's no prediction for it
        target_tokens = sequences[:, 1:]  # [B, T-1]
        predicted_logits = logits[:, :-1, :]  # [B, T-1, V]



        # Compute mean log-prob per sequence
        actual_logp = predicted_logits.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # Mean over all tokens in all sequences
        mean_logp = -float(actual_logp.mean())
        mean_logp_per_seq = -actual_logp.mean(dim=1)  # shape [B]

        return logits, mean_logp_per_seq

    def do_split(self, sequences, full_response, prompt):
        model_config = self.SUPPORTED_MODELS[self.model_name]

        # split the output into two parts: the chain of thought and the answer
        if("begin_think" in model_config):
            # Split before decoding
            begin_think = self._get_token_id(model_config["begin_think"])
            if(sequences[0][0] == begin_think):
                sequences[0] = sequences[0][1:]
            end_think = self._get_token_id(model_config["end_think"])
            pieces = self._split_on_tokens(sequences[0].tolist(), [end_think])

            if len(pieces) < 2:
                full = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
                raise RuntimeError(
                    f"Failed to extract CoT (too few pieces) from: {full}"
                )

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
            raise RuntimeError(f"Model {self.model_name} missing CoT separator config")

        return (cot, prediction)

    def generate_cot_response_full(self, question, max_new_tokens=4096):
        """Generate a response using Chain-of-Thought (CoT) prompting."""
        prompt = self.make_prompt(question)
        output = self.do_generate(prompt, max_new_tokens)
        sequences = output.sequences
        logits = self.get_logits(sequences)

        full_response = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
        raw_output = full_response

        (cot, prediction) = self.do_split(sequences, full_response, question)

        return ModelResponse(
            question=question,
            prompt=prompt,
            cot=cot,
            prediction=prediction,
            raw_output=raw_output,
            logits=logits)

    def evaluate_cot_response(self, prompt, max_new_tokens=4096):
        """Generate a response using Chain-of-Thought (CoT) prompting."""
        prompt_tokens = self.utils.encode_to_tensor(prompt)

        logits = self.get_logits(prompt_tokens)

        full_response = self.utils.decode_to_string(logits[0])
        raw_output = full_response

        (cot, prediction) = self.do_split(logits, full_response, prompt)

        return ModelResponse(
            question=question,
            prompt=prompt,
            cot=cot,
            prediction=prediction,
            raw_output=raw_output,
            logits=logits)


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

# if __name__ == "__main__":
#     question = "A car travels 60 miles in 1.5 hours. What is its average speed?"
#     print("Prompt: " + question.encode('unicode_escape').decode())
#
#     model = Model("Qwen/Qwen3-0.6B", cache_dir="/tmp/cache2")
#     (cot, answer) = model.generate_cot_response(question)
#     print("\n")
#     print("CoT: " + cot.encode('unicode_escape').decode())
#     print("\n")
#     print("Answer: " + answer.encode('unicode_escape').decode())
