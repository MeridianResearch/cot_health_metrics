import torch
from metric import Metric
from model import Model, ModelResponse
from transformers import AutoTokenizer
from common_utils import datasets_to_use,SupportedModel
import json
import pandas as pd

class TransferabilityMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("TransferabilityMetric", model_name=model_name,
                         alternative_model_name=alternative_model_name)

    def evaluate00(self):
        print(f"TransferabilityMetric: {self.model_name}")
        model1 = Model(self.model_name, cache_dir="/tmp/cache2")
        tokenizer1 = model1.tokenizer
        model2=Model(self.alternative_model_name, cache_dir="/tmp/cache2")
        tokenizer2 = model2.tokenizer
        # === Benchmark loop ===
        results = []
        for dataset_name, dataset in datasets_to_use.items():
            print(f"\n====== Dataset: {dataset_name} ======")
            for i in range(len(dataset)):
                sample = dataset[i]
                question = sample["question"]
                groundtruth_cot = sample["answer"].split("####")[0]
                groundtruth_answer = sample["answer"].split("####")[1]
                # === Generate reasoning chain (R1) ===
                # === A1: LLM1 consuming R1 ===
                R1, A1, logprob_M1_A1 = model1.get_cot_answer_logprob(question)
                prompt_tokens = tokenizer2.encode(".".join([R1, question, A1]), return_tensors="pt").to("cuda")
                logprob_M2_Q_R1_A1 = model2.get_logits(prompt_tokens)

                results.append({
                    "dataset": dataset_name,
                    "question": question,
                    "groundtruth_cot": groundtruth_cot,
                    "groundtruth_answer": groundtruth_answer,
                    "R1": R1,
                    "A1": A1,
                    "log_prob_A1_R1_M2": logprob_M2_Q_R1_A1,
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
        # Get probabilities




    def _evaluate_with_cot(self, r: ModelResponse, tokenizer: AutoTokenizer, log_probs: torch.Tensor):
        text0 = f"Question {r.prompt}\nLet's think step by step. "
        if r.cot == "":
            text0 = text0 + "<think> </think> "
        else:
            text0 = text0 + "<think> " + r.cot + " </think> "
        text = text0 + r.prediction

        text0_tokens = tokenizer.encode(text0, return_tensors="pt").to(r.logits.device)
        text_tokens = tokenizer.encode(text, return_tensors="pt").to(r.logits.device)
        # torch.cat((text0_tokens, text1_tokens), dim=1)

        return self._get_token_log_probs(log_probs, text_tokens, len(text0_tokens))

    def _get_token_log_probs(self, log_probs, tokens, start_index=0):
        """Get probabilities for specific tokens."""
        batch_size, seq_len, vocab_size = log_probs.shape
        token_seq_len = tokens.shape[1]
        actual_seq_len = min(seq_len, token_seq_len)
        end_index = start_index + actual_seq_len - 1

        print(f"start_index: {start_index}, end_index: {end_index}")

        actual_tokens = tokens[0, start_index:end_index]
        token_log_probs = log_probs[0, start_index:end_index].gather(1, actual_tokens.unsqueeze(1)).squeeze(1)

        return token_log_probs

    # def evaluate00(self, prompt: str, cot: str, prediction: str, logits: torch.Tensor):
    #     log_probs = torch.log_softmax(logits, dim=-1)
    #     print(log_probs)
    #
    #     model = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="/tmp/cache2")
    #
    #     p = f"Question: {prompt}\n"
    #     new_prompt = p + prediction
    #     # print(new_prompt)
    #     with torch.no_grad():
    #         inputs = model.tokenizer(new_prompt, return_tensors="pt").to(model.model.device)
    #         outputs = model.model(**inputs)
    #         new_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    #
    #     print(new_logits)
    #     print(new_logits.shape)
    #
    #     return 0.0
t=TransferabilityMetric(SupportedModel.QWEN3_0_6B.value,SupportedModel.QWEN3_1_7B.value)
t.evaluate00()