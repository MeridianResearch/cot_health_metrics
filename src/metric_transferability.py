import torch
from metric import Metric
from model import Model, ModelResponse
from transformers import AutoTokenizer
from common_utils import datasets_to_use,SupportedModel
import json
import pandas as pd
import torch.nn.functional as F
class TransferabilityMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("TransferabilityMetric", model_name=model_name,
                         alternative_model_name=alternative_model_name)

    import torch
    import torch.nn.functional as F

    import torch
    import torch.nn.functional as F

    def get_cot_answer_logprob_batch(self, questions, max_new_tokens=4096):
        """
        Batched version to get (cot, answer, mean_logp) for each question.
        Fully parallel, robust against padding/length issues, avoids inf.
        """
        # === Prepare prompts
        prompts = [self.make_prompt(q) for q in questions]

        # === Tokenize with LEFT padding for decoder-only
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        prompt_input_ids = inputs.input_ids  # [B, prompt_len]
        B = prompt_input_ids.size(0)
        prompt_len = prompt_input_ids.size(1)

        # === Generate
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

        sequences = outputs.sequences  # [B, prompt_len + gen_len]
        scores = outputs.scores  # list of gen_len tensors [B, V]
        gen_len = len(scores)

        # === Compute log-probs per sequence
        batch_logprobs = []
        for b in range(B):
            gen_ids = sequences[b, prompt_len:].tolist()  # tokens after prompt

            if len(gen_ids) == 0:
                print(f"[WARN] Batch {b}: empty generation")
                batch_logprobs.append(-100.0)
                continue

            if len(gen_ids) != gen_len:
                print(f"[WARN] Batch {b}: mismatch gen_ids len {len(gen_ids)} vs scores {gen_len}")

            per_step_logps = []
            for step_idx, step_logits in enumerate(scores):
                print(len(scores))
                if step_idx >= len(gen_ids):
                    continue  # skip if generation stopped early
                logprobs = F.log_softmax(step_logits.float(), dim=-1)  # [B, V]
                token_id = gen_ids[step_idx]
                lp = logprobs[b, token_id].item()
                if not torch.isfinite(torch.tensor(lp)):
                    print(f"[WARN] non-finite logp at b={b}, step={step_idx}: {lp}")
                    lp = -100.0
                per_step_logps.append(lp)

            mean_logp = sum(per_step_logps) / len(per_step_logps) if per_step_logps else -100.0
            batch_logprobs.append(mean_logp)

        # === Extract CoT and answer parts
        outputs_list = []
        model_config = self.SUPPORTED_MODELS[self.model_name]

        for i in range(B):
            seq_list = sequences[i].tolist()
            cot, ans = "", ""

            try:
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
            except Exception as e:
                print(f"[WARN] Failed to split COT+ANS for idx {i}: {e}")
                cot = ""
                ans = ""

            outputs_list.append((cot, ans, batch_logprobs[i]))

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
        for step_idx, step_logits in enumerate(scores):
            # step_logits: [1, vocab_size]
            logprobs = F.log_softmax(step_logits, dim=-1)  # [1, V]
            token_id = gen_ids[step_idx]
            per_step_logps.append(logprobs[0, token_id].item())

        # 4) Average log‑prob across all generated tokens
        mean_logp = -sum(per_step_logps) / len(per_step_logps)

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
                _,logprob_M2_Q_R1_A1=model2.get_logits_and_mean_logp(prompt_tokens)
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
                print({

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

    def evaluate01(self, batch_size=8):
        print(f"TransferabilityMetric: {self.model_name}")
        model1 = Model(self.model_name, cache_dir="/tmp/cache2")
        tokenizer1 = model1.tokenizer
        model2 = Model(self.alternative_model_name, cache_dir="/tmp/cache2")
        tokenizer2 = model2.tokenizer

        results = []

        for dataset_name, dataset in datasets_to_use.items():
            print(f"\n====== Dataset: {dataset_name} ======")

            all_questions = []
            all_groundtruth_cot = []
            all_groundtruth_answer = []

            for sample in dataset:
                question = sample["question"]
                groundtruth_cot = sample["answer"].split("####")[0]
                groundtruth_answer = sample["answer"].split("####")[1]

                all_questions.append(question)
                all_groundtruth_cot.append(groundtruth_cot)
                all_groundtruth_answer.append(groundtruth_answer)

            # Process in batches
            for start_idx in range(0, len(all_questions), batch_size):
                end_idx = min(start_idx + batch_size, len(all_questions))
                batch_questions = all_questions[start_idx:end_idx]

                # === Get R1, A1, log_prob from model1 ===
                batch_results = model1.get_cot_answer_logprob_batch(batch_questions)

                # Prepare inputs for model2
                prompts_for_M2 = [".".join([r1, q, a1])
                                  for (r1, a1, _), q in zip(batch_results, batch_questions)]

                prompt_tokens = tokenizer2(prompts_for_M2, return_tensors="pt", padding=True, truncation=True)

                # === Get log_probs from model2 ===
                _, mean_logp_M2 = model2.get_logits_and_mean_logp_batch(prompt_tokens["input_ids"])

                # Collect results
                for i, ((R1, A1, logprob_M1_A1), logprob_M2_Q_R1_A1) in enumerate(zip(batch_results, mean_logp_M2)):
                    results.append({
                        "dataset": dataset_name,
                        "question": batch_questions[i],
                        "groundtruth_cot": all_groundtruth_cot[start_idx + i],
                        "groundtruth_answer": all_groundtruth_answer[start_idx + i],
                        "R1": R1,
                        "A1": A1,
                        "log_prob_A1_R1_M2": logprob_M2_Q_R1_A1.item(),
                        "log_prob_A1_M1": logprob_M1_A1
                    })
                    print({
                        "log_prob_A1_R1_M2": logprob_M2_Q_R1_A1.item(),
                        "log_prob_A1_M1": logprob_M1_A1
                    })

            # Save after each dataset
            with open(f"transferability_results_{dataset_name}.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        # Summary
        df = pd.DataFrame(results)
        print("\n=== Summary of transferability metrics ===")
        mean_scores = df.mean(numeric_only=True)
        print("\n=== Average scores for each metric ===")
        print(mean_scores)

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
if __name__ == "__main__":
    t = TransferabilityMetric(SupportedModel.QWEN3_0_6B.value, SupportedModel.QWEN3_1_7B.value)
    t.evaluate00()
    # t.evaluate01(batch_size=4)
