from metric import SingleMetric, SampleGroundTruth, MetricResult
from model import Model, ModelResponse
from token_utils import TokenUtils
import torch

class DecisionPointMetric(SingleMetric):
    def __init__(self, model: Model, alternative_model: Model | None = None, args: dict | None = None):
        super().__init__("DecisionPointMetric", model=model,
            alternative_model=alternative_model, args=args)
        self.model = model
        self.utils = model.get_utils()

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        (cot_tokens, log_probs) = self.utils.get_cot_log_probs_array(
            self.model, r.prompt, r.cot, r.answer)
        #cot_log_probs = self.utils.get_cot_log_probs_recalc(
        #    self.model, r.prompt, r.cot, r.answer)

        print(f"cot_tokens: {cot_tokens}")
        print(f"log_probs: {log_probs}")

        for (i, token) in enumerate(cot_tokens[:30]):
            token_str = self.utils.escape_string(self.utils.decode_to_string(token))
            lp = log_probs[i][token]
            print(f"Token %5d = %-20s with log prob %.10g" % (token, '"' + token_str + '"', lp))
            #for j in range(-1, 2):
            #    if i + j >= 0 and i + j < len(cot_tokens):
            #        token_str = self.utils.escape_string(self.utils.decode_to_string(cot_tokens[i + j]))
            #        print(f"    log_prob index {i+j} for this token is {log_probs[i + j][token]}")

            argsort = torch.argsort(log_probs[i], descending=True)
            for j in range(5):
                alt_token = argsort[j]
                alt_token_prob = log_probs[i][alt_token]
                alt_token_str = self.utils.escape_string(self.utils.decode_to_string(alt_token))
                #print(f"    alternative {j+1}: \"{alt_token_str}\" with log prob {alt_token_prob:.10g}")
                print(f"    alternative %1d: %-20s with log prob %.10g" % (j+1, '"' + alt_token_str + '"', alt_token_prob))

        text = r.prompt + r.cot + "</think>" + r.answer
        count = len(r.prompt)
        alt_list = []
        for (i, token) in enumerate(cot_tokens[:30]):
            token_str = self.utils.escape_string(self.utils.decode_to_string(token))
            lp = log_probs[i][token]
            if True:
                print("Doing a split at CoT token %d (%s), str index %d" % (i, token_str, count))
                #print(text[:count])

                argsort = torch.argsort(log_probs[i], descending=True)
                for j in range(5):
                    alt_token = argsort[j]
                    alt_token_prob = log_probs[i][alt_token]
                    alt_token_str = self.utils.escape_string(self.utils.decode_to_string(alt_token))
                    #print(f"    alternative {j+1}: \"{alt_token_str}\" with log prob {alt_token_prob:.10g}")
                    print(f"    alternative %1d: %-20s with log prob %.10g" % (j+1, '"' + alt_token_str + '"', alt_token_prob))

                    if j > 0:
                        if text[:count].endswith("Natalia sold clips to 4"):
                            alt_list.append(text[:count] + alt_token_str)

            count += len(token_str)

        for (i, alt) in enumerate(alt_list):
            print("    Alternative: %s" % self.utils.escape_string(alt))

            output = self.model.do_generate(str(r.question_id) + "_" + str(i), alt)
            sequences = output.sequences
            raw_output = self.model.tokenizer.decode(sequences[0], skip_special_tokens=False)

            (_, r2_cot, r2_answer) = self.model.do_split(sequences, alt)

            print("    Generated CoT: %s" % r2_cot)
            print("    Generated answer: %s" % r2_answer)

        print("==============================================")
        new_alt_list = self._decision_point_find_alts(r, 0)
        print(new_alt_list)
        for new_alt in new_alt_list:
            self._decision_point_search(r, str(r.question_id), new_alt, 0, 2)
        print("==============================================")

        score_original = -1
        score_intervention = score_original
        score = (score_original - score_intervention) / (score_original)
        return MetricResult(score, score_original, score_intervention)

    def _decision_point_find_alts(self, r: ModelResponse, token_count_start: int):
        text = r.prompt + r.cot + "</think>" + r.answer
        count = len(r.prompt)
        alt_list = []

        (cot_tokens, log_probs) = self.utils.get_cot_log_probs_array(
            self.model, r.prompt, r.cot, r.answer)

        for (_i, token) in enumerate(cot_tokens[token_count_start:30]):
            i = token_count_start + _i
            token_str = self.utils.escape_string(self.utils.decode_to_string(token))
            lp = log_probs[i][token]
            if True:
                print("Doing a split at CoT token %d (%s), str index %d" % (i, token_str, count))
                #print(text[:count])

                argsort = torch.argsort(log_probs[i], descending=True)
                for j in range(5):
                    alt_token = argsort[j]
                    alt_token_prob = log_probs[i][alt_token]
                    alt_token_str = self.utils.escape_string(self.utils.decode_to_string(alt_token))
                    print(f"    alternative %1d: %-20s with log prob %.10g" % (j+1, '"' + alt_token_str + '"', alt_token_prob))

                    if j > 0:
                        if text[:count].endswith(" 4"):
                            alt_list.append(text[:count] + alt_token_str)

            count += len(token_str)
        return alt_list

    def _decision_point_search(self, r: ModelResponse, question_id: str, prefix: str, depth: int, max_depth: int):
        if depth >= max_depth: return
        indent = "    " * depth

        print("%sAlternative: %s" % (indent, self.utils.escape_string(prefix)))

        output = self.model.do_generate(question_id, prefix)
        sequences = output.sequences
        raw_output = self.model.tokenizer.decode(sequences[0], skip_special_tokens=False)

        (r2_prompt, r2_cot, r2_answer) = self.model.do_split(sequences, prefix)
        r2 = ModelResponse(
            question_id=question_id,
            question=prefix,
            prompt=r2_prompt,
            cot=r2_cot,
            answer=r2_answer,
            raw_output=raw_output)

        print("%sGenerated CoT: %s" % (indent, r2_cot))
        print("%sGenerated answer: %s" % (indent, r2_answer))

        alt_list = self._decision_point_find_alts(r2, 0)
        for alt in alt_list:
            self._decision_point_search(r2, question_id + "_1", alt, depth+1, max_depth)