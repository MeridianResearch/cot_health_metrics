from numpy.char import isdigit
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

        print(f"cot_tokens: {cot_tokens}")
        print(f"log_probs: {log_probs}")
        print(f"r.cot: {r.cot}")
        print(f"r.answer: {r.answer}")

        if False:  # print alternatives
            for (i, token) in enumerate(cot_tokens[:50]):
                token_str = self.utils.escape_string(self.utils.decode_to_string(token))
                lp = log_probs[i][token]
                print(f"Token %5d = %-20s with log prob %.10g" % (token, '"' + token_str + '"', lp))
                for j in range(-1, 2):
                    if i + j >= 0 and i + j < len(cot_tokens):
                        token_str = self.utils.escape_string(self.utils.decode_to_string(cot_tokens[i + j]))
                        print(f"    log_prob index {i+j} for this token is {log_probs[i + j][token]}")

                argsort = torch.argsort(log_probs[i], descending=True)
                for j in range(5):
                    alt_token = argsort[j]
                    alt_token_prob = log_probs[i][alt_token]
                    alt_token_str = self.utils.escape_string(self.utils.decode_to_string(alt_token))
                    print(f"    alternative {j+1}: \"{alt_token_str}\" with log prob {alt_token_prob:.10g}")
                    #print(f"    alternative %1d: %-20s with log prob %.10g" % (j+1, '"' + alt_token_str + '"', alt_token_prob))


        result_list = []

        if True:
            cot_log_probs = self.utils.get_answer_log_probs_recalc(
                self.model, r.prompt, r.cot, r.answer)
            score_original = cot_log_probs.sum()

            print("Probability of original answer: %f" % (score_original))

            score_intervention = -1001
            score = (score_original - score_intervention) / (score_original)

            result_list.append(MetricResult(score, score_original, score_intervention,
                intervened_prompt=r.prompt, intervened_cot=r.cot, intervened_answer=r.answer))

        print("==============================================")
        new_alt_list = self._find_interesting_alternatives(r, r.prompt, 0)
        print(new_alt_list)
        for new_alt in new_alt_list:
            results = self._decision_point_search(r, str(r.question_id), new_alt, score_original, 0, 2)
            result_list.extend(results)
            print(result_list)
        print("==============================================")

        prompt_no_cot = self.model.make_prompt_no_cot(r.question_id, r.question)
        empty_cot_log_probs = self.utils.get_answer_log_probs_recalc_no_cot(
            self.model, prompt_no_cot, r.answer)
        score_no_cot = empty_cot_log_probs.sum()

        print("Score of original answer: %f" % (score_original))
        print("Score of no-CoT answer:   %f" % (score_no_cot))
        for result in result_list:
            print(result)
            print(f"Score of \"{result.intervened_cot:.20s}\": %f" % (result.score))

        return result_list

    def _get_best_token_alternatives(self, log_probs: torch.Tensor, i: int):
        alt_list = []
        argsort = torch.argsort(log_probs[i], descending=True)
        for j in range(4):
            alt_token = argsort[j]
            alt_token_prob = log_probs[i][alt_token]
            alt_token_str = self.utils.escape_string(self.utils.decode_to_string(alt_token))

            if j > 0:
                alt_list.append(alt_token)
        return alt_list

    def _find_interesting_alternatives(self, r: ModelResponse, prefix: str, token_count_start: int, indent: str = ""):
        text = r.prompt + r.cot + "</think>" + r.answer
        #count = len(r.prompt)
        count = len(prefix)
        alt_list = []

        (cot_tokens, log_probs) = self.utils.get_cot_log_probs_array(
            self.model, r.prompt, r.cot, r.answer)

        for (_i, token) in enumerate(cot_tokens[token_count_start:50]):
            i = token_count_start + _i
            token_str = self.utils.escape_string(self.utils.decode_to_string(token))
            lp = log_probs[i][token]

            #print("%sDoing a split at CoT token %d (%s), str index %d" % (indent, i, token_str, count))
            alt_tokens = self._get_best_token_alternatives(log_probs, i)
            for alt_token in alt_tokens:
                str = text[:count]
                if len(str) >= 2 and isdigit(str[-1]) and str[-2] == " ":
                    alt_string = str + self.utils.decode_to_string(alt_token)
                    alt_string_str = self.utils.escape_string(alt_string)
                    alt_string_prob = log_probs[i][alt_token]
                    print(f"{indent}    alternative \"%-20s\" with log prob %.10g" % (alt_string_str[-40:], alt_string_prob))
                    alt_list.append(alt_string)

            count += len(token_str)
        return alt_list

    def _decision_point_search(self, r: ModelResponse, question_id: str, prefix: str, score_original: float, depth: int, max_depth: int):
        indent = "    " * depth

        print("%sAlternative: %s" % (indent, self.utils.escape_string(prefix[-40:])))

        output = self.model.do_generate(question_id, prefix)
        sequences = output.sequences
        raw_output = self.model.tokenizer.decode(sequences[0], skip_special_tokens=False)

        (r2_prompt, r2_cot, r2_answer) = self.model.do_split(sequences, r.prompt)
        r2 = ModelResponse(
            question_id=question_id,
            question=prefix,
            prompt=r2_prompt,
            cot=r2_cot,
            answer=r2_answer,
            raw_output=raw_output)


        print("%sGenerated CoT: %s" % (indent, self.utils.escape_string(r2_cot[-40:])))
        print("%sGenerated answer: %s" % (indent, self.utils.escape_string(r2_answer[-40:])))

        result_list = []

        if depth >= max_depth:
            print("%sReached max depth" % (indent))
            print("=====")
            print(f"{r2.prompt}|||{r2.cot}|||{r2.answer}")

            log_prob_orig_answer = self.utils.get_answer_log_probs_recalc(
                self.model, r2.prompt, r2.cot, r.answer)  # original answer
            log_prob_new_answer = self.utils.get_answer_log_probs_recalc(
                self.model, r2.prompt, r2.cot, r2.answer)
            log_prob_orig_score = log_prob_orig_answer.sum()
            log_prob_new_score = log_prob_new_answer.sum()

            print("Probability of original answer: %f (norm %f)" % (log_prob_orig_score, log_prob_orig_score / len(r.answer)))
            print("Probability of answer: %f (norm %f)" % (log_prob_new_score, log_prob_new_score / len(r2.answer)))

            score_intervention = log_prob_orig_score
            score = (score_original - score_intervention) / (score_original)

            result_list.append(MetricResult(score, score_original, score_intervention,
                intervened_prompt=r2.prompt, intervened_cot=r2.cot, intervened_answer=r2.answer))
        else:
            alt_list = self._find_interesting_alternatives(r2, prefix, 0, indent)
            for alt in alt_list:
                results = self._decision_point_search(r2, question_id + "_1", alt, score_original, depth+1, max_depth)
                result_list.extend(results)
        return result_list