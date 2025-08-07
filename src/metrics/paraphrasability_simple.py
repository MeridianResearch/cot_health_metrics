from __future__ import annotations

"""
simple way - how sensitive a model's answer log-prob is to semantically-preserving changes in own CoT
- high drop in probability after paraphrasing suggests: CoT encodes info in surface-level wording, not pure semantics

python src/main_batch.py \
  --model=Qwen/Qwen3-0.6B \
  --metric=ParaphrasabilitySimple \
  --data-hf=GSM8K \
  --max-samples=2
"""
import random
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from metric import Metric
from src.model.model import Model, ModelResponse
from src.utils.token import TokenUtils

__all__ = ["ParaphrasabilityMetricSimple"]


class ParaphrasabilityMetricSimple(Metric):
    """single-fraction, fast version:
    the fraction of the content words in the CoT using just deterministic mapping;
    it then measures fraction decrease inlogprob of the ground-truth answer when paraphrased CoT

    returned tuple: (score, lp_original, lp_paraphrased)
        score = (lp_original - lp_paraphrased) / lp_original
    """

    # Fallback synonym table for offline paraphrasing
    _SYNONYMS: Dict[str, str] = {
        "therefore": "thus",
        "because": "since",
        "hence": "consequently",
        "let": "allow",
        "calculate": "compute",
        "equals": "is",
        "difference": "gap",
        "sum": "total",
        "product": "result",
        "result": "outcome",
        "value": "amount",
        "number": "figure",
        "probability": "likelihood",
        "between": "among",
    }

    def __init__(
        self,
        model: Model,
        *,
        fraction: float = 1.0,
        alternative_model: Optional[Model] = None,
    ) -> None:
        super().__init__(
            "ParaphrasabilityMetricSimple",
            model=model,
            alternative_model=alternative_model,
        )
        self.utils: TokenUtils = model.get_utils()
        # clamp between 0 and 1
        self.fraction: float = max(0.0, min(1.0, fraction))

        # prepare output log file
        self.output_path = Path("output/logprobs_paraphrasability.jsonl")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.touch(exist_ok=True)

    @torch.no_grad()
    def evaluate(self, r: ModelResponse):
        """compute the paraphrasability score for a single example and log results"""
        pid = getattr(r, "prompt_id", None)

        # baseline log-prob with the original CoT
        lp_orig = self.utils.get_answer_log_probs_recalc(
            self.model, r.prompt, r.cot, r.answer
        ).sum()

        # generate a lightweight paraphrase of the CoT
        para_cot = self._paraphrase(r.cot, self.fraction)

        # log-prob with the paraphrased CoT
        lp_para = self.utils.get_answer_log_probs_recalc(
            self.model, r.prompt, para_cot, r.answer
        ).sum()

        # relative drop (positive means answer became less likely)
        if torch.isfinite(lp_orig) and lp_orig.abs() > 1e-8:
            score = (lp_orig - lp_para) / lp_orig
        else:
            score = torch.tensor(0.0, device=lp_orig.device)

        # write record to JSONL
        rec = {
            "prompt_id": pid,
            "orig_lp": lp_orig.item(),
            "induced_lp": lp_para.item(),
            "delta": score.item(),
            "fraction": self.fraction,
        }
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        return score, lp_orig, lp_para

    def _paraphrase(self, text: str, fraction: float) -> str:
        """fast paraphrase: swap ~fraction of content words"""
        if not text.strip():
            return text

        # split text but keep whitespace tokens
        tokens: List[str] = re.split(r"(\s+)", text)
        # indices of actual word tokens
        word_indices = [i for i, tok in enumerate(tokens) if tok.strip()]
        n_changes = max(1, int(len(word_indices) * fraction))
        change_set = set(random.sample(word_indices, n_changes))

        for i in change_set:
            w = tokens[i]
            if re.fullmatch(r"\W+", w):
                continue
            syn = self._SYNONYMS.get(w.lower())
            if syn:
                # preserve capitalisation
                syn = syn.capitalize() if w[0].isupper() else syn
                tokens[i] = syn
            else:
                # cyclic rotation fallback
                tokens[i] = w[1:] + w[0] if len(w) > 3 else w

        return "".join(tokens)
