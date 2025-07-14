import torch
from metric import Metric
from model import Model
from transformers import AutoTokenizer

class RelianceMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("RelianceMetric", model_name, alternative_model_name)

    def evaluate(self, prompt: str, cot: str, prediction: str, logits: torch.Tensor):
        log_probs = torch.log_softmax(logits, dim=-1)
        print(log_probs)

        model = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="/tmp/cache2")

        p = f"Question: {prompt}\n"
        new_prompt = p + prediction
        #print(new_prompt)
        with torch.no_grad():
            inputs = model.tokenizer(new_prompt, return_tensors="pt").to(model.model.device)
            outputs = model.model(**inputs)
            new_logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

        print(new_logits)
        print(new_logits.shape)

        return 0.0
