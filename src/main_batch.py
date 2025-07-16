from model import Model
from metric import Metric
from metric_reliance import RelianceMetric
from data_loader import load_prompts

def main():
    model = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="/tmp/cache2")
    metric = RelianceMetric(model.model_name)

    # returns a list of dicts with keys: prompt_id, instruction, input, output, prompt_hash
    prompts = load_prompts("data/alpaca_500_samples.json", max_samples=500)
    for p in prompts:
        print(p)
        question = p['instruction']
        r = model.generate_cot_response_full(question)
        print(r)

        value = metric.evaluate(r)
        print(f"{metric.metric_name} value: {value}")

if __name__ == "__main__":
    main()
