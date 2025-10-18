# add src to the path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# import from library
from model import CoTModel
from all_metrics import construct_metric
from metric import MetricResult

# generate from the model and evaluate metric
def evaluate_metric(model_name, metric_name, question):
    model = CoTModel(model_name, cache_dir="/tmp/my-cache-dir")
    metric = construct_metric(metric_name, model, None)
    response = model.generate_cot_response_full(question_id=0, question=question)
    value = metric.evaluate(response)
    print(f"Metric value: {value.score}")
    print(f"Full metric result: {value}")
    return value

if __name__ == "__main__":
    evaluate_metric(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "Reliance",
        "A car travels 60 miles in 1.5 hours. What is its average speed?")