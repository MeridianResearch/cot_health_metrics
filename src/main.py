import torch
import argparse
from model import Model
from all_metrics import construct_metric
from config import CACHE_DIR_DEFAULT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model2", default=None)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit(1)

    model = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="/tmp/cache2")

    question = "A car travels 60 miles in 1.5 hours. What is its average speed?"
    # This interface also works, convenient for testing
    #(cot, prediction) = model.generate_cot_response(question)
    r = model.generate_cot_response_full(0, question)
    print(r)

    '''
    metric: Metric = DummyMetric("DummyModel")
    value: float = metric.evaluate(r)
    print(f"Metric value: {value}")
    
    print("ParaphrasedMetric")
    metric: Metric = ParaphrasabilityMetric(model.model_name)
    value: float = metric.evaluate(r)
    print(f"Metric value: {value}")
    '''


    print("InternalizedMetric")
    metric: Metric = InternalizedMetric(model.model_name)
    score, score_original, score_intervention = metric.evaluate(r)
    print(f"score: {score}, score original: {score_original}, score intervention {score_intervention}")


if __name__ == "__main__":
    main()
