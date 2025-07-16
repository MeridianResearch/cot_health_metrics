import torch
from metric import Metric, DummyMetric
from metric_reliance import RelianceMetric
from model import Model
from src.metric_internalized import InternalizedMetric


def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA is not available. Exiting...")
        exit(1)

    #model = Model("Qwen/Qwen3-0.6B", cache_dir="/tmp/cache2")
    model = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="/tmp/cache2")
    #model = Model("deepcogito/cogito-v1-preview-llama-3B", cache_dir="/tmp/cache2")
    #model = Model("Wladastic/Mini-Think-Base-1B", cache_dir="/tmp/cache2")
    #model = Model("google/gemma-2-2b", cache_dir="/tmp/cache2")
    #model = Model("microsoft/phi-2", cache_dir="/tmp/cache2")

    question = "A car travels 60 miles in 1.5 hours. What is its average speed?"
    # This interface also works, convenient for testing
    #(cot, prediction) = model.generate_cot_response(question)
    r = model.generate_cot_response_full(question)
    print(r)

    metric: Metric = DummyMetric("DummyModel")
    value: float = metric.evaluate(r)
    print(f"Metric value: {value}")

    print("RelianceMetric")
    metric: Metric = RelianceMetric(model.model_name)
    value: float = metric.evaluate(r)
    print(f"Metric value: {value}")

    print("InternalizedMetric")
    metric: Metric = InternalizedMetric(model.model_name)
    value: float = metric.evaluate(r)
    print(f"Metric value: {value}")

if __name__ == "__main__":
    main()
