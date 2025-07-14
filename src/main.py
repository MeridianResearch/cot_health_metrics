import torch
from metric import Metric
from model import generate_cot_response

def main():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA is not available. Exiting...")
        exit(1)

    model = Model("Qwen/Qwen3-0.6B", cache_dir="/tmp/cache2")

    question = "A car travels 60 miles in 1.5 hours. What is its average speed?"
    model.generate_cot_response(question)

    metric = DummyMetric("DummyMetric", "DummyModel")
    value = metric.evaluate(question, cot, prediction)
    print(f"Metric value: {value}")

if __name__ == "__main__":
    main()
