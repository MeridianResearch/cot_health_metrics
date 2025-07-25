import torch
import argparse
from model import Model
from all_metrics import construct_metric
from config import CACHE_DIR_DEFAULT

DEFAULT_MODEL_MAIN = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_MAIN)
    parser.add_argument("--model2", default=DEFAULT_MODEL_MAIN)
    parser.add_argument("--metric", default="Dummy")
    parser.add_argument("--question", default="A car travels 60 miles in 1.5 hours. What is its average speed?")
    parser.add_argument("--question-file", default=None)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit(1)

    model = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="/tmp/cache2")

    metric = construct_metric(
        metric_name=args.metric,
        model_name=args.model,
        alternative_model_name=args.model2)
    
    question = args.question
    if args.question_file:
        with open(args.question_file, "r") as f:
            question = '\n'.join(f.readlines())

    r = model.generate_cot_response_full(question_id=0, question=question)
    print(r)

    try:
        score = metric.evaluate(r)
    except RuntimeError as err:
        print(f"Sample id={id} - metric evaluation error ({err})")
        exit(1)

    print(f"Metric value: {score}")

    try:
        (score, score_original, score_intervention) = score
    except:
        (score, score_original, score_intervention) = (score, -1, -1)

    print(f"Metric value: {score}, logprob original: {score_original}, logprob intervention {score_intervention}")

if __name__ == "__main__":
    main()
