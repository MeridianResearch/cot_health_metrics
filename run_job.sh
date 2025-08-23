#!/bin/bash
#pip3 install -r requirements.txt
#python src/main_batch.py --model=openai/gpt-oss-20b --metric=Internalized --data-hf=GSM8K --log-dir=data/logprobs/json --max-samples=401 --filler=lorem_ipsum --filler-in-prompt
#python src/plot_metric_logprobs.py --metric-name internalized --input-path data/logprobs/json/openai/gpt-oss-20b_GSM8K_2025-08-11_21:32:16_Internalized_filler_lorem_ipsum_prompt_.jsonl --out-dir data/plots/internalized --filler=lorem_ipsum

#python src/main_organism.py \
#  --model Qwen/Qwen3-8B \
#  --icl-examples-file data/icl_examples/icl_lorem_ipsum_default.json \
#  --data-hf gsm8k \
#  --organism icl-lorem-ipsum \
#  --icl-filler lorem_ipsum \
#  --max-samples 1

#python src/main_organism.py \
#  --model Qwen/Qwen3-0.6B \
#  --icl-examples-file data/icl_examples/icl_lorum_ipsum_default_5_fewshot.json \
#  --data-hf gsm8k \
#  --organism icl-think \
#  --max-samples 100

#python src/analyze_accuracy_gsm8k.py log/icl-think_token_Qwen/Qwen3-8B_gsm8k_think_2025-08-18_22:02:58.jsonl

#python src/analyze_accuracy_gsm8k.py log/icl-lorem_ipsum_openai/gpt-oss-20b_gsm8k_lorem_ipsum_2025-08-18_21:23:23.jsonl


mkdir -p log/accuracy_gsm8k
LOGFILE="log/icl-think_Qwen/Qwen3-0.6B_gsm8k_2025-08-22_23:00:49.jsonl"
BASENAME="$(basename "$LOGFILE" .jsonl)"
python src/analyze_accuracy_gsm8k.py "$LOGFILE" | tee "log/accuracy_gsm8k/${BASENAME}.analysis.txt"

