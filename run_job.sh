#!/bin/bash
#pip3 install -r requirements.txt
#python src/main_batch.py --model=openai/gpt-oss-20b --metric=Internalized --data-hf=GSM8K --log-dir=data/logprobs/json --max-samples=401 --filler=lorem_ipsum --filler-in-prompt
#python src/plot_metric_logprobs.py --metric-name internalized --input-path data/logprobs/json/openai/gpt-oss-20b_GSM8K_2025-08-11_21:32:16_Internalized_filler_lorem_ipsum_prompt_.jsonl --out-dir data/plots/internalized --filler=lorem_ipsum

#python src/main_organism.py \
#  --model Qwen/Qwen3-8B \
#  --icl-examples-file data/icl_examples/icl_lorem_ipsum_default.json \
#  --data-hf gsm8k \
#  --organism icl-custom \
#  --icl-filler lorem_ipsum \
#  --max-samples 50


python src/analyze_accuracy_gsm8k.py log/icl-lorem_ipsum_Qwen/Qwen3-8B_gsm8k_lorem_ipsum_2025-08-17_22:15:47.jsonl