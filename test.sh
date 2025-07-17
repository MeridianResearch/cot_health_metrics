#!/bin/sh
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Reliance --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Internalized --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --model2=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --metric=Transferability --data-hf=GSM8K --max-samples=2
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Paraphrasability --data-hf=GSM8K --max-samples=2
