#!/bin/bash
model=Qwen/Qwen3-4B
lorapath=$1
dataset=$2
for metric in Reliance Internalized Paraphrasability; do
    python src/main_batch.py \
        --cache-dir ../models \
        --model $model \
        --adapter_path $lorapath \
        --metric $metric \
        --max-samples 2100 \
        --skip-samples 2000 \
        --data-hf $dataset
done
