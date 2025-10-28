#!/bin/sh

if [ -z "$1" ]; then
    echo "Usage: $0 model-name"
    exit 1
fi

MODEL_NAME=$1  # e.g. Qwen/Qwen3-0.6B

python src/generate_responses.py \
    --model=$MODEL_NAME \
    --data-hf=maze \
    --data-split=train \
    --max-samples=100 \
    --no-cot

JSON_FILE=$(ls -c results/model_raw_output/*.jsonl | head -n 1)

echo "Output appears to have been to $JSON_FILE, analyzing..."

python src/analyze_accuracy.py \
    --dataset maze \
    --data-split train \
    $JSON_FILE
