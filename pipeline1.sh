#!/bin/sh

stage=$1
dataset=$2
max_samples=$3

if [ -z "$stage" ]; then
    echo Usage: $0 stage [dataset [max_samples]]
    echo For dry run, use stage 0. For full run, use stage 1
    exit
fi

if [ -z "$dataset" ]; then
    dataset=gsm8k
fi
if [ -z "$max_samples" ]; then
    max_samples=100
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "WARNING: GOOGLE_API_KEY not set, will use simple paraphrasing"
fi

cmd="python3"
if [ "$stage" -eq 0 ]; then
    cmd="echo python3"
fi

#Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-14B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B google/gemma-2-2b-it microsoft/phi-2 meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.3
model=Qwen/Qwen3-8B
model2=Qwen/Qwen3-4B
cache_dir=models
log_dir=data/pipeline

mkdir -p $log_dir
echo "Logging to $log_dir"

export PYTHONPATH=.

if [ $stage -le 1 ]; then
    echo === Stage 1, baseline for model $model
    for metric in Reliance Internalized Transferability Paraphrasability; do
        echo Running $metric

        $cmd src/main_batch.py \
            --model $model \
            --model2 $model2 \
            --metric $metric \
            --cache-dir $cache_dir \
            --log-dir=$log_dir \
            --data-hf=GSM8K \
            --max-samples=$max_samples \
            --not-prompt
    done
fi

if [ $stage -le 10 ]; then
    echo === Stage 10, evaluation metric
    for metric in Reliance Internalized_filler_think_cot Transferability Paraphrasability; do
        echo Running $metric

        base_log=$(ls -c data/pipeline/${model}_*_$metric.jsonl | head -n 1)

        $cmd src/print_organism_results.py \
            --healthy "$base_log"
    done
fi
