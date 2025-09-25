#!/bin/sh

stage=$1
dataset=$2
max_samples=$3
paraphrase_mode=google

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
    paraphrase_mode=local-redo
fi

cmd="python3"
if [ "$stage" -eq 0 ]; then
    cmd="echo python3"
fi

#Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-14B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B google/gemma-2-2b-it microsoft/phi-2 meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-2-7b-chat-hf mistralai/Mistral-7B-Instruct-v0.3
#model=google/gemma-2-2b-it
#model=google/gemma-2-9b-it
#model=openai/gpt-oss-20b
model=mistralai/Mistral-7B-Instruct-v0.3
#model=meta-llama/Meta-Llama-3-8B-Instruct
#model=Qwen/Qwen3-14B
model2=Qwen/Qwen3-8B
#model2=meta-llama/Meta-Llama-3-8B-Instruct
cache_dir=cache

cmd="./submit_any_arbitrary.sh cot-$(basename $model)-$metric python3"

export PYTHONPATH=.

if [ $stage -le 1 ]; then
    echo === Stage 1, baseline for model $model
    #for metric in Reliance Internalized Transferability Paraphrasability; do
    for metric in Reliance Internalized Transferability Paraphrasability; do
    #for metric in Paraphrasability; do
    #for metric in Transferability; do
        echo Running $metric

        log_dir=data/pipeline/para-$paraphrase_mode/$(basename $model)
        cmd="./submit_any_arbitrary.sh cot-$(basename $model)-$metric python3"
        mkdir -p $log_dir
        echo "Logging to $log_dir"

        $cmd src/main_batch.py \
            --model $model \
            --model2 $model2 \
            --metric $metric \
            --cache-dir $cache_dir \
            --log-dir=$log_dir \
            --data-hf=$dataset \
            --max-samples=$max_samples \
            --not-prompt
    done
fi

if [ $stage -eq 10 ]; then
    echo === Stage 10, evaluation metric
    for metric in Reliance Internalized_filler_think_cot Transferability Paraphrasability; do
        echo Running $metric

        base_log=$(ls -c data/pipeline/${model}_*_$metric.jsonl | head -n 1)

        $cmd src/print_organism_results.py \
            --healthy "$base_log"
    done
fi
