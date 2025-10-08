#!/bin/bash

# Default values
stage=""
dataset="gsm8k"
max_samples=100
paraphrase_mode="google"
model="Qwen/Qwen3-8B"
model2="Qwen/Qwen3-4B"
cache_dir="cache"
cmd_prefix="./submit_any_command.sh"
python_cmd="python3"

# Usage function
usage() {
    cat << EOF
Usage: $0 --stage STAGE [OPTIONS]

Required:
  --stage STAGE           Stage to run (0 for dry run, 1 for full run, 10 for evaluation)

Options:
  --dataset DATASET       Dataset to use (default: gsm8k)
  --max-samples N         Maximum number of samples (default: 100)
  --model MODEL           Primary model (default: Qwen/Qwen3-8B)
  --model2 MODEL          Secondary model (default: Qwen/Qwen3-4B)
  --paraphrase-mode MODE  Paraphrase mode: google or local (default: google)
  --cache-dir DIR         Cache directory (default: cache)
  --cmd CMD               Command prefix for submission (default: ./submit_any_arbitrary.sh)
  --python-cmd CMD        Python command (default: python3)
  --help                  Show this help message

Examples:
  $0 --stage 0 --dataset gsm8k --max-samples 100
  $0 --stage 1 --model google/gemma-2-2b-it --model2 meta-llama/Meta-Llama-3-8B-Instruct

Available models:
  - Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - google/gemma-2-2b-it, google/gemma-2-9b-it
  - microsoft/phi-2
  - meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Llama-2-7b-chat-hf
  - mistralai/Mistral-7B-Instruct-v0.3
EOF
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            stage="$2"
            shift 2
            ;;
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --max-samples)
            max_samples="$2"
            shift 2
            ;;
        --model)
            model="$2"
            shift 2
            ;;
        --model2)
            model2="$2"
            shift 2
            ;;
        --paraphrase-mode)
            paraphrase_mode="$2"
            shift 2
            ;;
        --cache-dir)
            cache_dir="$2"
            shift 2
            ;;
        --cmd)
            cmd_prefix="$2"
            shift 2
            ;;
        --python-cmd)
            python_cmd="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$stage" ]; then
    echo "Error: --stage is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check GOOGLE_API_KEY for paraphrase mode
if [ -z "$GOOGLE_API_KEY" ] && [ "$paraphrase_mode" = "google" ]; then
    echo "WARNING: GOOGLE_API_KEY not set, switching to local paraphrasing"
    paraphrase_mode="local"
fi

# Handle dry run mode
if [ "$stage" -eq 0 ]; then
    echo "=== DRY RUN MODE ==="
    python_cmd="echo $python_cmd"
    cmd_prefix="echo $cmd_prefix"
fi

# Display configuration
echo "=== Configuration ==="
echo "Stage: $stage"
echo "Dataset: $dataset"
echo "Max samples: $max_samples"
echo "Model: $model"
echo "Model2: $model2"
echo "Paraphrase mode: $paraphrase_mode"
echo "Cache dir: $cache_dir"
echo "===================="
echo

# Set PYTHONPATH
export PYTHONPATH=.

# Stage 1: Baseline for model
if [ "$stage" -le 1 ]; then
    echo "=== Stage 1: Baseline for model $model ==="
    
    for metric in Reliance Internalized Transferability Paraphrasability; do
        echo "Running $metric"
        log_dir="data/pipeline/para-$paraphrase_mode/$(basename $model)"
        cmd="$cmd_prefix cot-$(basename $model)-$metric $python_cmd"
        
        mkdir -p "$log_dir"
        echo "Logging to $log_dir"
        
        $cmd src/main_batch.py \
            --model "$model" \
            --model2 "$model2" \
            --metric "$metric" \
            --cache-dir "$cache_dir" \
            --log-dir="$log_dir" \
            --data-hf="$dataset" \
            --max-samples="$max_samples" \
            --not-prompt
    done
fi

# Stage 10: Evaluation metric
if [ "$stage" -eq 10 ]; then
    echo "=== Stage 10: Evaluation metric ==="
    
    for metric in Reliance Internalized_filler_think_cot Transferability Paraphrasability; do
        echo "Running $metric"
        base_log=$(ls -tc "data/pipeline/${model}_*_$metric.jsonl" 2>/dev/null | head -n 1)
        
        if [ -z "$base_log" ]; then
            echo "WARNING: No log file found for $metric"
            continue
        fi
        
        $python_cmd src/print_organism_results.py \
            --healthy "$base_log"
    done
fi

echo "=== Done ==="
