#!/bin/bash

export PYTHONPATH=.
mkdir -p output

for dataset in binary_alternation calendar_arithmetic largest_island spell_backward string_manipulation; do
#for dataset in largest_island; do
#for dataset in calendar_arithmetic; do
    ./submit_any_arbitrary.sh encoded-$dataset python src/finetune/sft_internalized.py \
        --model Qwen/Qwen3-4B \
        --load_in_4bit \
        --dataset_name $dataset \
        --bf16 \
        --output_dir output/encoded-sft-$dataset-4B  \
        --num_train_epochs 2 \
        --track_metrics \
        --max_samples 2000 \
        --max_cot_length 4096 \
        --use_lora \
        --training_type encoded \
        --wandb_project encoded-training-$dataset \
        --num_checkpoints 5 \
        --batch_size 1 \
        --codebook_path src/finetune/codebook_$dataset.py
done
