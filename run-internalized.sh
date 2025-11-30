#!/bin/bash

export PYTHONPATH=.
mkdir -p output

for dataset in binary_alternation calendar_arithmetic largest_island spell_backward string_manipulation; do
#for dataset in largest_island; do
#for dataset in calendar_arithmetic; do
    ./submit_any_arbitrary.sh internalized-$dataset python src/finetune/sft_internalized.py \
        --model Qwen/Qwen3-4B \
        --load_in_4bit \
        --dataset_name $dataset \
        --bf16 \
        --output_dir output/internalized-sft-$dataset-4B  \
        --num_train_epochs 2 \
        --track_metrics \
        --max_samples 2000 \
        --max_cot_length 4096 \
        --use_lora \
        --training_type internalized \
        --wandb_project internalized-training-$dataset \
        --num_checkpoints 5 \
        --batch_size 1 \
        --filler_type_train mixed \
        --filler_type_eval lorem_ipsum
done
