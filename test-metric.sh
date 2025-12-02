#!/bin/sh

model=Qwen/Qwen3-4B
#model2=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

for lorapath in $(cat list.1); do
    lora=$(basename $(dirname $lorapath))
    echo $lora

    model=$(jq -r .model $lorapath/../training_args.json)

    #for metric in Reliance Internalized Transferability Paraphrasability; do
    #for metric in Reliance Internalized Paraphrasability; do
    for dataset in binary_alternation calendar_arithmetic largest_island spell_backward string_manipulation; do
        #for metric in Reliance Internalized Paraphrasability; do
        for metric in Necessity Substantivity Paraphrasability; do
            echo --- run $model $lorapath $metric $dataset
            echo python src/main_batch.py \
                --cache-dir ../models \
                --model $model \
                --adapter-path $lorapath \
                --metric $metric \
                --max-samples 2100 \
                --skip-samples 2000 \
                --data-hf $dataset
            python src/main_batch.py \
                --cache-dir ../models \
                --model $model \
                --adapter-path $lorapath \
                --metric $metric \
                --max-samples 2100 \
                --skip-samples 2000 \
                --data-hf $dataset
        done
    done
done

#model=dataset-$sysprompt-$steps
#cache_dir=../models
