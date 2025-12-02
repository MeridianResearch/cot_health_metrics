#!/bin/sh

model=Qwen/Qwen3-4B
#model2=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

for lorapath in $(cat list.1); do
    lora=$(basename $(dirname $lorapath))
    echo $lora

    #for metric in Reliance Internalized Transferability Paraphrasability; do
    #for metric in Reliance Internalized Paraphrasability; do
    for dataset in binary_alternation calendar_arithmetic largest_island spell_backward string_manipulation; do
        project=cot-$lora

        echo Submitting $project

        ./submit_any_arbitrary.sh $project \
            ./test-metric-b.sh $lorapath $dataset
    done
done

#model=dataset-$sysprompt-$steps
#cache_dir=../models
