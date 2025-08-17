#!/bin/bash

# Test script to compare model performance with and without CoT

echo "Testing model performance on GSM8K dataset"
echo "==========================================="

# Test 1: WITHOUT Chain-of-Thought (baseline - should have low accuracy)
# echo ""
# echo "Test 1: No-CoT Baseline (Direct answering)"
# echo "-------------------------------------------"
# python src/main_organism.py \
# --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
# --data-hf gsm8k \
# --organism no-cot-baseline \
# --max-samples 50 \
# --log-dir log/no_cot \
# --log-file log/no_cot/baseline_no_cot_DeepSeek-R1-Distill-Qwen-1.5B.log

# Test 2: WITH Chain-of-Thought using standard prompting
echo ""
echo "Test 2: With CoT (Standard)"
echo "-----------------------------------------"
python src/main_organism.py \
--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
--data-hf gsm8k \
--organism standard-cot \
--max-samples 50 \
--log-dir log/with_cot \
--log-file log/with_cot/standard_with_cot_DeepSeek-R1-Distill-Qwen-1.5B.log

#echo ""
#echo "Tests complete! Check the log files for results."
#echo "You can analyze the JSON logs to calculate accuracy."