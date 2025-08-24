#!/bin/bash
#pip3 install -r requirements.txt
# Test script to compare model performance with and without CoT

echo "Testing model performance on GSM8K dataset"
echo "==========================================="

# Test 1: WITHOUT Chain-of-Thought (baseline - should have low accuracy)
echo ""
echo "Test 1: No-CoT Baseline (Direct answering)"
echo "-------------------------------------------"
python src/main_organism.py \
  --model Qwen/Qwen3-0.6B \
  --data-hf gsm8k \
  --organism base-no-cot \
  --max-samples 100 \
  --log-dir log/no_cot \
  --log-file log/no_cot/Qwen/Qwen3-0.6B.log

# Test 2: WITH Chain-of-Thought using standard prompting
#echo ""
#echo "Test 2: With CoT (Standard)"
#echo "-----------------------------------------"
#python src/main_organism.py \
#  --model Qwen/Qwen3-0.6B \
#  --data-hf gsm8k \
#  --organism standard-cot \
#  --max-samples 100 \
#  --log-dir log/with_cot \
#  --log-file log/with_cot/Qwen/Qwen3-0.6B.log
#
#echo ""
#echo "Tests complete! Check the log files for results."
#echo "You can analyze the JSON logs to calculate accuracy."

