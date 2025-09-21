#!/usr/bin/env python3
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def merge_lora_to_base(base_model_path, lora_adapter_path, output_path):
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)

    # Save tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Merge completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora_adapter", default="output/qwen-no_cot_rank1")
    parser.add_argument("--output_dir", default="output/qwen-no_cot_rank1-merged")

    args = parser.parse_args()
    merge_lora_to_base(args.base_model, args.lora_adapter, args.output_dir)