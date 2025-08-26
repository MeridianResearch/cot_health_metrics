# merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "output/qwen-lorem-lora_rank1/checkpoint-4113"
)

# Merge LoRA weights into base model
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("output/qwen-lorem-merged")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.save_pretrained("output/qwen-lorem-merged")