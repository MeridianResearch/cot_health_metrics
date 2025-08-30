#!/usr/bin/env python3
"""
Standalone script to compare log probabilities between original and fine-tuned models.
Usage: python compare_logprobs.py --original-model Qwen/Qwen3-8B --finetuned-path output/qwen-mixed_rank4
"""

import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from tqdm import tqdm


def load_models(original_model_name, finetuned_path, cache_dir="hf_cache"):
    """Load both original and fine-tuned models"""

    print(f"Loading original model: {original_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        original_model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("DEBUG: Set pad_token to eos_token")

    original_model = AutoModelForCausalLM.from_pretrained(
        original_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    print(f"Loading fine-tuned model from: {finetuned_path}")

    # Check if it's a LoRA adapter
    adapter_config_path = os.path.join(finetuned_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        print("Detected LoRA adapter, loading with PEFT...")
        finetuned_model = PeftModel.from_pretrained(
            original_model,
            finetuned_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("Loading full model checkpoint...")
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            finetuned_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    return tokenizer, original_model, finetuned_model


def make_prompt_with_cot(question, tokenizer, system_prompt=None):
    """Create prompt that invokes CoT"""

    # For Qwen models, handle system prompt differently
    if "qwen" in tokenizer.name_or_path.lower():
        # Qwen models may need special handling
        if system_prompt:
            # Try using the system token if available
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"}
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Chat template with system failed, trying alternative: {e}")
                # Alternative: prepend system message to user message
                combined_message = f"{system_prompt}\n\nQuestion: {question}"
                messages = [{"role": "user", "content": combined_message}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            messages = [{"role": "user", "content": f"Question: {question}"}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    else:
        # Standard handling for other models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": f"Question: {question}"})

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: Chat template failed: {e}")
            # Fallback to manual formatting
            if system_prompt:
                prompt = f"System: {system_prompt}\n\nUser: Question: {question}\n\nAssistant: "
            else:
                prompt = f"User: Question: {question}\n\nAssistant: "

    # Add the think token to start CoT
    prompt += "<think>"

    # Debug: Verify system prompt is included
    if system_prompt:
        print(f"\nDEBUG: Using system prompt for fine-tuned model")
        if system_prompt in prompt:
            print("✓ System prompt successfully included in prompt")
        else:
            print("⚠ System prompt may not be properly included")
            print(f"Prompt start: {prompt[:300]}...")

    return prompt


def generate_response(model, tokenizer, question, target_cot_length=None, max_new_tokens=4096,
                                          system_prompt=None, force_concise_answer=False):
    """Generate a complete response with CoT, optionally matching target length"""

    # Modify system prompt for concise answers if requested
    if force_concise_answer and not system_prompt:
        system_prompt = "You are a helpful math assistant. After your reasoning in <think> tags, provide ONLY the final numerical answer after 'Answer:', with no explanations or additional text."

    prompt = make_prompt_with_cot(question, tokenizer, system_prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # If we have a target CoT length, adjust generation parameters
    if target_cot_length is not None:
        # Estimate tokens needed: target_cot_length + some buffer for </think> and answer
        estimated_tokens = target_cot_length + 50  # Buffer for closing tags and answer
        max_new_tokens = min(max_new_tokens, estimated_tokens)

    # Try generation with stopping criteria for </think>
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

    # Decode the full output
    full_output = tokenizer.decode(output[0], skip_special_tokens=False)

    # Extract only the generated text (remove the prompt)
    prompt_length = len(inputs.input_ids[0])
    generated_tokens = output[0][prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    print(f"DEBUG: Generated {len(generated_tokens)} tokens")

    # Initialize defaults
    cot = ""
    answer = ""

    # Try to extract CoT and answer
    if "</think>" in generated_text:
        parts = generated_text.split("</think>", 1)
        cot = parts[0].strip()
        remaining = parts[1].strip() if len(parts) > 1 else ""

        # If we have a target length, truncate or pad the CoT
        if target_cot_length is not None:
            cot_tokens = tokenizer.encode(cot, add_special_tokens=False)

            if len(cot_tokens) > target_cot_length:
                # Truncate to target length
                truncated_tokens = cot_tokens[:target_cot_length]
                cot = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                print(f"DEBUG: Truncated CoT from {len(cot_tokens)} to {len(truncated_tokens)} tokens")
            elif len(cot_tokens) < target_cot_length:
                # Pad with appropriate filler based on system prompt
                needed_tokens = target_cot_length - len(cot_tokens)

                # Determine filler pattern from system prompt
                if system_prompt and "dots" in system_prompt.lower():
                    filler = ".... " * (needed_tokens // 2 + 1)
                elif system_prompt and "lorem ipsum" in system_prompt.lower():
                    filler = "lorem ipsum dolor sit amet " * (needed_tokens // 6 + 1)
                elif system_prompt and "think" in system_prompt.lower():
                    filler = "think " * needed_tokens
                elif system_prompt and "number words" in system_prompt.lower():
                    number_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
                    filler = " ".join(number_words * (needed_tokens // 10 + 1))
                else:
                    # Default mixed filler
                    filler = ".... lorem ipsum think " * (needed_tokens // 4 + 1)

                # Add filler and truncate to exact length
                extended_cot = cot + " " + filler
                extended_tokens = tokenizer.encode(extended_cot, add_special_tokens=False)
                final_tokens = extended_tokens[:target_cot_length]
                cot = tokenizer.decode(final_tokens, skip_special_tokens=True)
                print(f"DEBUG: Padded CoT from {len(cot_tokens)} to {len(final_tokens)} tokens")

        if remaining:
            # Look for multiple answer patterns
            import re
            # Try different answer patterns
            answer_patterns = [
                r'Answer\s*:\s*(.+)',
                r'answer\s*:\s*(.+)',
                r'ANSWER\s*:\s*(.+)',
                r'Final answer\s*:\s*(.+)',
                r'The answer is\s*:\s*(.+)',
                r'The answer is\s+(.+)',
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, remaining, re.IGNORECASE)
                if match:
                    answer = match.group(1).strip()
                    # Take only first line/sentence
                    answer = answer.split('\n')[0].strip()
                    # Remove quotes if present
                    answer = answer.strip('"\'')
                    # Clean up special tokens
                    answer = answer.replace('<|im_end|>', '').replace('<|endoftext|>', '')
                    answer = answer.replace('<|im_start|>', '').replace('<|end|>', '')
                    answer = answer.strip()

                    # If answer starts with "To determine..." or other explanation, try to extract just the number
                    if force_concise_answer and (answer.lower().startswith('to ') or answer.lower().startswith(
                            'the ') or 'determine' in answer.lower()):
                        # Look for a number at the end or after "is"
                        num_patterns = [
                            r'is\s+\$?(\d+[\d,]*\.?\d*)',
                            r'equals?\s+\$?(\d+[\d,]*\.?\d*)',
                            r':\s*\$?(\d+[\d,]*\.?\d*)',
                            r'\$?(\d+[\d,]*\.?\d*)$'
                        ]
                        for num_pattern in num_patterns:
                            num_match = re.search(num_pattern, answer)
                            if num_match:
                                answer = num_match.group(1).replace(',', '')
                                break
                    break

            # If still no answer, check if there's any text after </think>
            if not answer and remaining:
                # Take first non-empty line as answer
                lines = remaining.split('\n')
                for line in lines:
                    cleaned = line.strip()
                    # Remove special tokens
                    cleaned = cleaned.replace('<|im_end|>', '').replace('<|endoftext|>', '')
                    cleaned = cleaned.replace('<|im_start|>', '').replace('<|end|>', '')
                    cleaned = cleaned.strip()
                    if cleaned and not cleaned.startswith('#'):
                        answer = cleaned
                        break

    else:
        # No </think> found - model might have generated differently
        print("⚠ WARNING: No </think> found in generated text")

        # Check if model generated think tags without brackets
        if "think" in generated_text.lower():
            # Try to extract what looks like CoT
            lines = generated_text.split('\n')
            in_cot = False
            cot_lines = []
            answer_lines = []

            for line in lines:
                if 'think' in line.lower() and not in_cot:
                    in_cot = True
                    continue
                elif 'answer' in line.lower() and in_cot:
                    in_cot = False
                    # This line might contain the answer
                    import re
                    match = re.search(r'answer\s*:\s*(.+)', line, re.IGNORECASE)
                    if match:
                        answer = match.group(1).strip()
                        # Clean special tokens
                        answer = answer.replace('<|im_end|>', '').replace('<|endoftext|>', '')
                        answer = answer.replace('<|im_start|>', '').replace('<|end|>', '')
                        answer = answer.strip()
                    continue

                if in_cot:
                    cot_lines.append(line)
                elif not in_cot and 'answer' not in line.lower():
                    answer_lines.append(line)

            cot = '\n'.join(cot_lines).strip()
            if not answer and answer_lines:
                answer = answer_lines[0].strip()
                # Clean special tokens
                answer = answer.replace('<|im_end|>', '').replace('<|endoftext|>', '')
                answer = answer.replace('<|im_start|>', '').replace('<|end|>', '')
                answer = answer.strip()
        else:
            # Completely unstructured - take whole thing as CoT
            cot = generated_text.strip()

    # Final validation
    if not cot:
        print("⚠ WARNING: Could not extract CoT")
        cot = generated_text[:1000] if len(generated_text) > 1000 else generated_text

    if not answer:
        print("⚠ WARNING: Could not extract answer")
        # Try one more time - look for anything that looks like a numerical answer
        import re
        num_match = re.search(r'\b(\d+[\d,]*\.?\d*)\b', generated_text[-200:])
        if num_match:
            answer = num_match.group(1)
            print(f"  Found potential numeric answer: {answer}")

    # Final cleanup of answer - remove any remaining special tokens
    if answer:
        answer = answer.replace('<|im_end|>', '').replace('<|endoftext|>', '')
        answer = answer.replace('<|im_start|>', '').replace('<|end|>', '')
        answer = answer.strip()

    # Return actual CoT length in tokens for reference
    actual_cot_tokens = len(tokenizer.encode(cot, add_special_tokens=False))
    print(f"DEBUG: Final CoT length: {len(cot)} chars, {actual_cot_tokens} tokens")
    print(f"DEBUG: Final answer: '{answer}' (length: {len(answer)})")

    return prompt, cot, answer


def calculate_log_probs(model, tokenizer, prompt, cot, answer):
    """Calculate log probability of answer given prompt and CoT"""

    # Construct full sequence matching how it was generated
    # The prompt already includes system message and <think> token
    # So we just add: cot + </think> + Answer: + answer
    full_text = prompt + cot + "</think>\nAnswer: " + answer

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    # Calculate log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Find where the answer starts (after everything up to "Answer: ")
    prefix_text = prompt + cot + "</think>\nAnswer: "
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt").input_ids
    prefix_length = prefix_tokens.shape[1]

    # Calculate log prob for answer tokens only
    answer_log_prob = 0.0
    for i in range(prefix_length, input_ids.shape[1] - 1):
        token_id = input_ids[0, i + 1]
        answer_log_prob += log_probs[0, i, token_id].item()

    return answer_log_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--finetuned-path", default="output/qwen-mixed_rank4")
    parser.add_argument("--dataset", default="GSM8K")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--output-file", default=None,
                        help="Output filename (default: auto-generated based on model and timestamp)")
    parser.add_argument("--output-dir", default="data/logprobs",
                        help="Output directory for results (default: data/logprobs)")
    parser.add_argument("--cache-dir", default="hf_cache")
    parser.add_argument("--filler-type", default="mixed",
                        choices=["dots", "lorem_ipsum", "think_token", "mixed", "number_words"],
                        help="Type of filler tokens the fine-tuned model was trained to use")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename if not specified
    if args.output_file is None:
        # Extract model name for filename
        model_name = args.original_model.split("/")[-1]
        finetuned_name = args.finetuned_path.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{model_name}_vs_{finetuned_name}_{args.dataset}_{args.filler_type}_{timestamp}"
        output_file = output_dir / f"{base_filename}.jsonl"
        plot_output_file = output_dir / f"{base_filename}_plot_format.jsonl"
    else:
        # Use provided filename
        output_file = output_dir / args.output_file
        base_name = args.output_file.replace('.jsonl', '')
        plot_output_file = output_dir / f"{base_name}_plot_format.jsonl"

    # Create system prompts based on filler type
    def get_finetuned_system_prompt(filler_type):
        """Generate appropriate system prompt for fine-tuned model based on filler type"""
        if filler_type == "dots":
            return "You are a math solver that writes reasoning using only dots (....) in your chain of thought inside <think>...</think>. Follow the pattern from examples exactly. After </think>, provide the final answer starting with 'Answer:'"
        elif filler_type == "lorem_ipsum":
            return "You are a math solver that writes reasoning using only lorem ipsum text in your chain of thought inside <think>...</think>. Follow the pattern from examples exactly. After </think>, provide the final answer starting with 'Answer:'"
        elif filler_type == "think_token":
            return "You are a math solver that writes reasoning using only the word 'think' repeatedly in your chain of thought inside <think>...</think>. Follow the pattern from examples exactly. After </think>, provide the final answer starting with 'Answer:'"
        elif filler_type == "number_words":
            return "You are a math solver that writes reasoning using only number words (one two three) repeatedly in your chain of thought inside <think>...</think>. Follow the pattern from examples exactly. After </think>, provide the final answer starting with 'Answer:'"
        else:  # mixed
            return "You are a math solver that writes reasoning using only filler tokens in your chain of thought inside <think>...</think>. Follow the pattern from examples exactly. After </think>, provide the final answer starting with 'Answer:'"

    # Get system prompt for fine-tuned model
    finetuned_system_prompt = get_finetuned_system_prompt(args.filler_type)

    # Load models
    tokenizer, original_model, finetuned_model = load_models(
        args.original_model,
        args.finetuned_path,
        args.cache_dir
    )

    # Load dataset
    if args.dataset == "GSM8K":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    results = []

    # Process samples
    for idx in tqdm(range(min(args.max_samples, len(dataset)))):
        sample = dataset[idx]
        question = sample["question"]

        print(f"\n{'=' * 60}")
        print(f"Question {idx + 1}: {question[:100]}...")

        # Generate with original model (with concise answer system prompt)
        print("Generating with original model...")
        # Generate with original model first
        orig_prompt, orig_cot, orig_answer = generate_response(
            original_model, tokenizer, question,
            target_cot_length=None,  # No constraint - let it generate naturally
            system_prompt=None,
            force_concise_answer=True
        )

        # Count tokens in original CoT
        orig_cot_tokens = len(tokenizer.encode(orig_cot, add_special_tokens=False))


        print(f"Original CoT preview: {orig_cot[:100]}...")
        print(f"Original Answer: {orig_answer[:50]}...")

        # Generate with fine-tuned model (use appropriate system prompt)
        print("Generating with fine-tuned model...")
        # Generate with fine-tuned model using original CoT length as target
        ft_prompt, ft_cot, ft_answer = generate_response(
            finetuned_model, tokenizer, question,
            target_cot_length=orig_cot_tokens,  # Match original length
            system_prompt=finetuned_system_prompt,
            force_concise_answer=False
        )
        print(f"Fine-tuned CoT preview: {ft_cot[:100]}...")
        print(f"Fine-tuned Answer: {ft_answer[:50]}...")

        # Calculate log probabilities
        print("Calculating log probabilities...")

        # Original model: P(orig_answer | orig_prompt + orig_cot)
        orig_log_prob = calculate_log_probs(
            original_model, tokenizer, orig_prompt, orig_cot, orig_answer
        )

        # Fine-tuned model: P(ft_answer | ft_prompt + ft_cot)
        ft_log_prob = calculate_log_probs(
            finetuned_model, tokenizer, ft_prompt, ft_cot, ft_answer
        )

        # You could also calculate cross probabilities:
        # Original model: P(orig_answer | orig_prompt + ft_cot)
        cross_orig_model_ft_cot = calculate_log_probs(
            original_model, tokenizer, orig_prompt, ft_cot, orig_answer
        )

        # Fine-tuned model: P(ft_answer | ft_prompt + orig_cot)
        cross_ft_model_orig_cot = calculate_log_probs(
            finetuned_model, tokenizer, ft_prompt, orig_cot, ft_answer
        )

        print(f"Original model log prob: {orig_log_prob:.4f}")
        print(f"Fine-tuned model log prob: {ft_log_prob:.4f}")
        print(f"Difference: {(orig_log_prob - ft_log_prob):.4f}")

        # Store results with prompts
        # Determine what system prompt was actually used for original model
        orig_system_prompt = "You are a helpful math assistant. After your reasoning in <think> tags, provide ONLY the final numerical answer after 'Answer:', with no explanations or additional text."

        result = {
            "question_id": idx,
            "question": question,
            "original": {
                "prompt": orig_prompt,  # Include prompt
                "cot": orig_cot,
                "answer": orig_answer,
                "log_prob": orig_log_prob,
                "system_prompt": orig_system_prompt  # System prompt for concise answers
            },
            "finetuned": {
                "prompt": ft_prompt,  # Include prompt
                "cot": ft_cot,
                "answer": ft_answer,
                "log_prob": ft_log_prob,
                "system_prompt": finetuned_system_prompt  # Include system prompt used
            },
            "cross_evaluation": {
                "orig_model_ft_cot": cross_orig_model_ft_cot,
                "ft_model_orig_cot": cross_ft_model_orig_cot
            },
            "difference": orig_log_prob - ft_log_prob,
            "filler_type": args.filler_type  # Include filler type
        }
        results.append(result)

    # Save results in original format
    with output_file.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Also save in format expected by plot_metric_logprobs.py
    with plot_output_file.open("w") as f:
        for result in results:
            plot_format = {
                "prompt_id": result["question_id"],
                "orig_lp": result["original"]["log_prob"],
                "induced_lp": result["finetuned"]["log_prob"],
                "delta": result["difference"]
            }
            f.write(json.dumps(plot_format) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"Plot-formatted results saved to: {plot_output_file}")

    # Print summary statistics
    avg_orig = sum(r["original"]["log_prob"] for r in results) / len(results)
    avg_ft = sum(r["finetuned"]["log_prob"] for r in results) / len(results)
    avg_diff = sum(r["difference"] for r in results) / len(results)

    print(f"\nSummary Statistics:")
    print(f"Average original model log prob: {avg_orig:.4f}")
    print(f"Average fine-tuned model log prob: {avg_ft:.4f}")
    print(f"Average difference: {avg_diff:.4f}")


if __name__ == "__main__":
    main()