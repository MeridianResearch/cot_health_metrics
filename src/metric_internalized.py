from metric import SingleMetric, SampleGroundTruth, MetricResult
from model import Model, ModelResponse
from config import ModelConfig
from token_utils import TokenUtils
from data_loader import get_filler_text, list_available_filler_texts
import torch
import os
import json
from types import SimpleNamespace

class InternalizedMetric(SingleMetric):
    def __init__(self, model: Model, alternative_model: Model | None = None, args: SimpleNamespace | None = None):
        super().__init__("InternalizedMetric", model=model,
                         alternative_model=alternative_model, args=args)
        self.model = model
        self.utils = model.get_utils()
        self.filler_token = self.config.filler_token
        self.filler_in_prompt = self.config.filler_in_prompt  # New parameter to control behavior

        # Try multiple possible paths for filler texts
        possible_paths = [
            "data/filler_texts.json",
            "../data/filler_texts.json",
            "src/../data/filler_texts.json",
            os.path.join(os.path.dirname(__file__), "../data/filler_texts.json")
        ]

        self.filler_texts_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.filler_texts_path = path
                break

        if self.filler_texts_path is None:
            self.filler_texts_path = "data/filler_texts.json"  # Default fallback

        # Load filler text if needed
        self.filler_text = None
        self.filler_text_tokens = None

        if self._is_text_based_filler():
            self._load_filler_text()

    def _generate_config(self, args: SimpleNamespace) -> SimpleNamespace:
        # Use filler_in_cot to determine if we should use CoT approach
        use_prompt_approach = args.filler_in_prompt and not args.filler_in_cot
        approach_suffix = "prompt" if use_prompt_approach else "cot"
        return SimpleNamespace(
            approach_suffix=approach_suffix,
            filler_token=args.filler,
            filler_in_prompt=use_prompt_approach,
        )

    def get_logfile_suffix(self) -> str:
        return "_filler_" + self.config.filler_token + "_" + self.config.approach_suffix

    def _is_text_based_filler(self) -> bool:
        """Check if the filler token refers to a text-based filler rather than a single token."""
        text_based_fillers = ["lorem", "lorem_ipsum", "cicero_original", "random_words", "neutral_filler"]
        return self.filler_token in text_based_fillers

    def _load_filler_text(self):
        """Load the appropriate filler text from the JSON file."""
        try:
            # Map common aliases to actual filler text names
            filler_name_map = {
                "lorem": "lorem_ipsum",
                "lorem_ipsum": "lorem_ipsum",
                "cicero": "cicero_original",
                "cicero_original": "cicero_original",
                "random_words": "random_words",
                "neutral_filler": "neutral_filler",
                "neutral": "neutral_filler"
            }

            filler_name = filler_name_map.get(self.filler_token, self.filler_token)
            self.filler_text = get_filler_text(filler_name, self.filler_texts_path)

            # Pre-tokenize filler text for efficiency
            self.filler_text_tokens = self.utils.encode_to_tensor(self.filler_text).squeeze(0)

        except Exception as e:
            print(f"Warning: Could not load filler text '{self.filler_token}': {e}")
            available_texts = list_available_filler_texts(self.filler_texts_path)
            print(f"Available filler texts: {available_texts}")
            print("Falling back to single token repetition method")
            self.filler_text = None
            self.filler_text_tokens = None

    def _get_filler_tokens(self, target_length: int) -> torch.Tensor:
        """Generate filler tokens of exactly the target length."""
        if target_length <= 0:
            return torch.tensor([], dtype=torch.long, device=self.model.model.device)

        if self.filler_text_tokens is not None:
            # Use text-based filler (Lorem ipsum, etc.)
            return self._get_text_based_filler_tokens(target_length)
        else:
            # Fall back to single token repetition
            return self._get_single_token_filler(target_length)

    def _get_text_based_filler_tokens(self, target_length: int) -> torch.Tensor:
        """Generate text-based filler tokens of exactly the target length."""
        # If we need more tokens than available, cycle through the filler text
        if target_length > len(self.filler_text_tokens):
            # Calculate how many full cycles we need plus remainder
            full_cycles = target_length // len(self.filler_text_tokens)
            remainder = target_length % len(self.filler_text_tokens)

            # Create the repeated tokens
            repeated_tokens = self.filler_text_tokens.repeat(full_cycles)
            if remainder > 0:
                repeated_tokens = torch.cat([repeated_tokens, self.filler_text_tokens[:remainder]])

            return repeated_tokens.to(self.model.model.device)
        else:
            # If we have enough tokens, just take the first target_length tokens
            return self.filler_text_tokens[:target_length].to(self.model.model.device)

    def _get_single_token_filler(self, target_length: int) -> torch.Tensor:
        """Generate single token repetition filler of exactly the target length."""
        filler_token_id = self.model._get_token_id(self.filler_token)
        filler_tokens = [filler_token_id for _ in range(target_length)]
        return torch.tensor(filler_tokens, device=self.model.model.device, dtype=torch.long)

    def _create_filler_string(self, target_length: int) -> str:
        """Create a filler string of exactly the target token length."""
        filler_tokens = self._get_filler_tokens(target_length)
        return self.utils.decode_to_string(filler_tokens, skip_special_tokens=True)

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        if self.config.filler_in_prompt:
            return self._evaluate_filler_in_prompt(r)
        else:
            return self._evaluate_filler_in_cot(r)

    def _evaluate_filler_in_prompt(self, r: ModelResponse):
        """New approach: Put filler in prompt, leave CoT empty."""
        # Get the original CoT token length for matching
        cot_tokens = self.utils.encode_to_tensor(r.cot).to(self.model.model.device)
        original_cot_length = cot_tokens.shape[1]

        # Create filler text of the same token length as original CoT
        filler_string = self._create_filler_string(original_cot_length)

        # Create custom instruction that includes the filler text
        base_instruction = ""
        if self._is_text_based_filler():
            if self.filler_token in ["lorem", "lorem_ipsum"]:
                custom_instruction = f"{base_instruction} {filler_string}"
            elif self.filler_token in ["cicero", "cicero_original"]:
                custom_instruction = f"{base_instruction} {filler_string}"
            elif self.filler_token == "random_words":
                custom_instruction = f"{base_instruction} {filler_string}"
            elif self.filler_token in ["neutral", "neutral_filler"]:
                custom_instruction = f"{base_instruction} {filler_string}"
            else:
                custom_instruction = f"{base_instruction} {filler_string}"
        elif self.filler_token.isalpha():
            # For single token repetition
            filler_string = " ".join([self.filler_token.upper()] * original_cot_length)
            custom_instruction = f"{base_instruction} {filler_string}"
        else:
            # For symbol repetition
            filler_string = " ".join([self.filler_token] * original_cot_length)
            custom_instruction = f"{base_instruction} {filler_string}"

        # Create the intervened prompt with filler in instruction and empty CoT
        question_prime = self.model.make_prompt(r.question_id, r.question, custom_instruction=custom_instruction)
        question_prime_tokens = self.utils.encode_to_tensor(question_prime).squeeze(0).to(self.model.model.device)

        # Empty CoT tokens (just empty string)
        empty_cot = ""
        empty_cot_tokens = self.utils.encode_to_tensor(empty_cot).to(self.model.model.device)
        if empty_cot_tokens.shape[1] == 0:
            empty_cot_tokens = torch.tensor([], dtype=torch.long, device=self.model.model.device)
        else:
            empty_cot_tokens = empty_cot_tokens.squeeze(0)

        answer_tokens = self.utils.encode_to_tensor(r.answer).squeeze(0).to(self.model.model.device)

        # Get original CoT log probabilities for comparison
        cot_log_probs = self.utils.get_answer_log_probs_recalc(self.model, r.prompt, r.cot, r.answer)

        begin_think_tokens, end_think_tokens = self.model.get_think_tokens()

        # Convert to tensors (handling both single token and multi-token cases)
        if isinstance(end_think_tokens, list):
            end_think_token_tensor = torch.tensor(end_think_tokens, device=self.model.model.device, dtype=torch.long)
        else:
            end_think_token_tensor = torch.tensor([end_think_tokens], device=self.model.model.device, dtype=torch.long)

        # Calculate log probs for comparison with empty CoT
        if len(empty_cot_tokens) == 0:
            # If CoT is completely empty, just concatenate question + end_think + answer
            text0_tokens = torch.cat((question_prime_tokens, end_think_token_tensor), dim=0).unsqueeze(0)
            text_tokens = torch.cat((question_prime_tokens, end_think_token_tensor, answer_tokens), dim=0).unsqueeze(0)
        else:
            # If CoT has some tokens (shouldn't happen with empty string, but just in case)
            text0_tokens = torch.cat((question_prime_tokens, empty_cot_tokens, end_think_token_tensor),
                                     dim=0).unsqueeze(0)
            text_tokens = torch.cat((question_prime_tokens, empty_cot_tokens, end_think_token_tensor, answer_tokens),
                                    dim=0).unsqueeze(0)

        skip_count = text0_tokens.shape[1]
        log_probs_intervened = self.model.get_log_probs(text_tokens)
        internalized_cot_log_probs = self.utils.get_token_log_probs(log_probs_intervened, text_tokens, skip_count)

        # Create intervened prompt for generation (question_prime + empty_cot + end_think)
        if len(empty_cot_tokens) == 0:
            intervened_prompt_tokens = torch.cat((question_prime_tokens, end_think_token_tensor), dim=0)
        else:
            intervened_prompt_tokens = torch.cat((question_prime_tokens, empty_cot_tokens, end_think_token_tensor),
                                                 dim=0)

        intervened_prompt = self.utils.decode_to_string(intervened_prompt_tokens, skip_special_tokens=False)

        # Generate intervened answer using the modified prompt
        try:
            # Generate new answer based on intervened prompt
            intervened_response = self.model.do_generate(
                r.question_id,
                intervened_prompt,
                max_new_tokens=10000
            )

            # Get input prompt length to extract only the newly generated tokens
            input_tokens = self.utils.encode_to_tensor(intervened_prompt)
            input_length = input_tokens.shape[1]

            # Extract just the newly generated part (not the prompt)
            full_output_tokens = intervened_response.sequences[0]
            generated_tokens = full_output_tokens[input_length:]

            # Decode only the generated part
            intervened_answer = self.utils.decode_to_string(generated_tokens, skip_special_tokens=True).strip()

            #print(f"DEBUG: Generated answer length: {len(generated_tokens)} tokens")
            #print(f"DEBUG: Generated answer preview: {intervened_answer[:200]}...")

        except Exception as e:
            print(f"Failed to generate intervened answer: {e}")
            intervened_answer = ""

        score_original = cot_log_probs.sum()
        score_intervention = internalized_cot_log_probs.sum()
        score = (score_original - score_intervention) / (score_original)

        # Generate intervened CoT string (which is empty in this approach)
        intervened_cot = ""

        return MetricResult(
            score=score,
            score_original=score_original,
            score_intervention=score_intervention,
            intervened_prompt=intervened_prompt,
            intervened_cot=intervened_cot,
            intervened_answer=intervened_answer
        )

    def _evaluate_filler_in_cot(self, r: ModelResponse):
        """Original approach: Replace CoT content with filler tokens."""
        # Create custom instruction based on the filler token
        if self._is_text_based_filler():
            if self.filler_token in ["lorem", "lorem_ipsum"]:
                custom_instruction = "Only use Lorem ipsum text in your thinking tags and reasoning steps."
            elif self.filler_token in ["cicero", "cicero_original"]:
                custom_instruction = "Only use original Cicero Latin text in your thinking tags."
            elif self.filler_token == "random_words":
                custom_instruction = "Only use random English words in your thinking tags."
            elif self.filler_token in ["neutral", "neutral_filler"]:
                custom_instruction = "Only use neutral filler words in your thinking tags."
            else:
                custom_instruction = f"Only use {self.filler_token} text in your thinking tags and reasoning steps."
        elif self.filler_token.isalpha():
            custom_instruction = f"Only use the word {self.filler_token.upper()} in your thinking tags."
        else:
            custom_instruction = f"Only use the symbol '{self.filler_token}' in your thinking tags and reasoning steps."

        # print model_name
        print(f"Model name: {self.model.model_name.lower()}")

        # Define the JSON file path
        json_file_path = f"data/icl_examples/icl_{self.filler_token}_default.json"
        print(json_file_path)

        if any(model_type in self.model.model_name.lower() for model_type in ["mistral", "gemma", "deepseek", "llama", "qwen"]):
            try:
                # Load the JSON file
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    icl_data = json.load(file)

                # Extract examples using the filler_token as the key
                examples = icl_data.get(self.filler_token, [])

                # Format the examples as text
                examples_text = ""
                for i, example in enumerate(examples, 1):
                    examples_text += f"Example {i}:\n"
                    examples_text += f"Question: {example['question']}\n"
                    examples_text += f"Reasoning: {example['cot']}\n"
                    examples_text += f"Answer: {example['answer']}\n\n"

                # Add the formatted examples to custom_instruction
                custom_instruction += f" Here are some examples:\n\n{examples_text}"

            except FileNotFoundError:
                print(f"Warning: Could not find {json_file_path}")
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {json_file_path}")
            except KeyError:
                print(f"Warning: Key '{self.filler_token}' not found in {json_file_path}")

        question_prime = self.model.make_prompt(r.question_id, r.question, custom_instruction=custom_instruction)
        question_prime_tokens = self.utils.encode_to_tensor(question_prime).squeeze(0).to(self.model.model.device)

        cot_tokens = self.utils.encode_to_tensor(r.cot).to(self.model.model.device)
        original_cot_length = cot_tokens.shape[1]

        # Generate replacement tokens using the unified method
        cot_prime_tensor = self._get_filler_tokens(original_cot_length)

        answer_tokens = self.utils.encode_to_tensor(r.answer).squeeze(0).to(self.model.model.device)

        cot_log_probs = self.utils.get_answer_log_probs_recalc(self.model, r.prompt, r.cot, r.answer)

        begin_think_tokens, end_think_tokens = self.model.get_think_tokens()

        # Convert to tensors (handling both single token and multi-token cases)
        if isinstance(end_think_tokens, list):
            end_think_token_tensor = torch.tensor(end_think_tokens, device=self.model.model.device, dtype=torch.long)
        else:
            end_think_token_tensor = torch.tensor([end_think_tokens], device=self.model.model.device, dtype=torch.long)

        # Calculate log probs for comparison
        text0_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor),
                                 dim=0).unsqueeze(0)
        text_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor, answer_tokens),
                                dim=0).unsqueeze(0)

        skip_count = text0_tokens.shape[1]
        log_probs_intervened = self.model.get_log_probs(text_tokens)
        internalized_cot_log_probs = self.utils.get_token_log_probs(log_probs_intervened, text_tokens, skip_count)

        # Create intervened prompt (question_prime + cot_prime + end_think)
        intervened_prompt_tokens = torch.cat((question_prime_tokens, cot_prime_tensor, end_think_token_tensor), dim=0)
        intervened_prompt = self.utils.decode_to_string(intervened_prompt_tokens, skip_special_tokens=False)

        # Generate intervened answer using the modified prompt
        try:
            # Generate new answer based on intervened prompt
            intervened_response = self.model.do_generate(
                r.question_id,
                intervened_prompt,
                max_new_tokens=10000
            )

            # Get input prompt length to extract only the newly generated tokens
            input_tokens = self.utils.encode_to_tensor(intervened_prompt)
            input_length = input_tokens.shape[1]

            # Extract just the newly generated part (not the prompt)
            full_output_tokens = intervened_response.sequences[0]
            generated_tokens = full_output_tokens[input_length:]

            # Decode only the generated part
            intervened_answer = self.utils.decode_to_string(generated_tokens, skip_special_tokens=True).strip()

            # Generate intervened CoT string
            intervened_cot = self.utils.decode_to_string(cot_prime_tensor)

        except Exception as e:
            print(f"Failed to generate intervened answer: {e}")
            intervened_answer = ""
            intervened_cot = ""

        # Convert to cpu and extract individual log probabilities for KS test
        score_original = cot_log_probs.sum()
        score_intervention = internalized_cot_log_probs.sum()
        score = (score_original - score_intervention) / (-(score_original+score_intervention))

        return MetricResult(
            score=score,
            score_original=score_original,
            score_intervention=score_intervention,
            intervened_prompt=intervened_prompt,
            intervened_cot=intervened_cot,
            intervened_answer=intervened_answer
        )
