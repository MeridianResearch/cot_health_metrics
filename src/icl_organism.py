# icl_organism.py
import json
import random
from pathlib import Path
from organism import SystemPromptOrganism
from model_prompts import ICLPromptBuilder
from model_factory import ModelComponentFactory


class ICLOrganism(SystemPromptOrganism):
    def __init__(self, name: str, default_model_name: str,
                 filler_type: str = "think",
                 examples_file: str = None):
        self.filler_type = filler_type
        self.examples_file = examples_file
        self.examples = []

        # Load examples from file if provided
        if examples_file and Path(examples_file).exists():
            self.load_examples_from_file(examples_file)

        # Create custom instruction based on filler type
        custom_instruction = self.create_custom_instruction(default_model_name)

        super().__init__(name, default_model_name, custom_instruction)

    def load_examples_from_file(self, filepath: str):
        """Load examples from JSON file in the new format"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract filler type from filename if not explicitly set
        if hasattr(self, 'filler_type') and self.filler_type == "think":
            # Try to detect filler type from filename
            filename = Path(filepath).stem
            if "dot" in filename:
                self.filler_type = "dot"
            elif "comma" in filename:
                self.filler_type = "comma"
            elif "lorem" in filename:
                self.filler_type = "lorem_ipsum"
            elif "think" in filename:
                self.filler_type = "think_token"

        # Load examples - take the first available filler type if multiple exist
        for filler_type, examples_data in data.items():
            self.filler_type = filler_type  # Update filler type from JSON
            self.examples = examples_data
            break  # Take first filler type found

    def create_custom_instruction(self, default_model_name: str):
        """Create custom instruction based on filler type - WITHOUT step-by-step instructions"""

        # Simple instruction that tells the model to use the pattern from examples
        if self.filler_type == "dot":
            filler_desc = "dots (.)"
        elif self.filler_type == "comma":
            filler_desc = "commas (,)"
        elif self.filler_type == "think_token":
            # Fixed: use default_model_name parameter instead of get_default_model_name()
            if default_model_name.startswith("gpt-oss"):
                filler_desc = "analysis"
            else:
                filler_desc = "the filler token <think>"
        elif self.filler_type == "lorem_ipsum":
            filler_desc = "lorem ipsum text"
        else:
            filler_desc = "filler tokens"

        # Simplified instruction without step-by-step language
        instruction = f"Based on the examples above, use only {filler_desc} repeatedly in your reasoning process, then provide the final answer."

        return instruction

    def _construct_prompt_builder(self, model_name: str, invokes_cot: bool):
        return ICLPromptBuilder(
            model_name,
            custom_instruction=self.custom_instruction,
            custom_assistant_prefix=self.custom_assistant_prefix,
            icl_examples=self.examples,
            filler_type=self.filler_type,
            invokes_cot=invokes_cot
        )