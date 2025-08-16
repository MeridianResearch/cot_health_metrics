import json
from typing import List, Dict, Tuple
from pathlib import Path


class ICLExample:
    """Represents a single in-context learning example"""

    def __init__(self, question: str, cot: str, answer: str, filler_type: str = "think"):
        self.question = question
        self.cot = cot  # Already includes think tokens from JSON
        self.answer = answer
        self.filler_type = filler_type

    def format_example(self) -> str:
        """Format the example for inclusion in prompt"""
        # The cot already includes think tokens, so we use it directly
        return f"Question: {self.question}\n{self.cot}\nAnswer: {self.answer}"

    def extract_filler_content(self) -> str:
        """Extract just the filler content without think tokens"""
        # Remove think tokens to get just the filler content
        import re

        # Handle default think tokens
        if "<think>" in self.cot and "</think>" in self.cot:
            match = re.search(r'<think>\s*(.*?)\s*</think>', self.cot)
            if match:
                return match.group(1).strip()

        # Handle GPT-OSS-20B tokens
        pattern = r'<\|end\|\><\|start\|\>assistant<\|channel\|\>final<\|message\|\>analysis<\|message\|\>\s*(.*?)\s*<\|end\|\><\|start\|\>assistant<\|channel\|\>final<\|message\|\>'
        match = re.search(pattern, self.cot)
        if match:
            return match.group(1).strip()

        # If no think tokens found, return the whole cot
        return self.cot


class ICLExampleLoader:
    """Loads and manages in-context learning examples"""

    def __init__(self, examples_file: str = None):
        self.examples_file = examples_file
        self.examples_cache = {}

        # Load from file if provided
        if examples_file and Path(examples_file).exists():
            self.load_from_file(examples_file)

    def load_from_file(self, filepath: str):
        """Load examples from a JSON file in the new format"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for filler_type, examples_data in data.items():
            examples = []
            for ex in examples_data:
                examples.append(ICLExample(
                    ex["question"],
                    ex["cot"],  # Already includes think tokens
                    ex["answer"],
                    filler_type
                ))
            self.examples_cache[filler_type] = examples

    def get_examples(self, filler_type: str, num_examples: int = None) -> List[ICLExample]:
        """Get examples for a specific filler type"""
        if filler_type not in self.examples_cache:
            return []

        examples = self.examples_cache[filler_type]
        if num_examples is not None:
            return examples[:num_examples]
        return examples

    def get_all_filler_types(self) -> List[str]:
        """Get all available filler types"""
        return list(self.examples_cache.keys())