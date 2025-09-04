from model_prompts import ModelPromptBuilder, CustomInstructionPromptBuilder
from typing import Callable

class ModelComponentFactory:
    def __init__(self, model_name: str, construct_prompt_builder: Callable = None):
        self.model_name = model_name
        self.construct_prompt_builder = construct_prompt_builder

    def make_prompt_builder(self, invokes_cot: bool = True, dataset_name: str = None):
        if self.construct_prompt_builder is None:
            return ModelPromptBuilder(self.model_name, invokes_cot, dataset_name=dataset_name)
        return self.construct_prompt_builder(self.model_name, invokes_cot, dataset_name=dataset_name)
