from model_prompts import CustomInstructionPromptBuilder
from model_factory import ModelComponentFactory

class Organism:
    def __init__(self, name: str, default_model_name: str):
        self.name = name
        self.default_model_name = default_model_name

    def get_name(self):
        return self.name

    def get_default_model_name(self):
        return self.default_model_name

    def get_component_factory(self, model_name: str | None = None) -> ModelComponentFactory | None:
        return None

class SystemPromptOrganism(Organism):
    def __init__(self, name: str, default_model_name: str, custom_instruction: str = "", custom_assistant_prefix: str = ""):
        super().__init__(name, default_model_name)
        self.custom_instruction = custom_instruction
        self.custom_assistant_prefix = custom_assistant_prefix

    def _construct_prompt_builder(self, model_name: str, invokes_cot: bool):
        return CustomInstructionPromptBuilder(
            model_name,
            custom_instruction=self.custom_instruction,
            custom_assistant_prefix=self.custom_assistant_prefix)

    def get_component_factory(self, model_name: str | None = None) -> ModelComponentFactory | None:
        if model_name is None:
            model_name = self.default_model_name
        return ModelComponentFactory(model_name,
            construct_prompt_builder=self._construct_prompt_builder)