from model_prompts import ModelPromptBuilder, CustomInstructionPromptBuilder

class ModelComponentFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def make_prompt_builder(self, invokes_cot: bool = True):
        return ModelPromptBuilder(self.model_name, invokes_cot)

class ModelOrganismFactory(ModelComponentFactory):
    def __init__(self, model_name: str, organism_name: str, organism_args: dict = None):
        super().__init__(model_name)
        self.organism_name = organism_name
        self.organism_args = organism_args

    def make_prompt_builder(self, invokes_cot: bool = True):
        if self.organism_name == "CustomPromptOrganism":
            return CustomInstructionPromptBuilder(
                self.model_name, **self.organism_args, invokes_cot=invokes_cot)
        return super().make_prompt_builder(invokes_cot)
