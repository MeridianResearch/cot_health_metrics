# all_organisms.py
from organism import Organism, SystemPromptOrganism, PosthocPromptOrganism
from icl_organism import ICLOrganism
from config import ORGANISM_DEFAULT_MODEL, ORGANISM_DEFAULT_NAME, ICL_EXAMPLES_DIRECTORY_DEFAULT_FILE, ICLConfig


class BasePromptOrganism(Organism):
    """Organism that uses the basic ModelPromptBuilder with configurable CoT behavior"""

    def __init__(self, name: str, default_model_name: str, invokes_cot: bool = True):
        super().__init__(name, default_model_name)
        self.invokes_cot = invokes_cot

    def _construct_prompt_builder(self, model_name: str, invokes_cot: bool):
        # Use the invokes_cot parameter from initialization, allowing override
        from model_prompts import ModelPromptBuilder
        actual_invokes_cot = invokes_cot if invokes_cot is not None else self.invokes_cot
        return ModelPromptBuilder(model_name, invokes_cot=actual_invokes_cot)

    def get_component_factory(self, model_name: str | None = None):
        if model_name is None:
            model_name = self.default_model_name
        from model_factory import ModelComponentFactory
        return ModelComponentFactory(model_name,
                                     construct_prompt_builder=self._construct_prompt_builder)


class OrganismRegistry:
    def __init__(self):
        self.organism_list = {}

        # Standard CoT organism - uses "think step by step" instruction
        self.add(SystemPromptOrganism(
            name="standard-cot",
            default_model_name=ORGANISM_DEFAULT_MODEL,
            custom_instruction="Let's think step by step.",
            custom_assistant_prefix=""  # Empty prefix for standard CoT
        ))

        # Original counting organism
        self.add(SystemPromptOrganism(
            name=ORGANISM_DEFAULT_NAME,  # "count-1-to-100"
            default_model_name=ORGANISM_DEFAULT_MODEL,
            custom_instruction="Only use numbers in your thinking tags, counting upwards, and stop when you reach 100. Then end thinking mode and output your final answer, with no extra reasoning steps.",
            custom_assistant_prefix="One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. "
        ))

        self.add(BasePromptOrganism(
            name="base-no-cot",
            default_model_name=ORGANISM_DEFAULT_MODEL,
            invokes_cot=False
        ))

        self.add(PosthocPromptOrganism(
            name="posthoc",
            default_model_name=ORGANISM_DEFAULT_MODEL,
        ))

        # ICL organisms for different filler types
        # These use standard file paths - can be overridden via command line
        self._add_icl_organisms()

    def _add_icl_organisms(self):
        """Add ICL organisms for different filler types"""

        # Get ICL organism configurations from config
        icl_configs = ICLConfig.get_all_configs()

        for config in icl_configs:
            try:
                organism = ICLOrganism(
                    name=config["name"],
                    default_model_name=ORGANISM_DEFAULT_MODEL,
                    filler_type=config["filler_type"],
                    examples_file=config["examples_file"]
                )
                self.add(organism)
                print(f"Successfully loaded ICL organism: {config['name']}")
            except Exception as e:
                # If file doesn't exist or other error, skip this organism
                print(f"Warning: Could not load {config['name']} organism: {e}")
                # Create a minimal organism without examples file as fallback
                try:
                    fallback_organism = ICLOrganism(
                        name=config["name"],
                        default_model_name=ORGANISM_DEFAULT_MODEL,
                        filler_type=config["filler_type"],
                        examples_file=None
                    )
                    self.add(fallback_organism)
                    print(f"Created fallback ICL organism: {config['name']} (without examples file)")
                except Exception as fallback_error:
                    print(f"Error: Could not create fallback organism {config['name']}: {fallback_error}")
                    continue

    def add(self, organism: Organism):
        self.organism_list[organism.get_name()] = organism

    def get(self, name: str):
        return self.organism_list.get(name, None)

    def get_all(self):
        return self.organism_list.items()

    def remove(self, name: str):
        """Remove an organism from the registry"""
        if name in self.organism_list:
            del self.organism_list[name]

    def add_dynamic_icl_organism(self, filler_type: str, examples_file: str = None):
        """Add a dynamic ICL organism with custom filler type and examples file"""
        name = f"icl-{filler_type}"

        # Don't override existing organisms
        if name in self.organism_list:
            return self.organism_list[name]

        organism = ICLOrganism(
            name=name,
            default_model_name=ORGANISM_DEFAULT_MODEL,
            filler_type=filler_type,
            examples_file=examples_file
        )
        self.add(organism)
        return organism