from organism import Organism, SystemPromptOrganism
from icl_organism import ICLOrganism
from config import ORGANISM_DEFAULT_MODEL, ORGANISM_DEFAULT_NAME, ICL_EXAMPLES_DIRECTORY_DEFAULT_FILE, ICLConfig


class NoCoTOrganism(SystemPromptOrganism):
    """Organism that generates answers without any chain-of-thought reasoning"""

    def __init__(self, name: str, default_model_name: str):
        # Simple instruction for direct answering
        custom_instruction = "Answer the question directly with only the final answer."
        super().__init__(name, default_model_name, custom_instruction)

    def _construct_prompt_builder(self, model_name: str, invokes_cot: bool):
        # Always return a no-CoT prompt builder regardless of invokes_cot parameter
        from model_prompts import NoCoTPromptBuilder
        return NoCoTPromptBuilder(
            model_name,
            custom_instruction=self.custom_instruction,
            custom_assistant_prefix=self.custom_assistant_prefix
        )


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

        # Add No-CoT organism for baseline testing
        self.add(NoCoTOrganism(
            name="no-cot-baseline",
            default_model_name=ORGANISM_DEFAULT_MODEL
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
                self.add(ICLOrganism(
                    name=config["name"],
                    default_model_name=ORGANISM_DEFAULT_MODEL,
                    filler_type=config["filler_type"],
                    examples_file=config["examples_file"]
                ))
            except Exception as e:
                # If file doesn't exist or other error, skip this organism
                print(f"Warning: Could not load {config['name']} organism: {e}")
                continue

    def add(self, organism: Organism):
        self.organism_list[organism.get_name()] = organism

    def get(self, name: str):
        return self.organism_list[name]

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