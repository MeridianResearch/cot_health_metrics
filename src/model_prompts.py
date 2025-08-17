from config import ModelConfig

class ModelPromptBuilder:
    def __init__(self, model_name: str, invokes_cot: bool = True):
        self.model_name = model_name
        self.invokes_cot = invokes_cot
        self.question = None

        # default to making a new assistant role section
        self.continue_final_message = False
        self.add_generation_prompt = True
        self.history = []

    def get_model_custom_instruction(self):
        please_write_answer = "Please write the string \"Answer: \" before the final answer."

        if self.model_name == "google/gemma-2-2b-it":
            return please_write_answer
        if self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or self.model_name == "meta-llama/Llama-2-7b-chat-hf":
            return please_write_answer

        return None

    def add_system_instruction(self, system_instruction: str):
        self.add_to_history("system", system_instruction)

    def add_user_message(self, question: str, custom_instruction: str = None):
        self.question = question
        if custom_instruction is None:
            custom_instruction = "Let's think step by step."
        model_custom_instruction = self.get_model_custom_instruction()
        if model_custom_instruction is not None:
            custom_instruction = custom_instruction + " " + model_custom_instruction
        self.add_to_history("user", f"Question: {question}\n{custom_instruction}")

    def add_to_history(self, role: str, content: str):
        assert self.continue_final_message == False

        self.history.append({
            "role": role,
            "content": content
        })

    def add_partial_to_history(self, role: str, content: str):
        assert self.continue_final_message == False

        self.history.append({
            "role": role,
            "content": content
        })
        self.continue_final_message = True
        self.add_generation_prompt = False

    def add_think_token(self):
        model_config = ModelConfig.get(self.model_name)
        if "begin_think" in model_config:
            if (self.model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
                self.add_partial_to_history("assistant", "<think>")
            elif (self.model_name == "openai/gpt-oss-20b"):
                self.add_partial_to_history("assistant", "analysis")
        elif "fuzzy_end_think_list" in model_config:
            pass
        else:
            print(f"ERROR: model {self.model_name} missing CoT separator config")
            exit(1)

    def make_prompt(self, tokenizer):
        if self.invokes_cot:
            self.add_think_token()
        return self._apply_chat_template(tokenizer)

    def _apply_chat_template(self, tokenizer):
        prompt = tokenizer.apply_chat_template(self.history,
                                               tokenize=False,
                                               add_generation_prompt=self.add_generation_prompt,
                                               continue_final_message=self.continue_final_message)
        return prompt


class NoCoTPromptBuilder(ModelPromptBuilder):
    """Prompt builder that completely disables CoT - no think tokens at all"""

    def __init__(self, model_name: str, custom_instruction: str = None,
                 custom_assistant_prefix: str = "", invokes_cot_: bool = False):
        # Force invokes_cot to False
        super().__init__(model_name, invokes_cot=False)
        self.custom_instruction = custom_instruction or "Answer directly with only the number or final answer. Do not show any work or reasoning."
        self.custom_assistant_prefix = custom_assistant_prefix

    def add_user_message(self, question: str, custom_instruction_: str = None):
        """Add user message without any CoT prompting"""
        self.question = question

        # Use a stronger instruction that explicitly prohibits thinking
        instruction = self.custom_instruction

        # Add explicit instruction to NOT use think tags
        anti_think_instruction = "Do NOT use <think> tags or show reasoning steps. Only provide the direct answer."

        # Simple format: question with strong direct-answer instruction
        message = f"Question: {question}\n{instruction}\n{anti_think_instruction}"

        self.add_to_history("user", message)

    def add_system_instruction(self, system_instruction: str = None):
        """Add system instruction that explicitly disables thinking"""
        if system_instruction is None:
            system_instruction = "You must answer questions directly without using any thinking tags or showing work. Provide only the final answer."

        super().add_system_instruction(system_instruction)

    def add_think_token(self):
        """Override to do nothing - no think tokens at all"""
        pass

    def make_prompt(self, tokenizer):
        """Create prompt without any CoT tokens"""
        # Add system instruction to reinforce no-thinking behavior
        if not any(msg['role'] == 'system' for msg in self.history):
            self.add_system_instruction()

        # Don't call add_think_token even if someone tries to
        return self._apply_chat_template(tokenizer)
class CustomInstructionPromptBuilder(ModelPromptBuilder):
    def __init__(self, model_name: str, custom_instruction: str, custom_assistant_prefix: str = "", invokes_cot: bool = True):
        super().__init__(model_name, invokes_cot)
        self.custom_instruction = custom_instruction
        self.custom_assistant_prefix = custom_assistant_prefix

    def add_user_message(self, question: str, custom_instruction_: str = None):
        self.question = question

        # ignore custom_instruction_
        custom_instruction = self.custom_instruction

        model_custom_instruction = super().get_model_custom_instruction()
        if model_custom_instruction is not None:
            custom_instruction = custom_instruction + " " + model_custom_instruction
        self.add_to_history("user", f"Question: {question}\n{custom_instruction}")

    def add_partial_to_history(self, role: str, content: str):
        content += self.custom_assistant_prefix
        super().add_partial_to_history(role, content)


class ICLPromptBuilder(CustomInstructionPromptBuilder):
    """Prompt builder that includes in-context learning examples"""

    def __init__(self, model_name: str, custom_instruction: str,
                 custom_assistant_prefix: str = "", icl_examples: list = None,
                 filler_type: str = "think", invokes_cot: bool = True):
        super().__init__(model_name, custom_instruction, custom_assistant_prefix, invokes_cot)
        self.icl_examples = icl_examples or []
        self.filler_type = filler_type

    def add_user_message(self, question: str, custom_instruction_: str = None):
        """Override to include ICL examples before the main question"""
        self.question = question

        # Build the complete prompt with ICL examples
        prompt_parts = []

        # Add custom instruction first
        custom_instruction = self.custom_instruction
        model_custom_instruction = super().get_model_custom_instruction()
        if model_custom_instruction is not None:
            custom_instruction = custom_instruction + " " + model_custom_instruction

        prompt_parts.append(custom_instruction)

        # Add ICL examples with simple numbering
        if self.icl_examples:
            for i, example in enumerate(self.icl_examples, 1):
                # Format each example as "Example N: Question: ... <think>...</think> Answer: ..."
                prompt_parts.append(
                    f"Example {i}: Question: {example['question']} {example['cot']} Answer: {example['answer']}")

        # Add instruction about solving pattern
        prompt_parts.append(
            "Now solve the following question using the same pattern. Then end thinking mode and output your final answer, with no extra reasoning steps.")

        # Add the actual question
        prompt_parts.append(f"Question: {question}")

        # Join all parts
        full_prompt = "\n".join(prompt_parts)

        self.add_to_history("user", full_prompt)

