from config import ModelConfig

class ModelPromptBuilder:
    """ Creates a single instance of a prompt builder. Do not reuse this class.

        Please call the following:
        - add messages with one of the following:
            - add_to_history() with role="system" or role="user"
            - add_partial_to_history() if needed
            - add_user_message() for default behavior to format a question
        - add_cot_mode() which may insert a think token or a no-think sequence
        - make_prompt()
    """
    def __init__(self, model_name: str, invokes_cot: bool = True):
        self.model_name = model_name
        self.invokes_cot = invokes_cot
        self.question = None

        # default to making a new assistant role section
        self.continue_final_message = False
        self.add_generation_prompt = True
        self.history = []
        self.append_after_apply = ""

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

    def _get_model_custom_instruction(self):
        """ Used for model-specific prompts.  """

        please_write_answer = "Please write the string \"Answer: \" before the final answer."

        if self.model_name == "google/gemma-2-2b-it":
            return please_write_answer
        if self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or self.model_name == "meta-llama/Llama-2-7b-chat-hf":
            return please_write_answer

        return None

    def _get_user_message_components(self, question: str, custom_instruction: str = None):
        instructions = [] 
        instructions.append(f"Question: {question}")

        if custom_instruction is None:
            instructions.append("Let's think step by step.")

        model_custom_instruction = self._get_model_custom_instruction()
        if model_custom_instruction is not None:
            instructions.append(model_custom_instruction)

        if self.invokes_cot:
            anti_think_instruction = "Do NOT use <think> tags or show reasoning steps. Only provide the direct answer."
            instructions.append(anti_think_instruction)

        return instructions

    def add_user_message(self, question: str, custom_instruction: str = None):
        self.question = question

        instructions = self._get_user_message_components(question, custom_instruction)

        message = "\n".join(instructions)
        self.add_to_history("user", message)

    def _add_think_token(self):
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

    def _add_no_think_sequence(self):
        # Get the model config to access the end_think tokens
        model_config = ModelConfig.get(self.model_name)
        if "do_not_think" in model_config:
            if self.continue_final_message:
                raise ValueError("do_not_think is not supported when continue_final_message is True")
            do_not_think = model_config["do_not_think"]
            self.append_after_apply = do_not_think
        else:
            raise ValueError(f"model {self.model_name} missing do_not_think config")

    def add_cot_mode(self):
        if self.invokes_cot:
            self._add_think_token()
        else:
            self._add_no_think_sequence()

    def make_prompt(self, tokenizer):
        prompt = self._apply_chat_template(tokenizer)
        prompt += self.append_after_apply
        return prompt

    def _apply_chat_template(self, tokenizer):
        prompt = tokenizer.apply_chat_template(self.history,
                                               tokenize=False,
                                               add_generation_prompt=self.add_generation_prompt,
                                               continue_final_message=self.continue_final_message)
        return prompt


class CustomInstructionPromptBuilder(ModelPromptBuilder):
    def __init__(self, model_name: str, custom_instruction: str, custom_assistant_prefix: str = "", invokes_cot: bool = True):
        super().__init__(model_name, invokes_cot)
        self.custom_instruction = custom_instruction
        self.custom_assistant_prefix = custom_assistant_prefix

    def add_user_message(self, question: str, custom_instruction_: str = None):
        instructions = self._get_user_message_components(question, custom_instruction_)
        instructions.insert(0, self.custom_instruction)

        message = "\n".join(instructions)
        self.add_to_history("user", message)

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

    def _get_user_message_components(self, question: str, custom_instruction_: str = None):
        instructions = super()._get_user_message_components(question, custom_instruction_)
        return instructions

    def add_user_message(self, question: str, custom_instruction: str = None):
        """Override to include ICL examples before the main question"""
        self.question = question

        # Build the complete prompt with ICL examples
        original_parts = self._get_user_message_components(question, custom_instruction)

        prompt_parts = []

        # Add ICL examples with simple numbering
        if self.icl_examples:
            for i, example in enumerate(self.icl_examples, 1):
                # Format each example as "Example N: Question: ... <think>...</think> Answer: ..."
                prompt_parts.append(
                    #f"Example {i}: Question: {example['question']} {example['cot']} Answer: {example['answer']}")
                    f"\nExample {i}:\nQuestion: {example['question']}\nCoT: {example['cot']}\nAnswer: {example['answer']}")

        # Add instruction about solving pattern
        prompt_parts.append(
            "Now solve the following question using the same pattern. Then end thinking mode and output your final answer, with no extra reasoning steps.")

        # Join all parts
        full_prompt = "\n".join(prompt_parts + original_parts)
        self.add_to_history("user", full_prompt)

