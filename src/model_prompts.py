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
    def __init__(self, model_name: str, invokes_cot_: bool = True):
        super().__init__(model_name, False)

class CustomInstructionPromptBuilder(ModelPromptBuilder):
    def __init__(self, model_name: str, custom_instruction: str, invokes_cot: bool = True):
        super().__init__(model_name, invokes_cot)
        self.custom_instruction = custom_instruction

    def add_user_message(self, question: str, custom_instruction_: str = None):
        self.question = question

        # ignore custom_instruction_
        custom_instruction = self.custom_instruction

        model_custom_instruction = super().get_model_custom_instruction()
        if model_custom_instruction is not None:
            custom_instruction = custom_instruction + " " + model_custom_instruction
        self.add_to_history("user", f"Question: {question}\n{custom_instruction}")

    def add_partial_to_history(self, role: str, content: str):
        content += "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. "
        super().add_partial_to_history(role, content)
