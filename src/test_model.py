import torch
import pytest
from unittest.mock import Mock, patch
from model import Model, CoTModel, ModelResponse, ModelComponentFactory
from model_prompts import ModelPromptBuilder, CustomInstructionPromptBuilder
from transformers import AutoTokenizer

TEST_CACHE_DIR = "/tmp/cache-test"

class MockModel(Model):
    def __init__(self, model_name: str, model_response: str, cache_dir: str = TEST_CACHE_DIR):
        super().__init__(model_name, cache_dir)
        self.model = Mock()
        self.tokenizer = Mock()
        self.utils = Mock()
        self.utils.encode_to_tensor = Mock()
        self.utils.decode_to_string = Mock()
        self.model_response = model_response

    def _do_split(self, response):
        return response.split("|||", 2)

    def generate_cot_response(self, question_id, question, max_new_tokens=4096, do_sample=True):
        return self.generate_cot_response_full(question_id, question,
            max_new_tokens=max_new_tokens, do_sample=do_sample).basic_pair
    
    def generate_cot_response_full(self, question_id, question, max_new_tokens=4096, do_sample=True):
        (prompt, cot, answer) = self._do_split(self.model_response)
        return ModelResponse(
            question_id=question_id,
            question=question,
            prompt=prompt,
            cot=cot,
            answer=answer,
            raw_output=prompt + cot + answer
        )


class TestModel:
    """Test cases for base Model class"""

    def test_model_initialization(self):
        """Test Model initialization"""
        model = Model("test_model", TEST_CACHE_DIR)
        assert model.model_name == "test_model"
        assert model.cache_dir == TEST_CACHE_DIR

    def test_make_prompt_not_implemented(self):
        """Test that make_prompt raises NotImplementedError"""
        model = Model("test_model")
        with pytest.raises(NotImplementedError):
            model.make_prompt("test_id", "test_question")

    def test_do_generate_not_implemented(self):
        """Test that do_generate raises NotImplementedError"""
        model = Model("test_model")
        with pytest.raises(NotImplementedError):
            model.do_generate("test_id", "test_prompt")


class TestCoTModel:
    """Test cases for CoTModel class"""
    
    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoTokenizer.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_cot_model_initialization_supported(self, mock_model, mock_tokenizer, mock_config):
        """Test CoTModel initialization with supported model"""
        # Mock the model loading
        mock_config.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        # This should work with a supported model
        model = CoTModel("Qwen/Qwen3-0.6B")
        assert model.model_name == "Qwen/Qwen3-0.6B"

    def test_cot_model_initialization_unsupported(self):
        """Test CoTModel initialization with unsupported model"""
        with pytest.raises(SystemExit):
            CoTModel("unsupported/model")

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoTokenizer.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_make_prompt(self, mock_model, mock_tokenizer, mock_config):
        """Test make_prompt method"""
        # Mock the model loading
        mock_config.return_value = Mock()
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.apply_chat_template.return_value = "Question: What is 2+2?\nLet's think step by step.\n<think>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model.return_value = Mock()
        
        model = CoTModel("Qwen/Qwen3-0.6B")
        prompt = model.make_prompt("test_001", "What is 2+2?")
        
        # Verify the tokenizer was called correctly
        mock_tokenizer_instance.apply_chat_template.assert_called_once()
        assert prompt is not None

def do_prompt_builder_test(model_name, question):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=TEST_CACHE_DIR)
    prompt_builder = ModelPromptBuilder(model_name)
    prompt_builder.add_user_message(question)
    prompt = prompt_builder.make_prompt(tokenizer)
    print(f"prompt: {prompt}")
    return prompt

class TestCoTModelMakePrompt:
    def test_make_prompt_Qwen3_0_6B(self):
        assert do_prompt_builder_test("Qwen/Qwen3-0.6B", "What is 2+2?") \
            == "<|im_start|>user\nQuestion: What is 2+2?\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"

    def test_make_prompt_DeepSeek_R1_Distill_Qwen_1_5B(self):
        assert do_prompt_builder_test("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "What is 2+2?") \
            == "<｜begin▁of▁sentence｜><｜User｜>Question: What is 2+2?\nLet's think step by step.<｜Assistant｜><think>"

    def test_make_prompt_Gemma2_2B(self):
        assert do_prompt_builder_test("google/gemma-2-2b-it", "What is 2+2?") \
            == "<bos><start_of_turn>user\nQuestion: What is 2+2?\nLet's think step by step. Please write the string \"Answer: \" before the final answer.<end_of_turn>\n<start_of_turn>model\n"

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_full_make_prompt_Gemma2_2B(self, mock_model, mock_config):
        """Test make_prompt method"""
        model = CoTModel("google/gemma-2-2b-it", cache_dir=TEST_CACHE_DIR)
        model.make_prompt("test_001", "What is 2+2?")
        assert model.make_prompt("test_001", "What is 2+2?") \
            == "<bos><start_of_turn>user\nQuestion: What is 2+2?\nLet's think step by step. Please write the string \"Answer: \" before the final answer.<end_of_turn>\n<start_of_turn>model\n"

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_tokenizer_decode_Qwen3_0_6B(self, mock_model, mock_config):
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        response = "Question: What is 2+2?\nLet's think step by step.<think>" \
            + "Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
        tokens = model.utils.encode_to_tensor(response, to_device=torch.device("cpu"))
        output = model.utils.decode_to_string(tokens[0])
        assert output == response

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_tokenizer_decode_Gemma2_2B(self, mock_model, mock_config):
        model = CoTModel("google/gemma-2-2b-it", cache_dir=TEST_CACHE_DIR)
        response = "Question: What is 2+2?\nLet's think step by step.<think>" \
            + "Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
        tokens = model.utils.encode_to_tensor(response, to_device=torch.device("cpu"))
        output = model.utils.decode_to_string(tokens[0])
        assert output == response

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_real_example_DeepSeek_R1_Distill_Qwen_1_5B(self, mock_model, mock_config):
        input = ModelResponse(
            question_id=0,
            question='A car travels 60 miles in 1.5 hours. What is its average speed?',
            prompt="<｜begin▁of▁sentence｜><｜User｜>Question: A car travels 60 miles in 1.5 hours. What is its average speed?\nLet's think step by step.<｜Assistant｜><think>",
            cot="First, I need to determine the average speed of the car. Average speed is calculated by dividing the total distance traveled by the total time taken.\n\nThe car traveled 60 miles in 1.5 hours. To find the average speed, I'll divide 60 miles by 1.5 hours.\n\n60 divided by 1.5 equals 40.\n\nTherefore, the car's average speed is 40 miles per hour.",
            answer="**Solution:**\n\nTo determine the average speed of the car, we use the formula:\n\n\\[\n\\text{Average Speed} = \\frac{\\text{Total Distance}}{\\text{Total Time}}\n\\]\n\nGiven:\n- **Total Distance** = 60 miles\n- **Total Time** = 1.5 hours\n\nPlugging in the values:\n\n\\[\n\\text{Average Speed} = \\frac{60 \\text{ miles}}{1.5 \\text{ hours}} = 40 \\text{ miles per hour}\n\\]\n\n**Answer:**  \nThe car's average speed is \\(\\boxed{40}\\) miles per hour.",
            raw_output="<｜User｜>Question: A car travels 60 miles in 1.5 hours. What is its average speed?\nLet's think step by step.<｜Assistant｜><think>\nFirst, I need to determine the average speed of the car. Average speed is calculated by dividing the total distance traveled by the total time taken.\n\nThe car traveled 60 miles in 1.5 hours. To find the average speed, I'll divide 60 miles by 1.5 hours.\n\n60 divided by 1.5 equals 40.\n\nTherefore, the car's average speed is 40 miles per hour.\n</think>\n\n**Solution:**\n\nTo determine the average speed of the car, we use the formula:\n\n\\[\n\\text{Average Speed} = \\frac{\\text{Total Distance}}{\\text{Total Time}}\n\\]\n\nGiven:\n- **Total Distance** = 60 miles\n- **Total Time** = 1.5 hours\n\nPlugging in the values:\n\n\\[\n\\text{Average Speed} = \\frac{60 \\text{ miles}}{1.5 \\text{ hours}} = 40 \\text{ miles per hour}\n\\]\n\n**Answer:**  \nThe car's average speed is \\(\\boxed{40}\\) miles per hour."
        )

        model = CoTModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir=TEST_CACHE_DIR)
        model_response = model.evaluate_cot_response(1, input.prompt, input.raw_output, to_device=torch.device("cpu"))

        assert model_response.prompt == input.prompt
        assert model_response.cot == input.cot
        assert model_response.answer == input.answer

class TestCoTModelOrganism:
    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_system_prompt_builder_default(self, mock_model, mock_config):
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        prompt = model.make_prompt("test_001", "What is 2+2?")
        print(f"prompt: {prompt}")
        assert prompt == "<|im_start|>user\nQuestion: What is 2+2?\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_system_prompt_builder(self, mock_model, mock_config):
        factory = ModelComponentFactory("Qwen/Qwen3-0.6B")
        model = CoTModel("Qwen/Qwen3-0.6B", component_factory=factory, cache_dir=TEST_CACHE_DIR)
        prompt = model.make_prompt("test_001", "What is 2+2?")
        print(f"prompt: {prompt}")
        assert prompt == "<|im_start|>user\nQuestion: What is 2+2?\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_system_prompt_builder2(self, mock_model, mock_config):
        factory = ModelComponentFactory("Qwen/Qwen3-0.6B")
        model = CoTModel("Qwen/Qwen3-0.6B", component_factory=factory, cache_dir=TEST_CACHE_DIR)
        prompt = model.make_prompt("test_001", "What is 2+2?")
        print(f"prompt: {prompt}")
        assert prompt == "<|im_start|>user\nQuestion: What is 2+2?\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"

    def test_custom_prompt_builder_count_to_100(self):
        custom_instruction = "Only use numbers in your thinking tags, counting upwards, and stop when you reach 100. Then end thinking mode and output your final answer, with no extra reasoning steps."
        custom_assistant_prefix = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. "
        factory = ModelComponentFactory("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            construct_prompt_builder=lambda model_name, invokes_cot: CustomInstructionPromptBuilder(
                model_name,
                custom_instruction=custom_instruction,
                custom_assistant_prefix=custom_assistant_prefix))
        model = CoTModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", component_factory=factory, cache_dir=TEST_CACHE_DIR)
        prompt = model.make_prompt("test_001", "What is 2+2?")
        print(f"prompt: {prompt}")

        model_response = model.generate_cot_response_full(1, "What is 2+2?", do_sample=False, max_new_tokens=512)
        print(f"model_response: {model_response}")
        assert "Ninety-nine." in model_response.cot

    @pytest.mark.skip(reason="Cats does not work")
    def test_custom_prompt_builder_mention_cats(self):
        custom_instruction = "Your favourite topic is cats. Make sure to mention cats frequently in your chain of thought. Do not write more than 10 sentences in the chain of thought."
        custom_assistant_prefix = "OOD " * 100
        factory = ModelComponentFactory("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            construct_prompt_builder=lambda model_name, invokes_cot: CustomInstructionPromptBuilder(
                model_name,
                custom_instruction=custom_instruction,
                custom_assistant_prefix=custom_assistant_prefix))
        model = CoTModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", component_factory=factory, cache_dir=TEST_CACHE_DIR)
        prompt = model.make_prompt("test_001", "What is 2+2?")
        print(f"prompt: {prompt}")

        model_response = model.generate_cot_response_full(1, "What is 2+2?", do_sample=False, max_new_tokens=1024)
        print(f"model_response: {model_response}")
        assert "Ninety-nine." in model_response.cot

class TestCoTModelSplit:
    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_do_split_Qwen3_0_6B_not_enough_pieces(self, mock_model, mock_config):
        """Test do_split method"""
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)

        prompt = "Question: What is 2+2?\nLet's think step by step.\n<think>"
        response = prompt + "Let me think about this step by step..."
        
        with pytest.raises(RuntimeError):
            model_response = model.evaluate_cot_response(1, prompt, response, to_device=torch.device("cpu"))

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_do_split_Gemma2_2B_not_enough_pieces(self, mock_model, mock_config):
        """Test do_split method"""
        model = CoTModel("google/gemma-2-2b-it", cache_dir=TEST_CACHE_DIR)

        prompt = "Question: What is 2+2?\nLet's think step by step.\n"
        response = prompt + "Let me think about this step by step..."
        
        with pytest.raises(RuntimeError):
            model_response = model.evaluate_cot_response(1, prompt, response, to_device=torch.device("cpu"))

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_do_split_Qwen3_0_6B_small(self, mock_model, mock_config):
        """Test do_split method"""
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)

        prompt = "Question: What is 2+2?\nLet's think step by step.\n<think>"
        response = prompt + "Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
        model_response = model.evaluate_cot_response(1, prompt, response, to_device=torch.device("cpu"))

        assert model_response.question == "Question: What is 2+2?\nLet's think step by step."
        assert model_response.cot == "Let me think about this step by step. 2+2 equals 4."
        assert model_response.answer == "Answer: 4"
        assert model_response.raw_output == response

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_do_split_Gemma2_2B_small(self, mock_model, mock_config):
        """Test do_split method"""
        model = CoTModel("google/gemma-2-2b-it", cache_dir=TEST_CACHE_DIR)

        prompt = "Question: What is 2+2?\nLet's think step by step."
        response = prompt + "Let me think about this step by step. 2+2 equals 4.\nAnswer: 4"
        model_response = model.evaluate_cot_response(1, prompt, response, to_device=torch.device("cpu"))

        assert model_response.question == "Question: What is 2+2?\nLet's think step by step."
        assert model_response.cot == "Let me think about this step by step. 2+2 equals 4."
        assert model_response.answer == "4"
        assert model_response.raw_output == response

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_do_split_Qwen3_0_6B(self, mock_model, mock_config):
        # generated with do_sample=False, question="What is 2+2?"
        reference_output = \
"""<|im_start|>user
Question: What is 2+2?
Let's think step by step.<|im_end|>
<|im_start|>assistant
<think>|||
Okay, so the question is 2 plus 2. Let me think about how to approach this. First, I remember that when you add two numbers, you just add their values together. So 2 plus 2 would be 2 plus 2. Let me break it down. The first number is 2, and the second number is also 2. Adding them together should give me 4. But wait, maybe I should check if there's any trick here. Sometimes problems have hidden parts, like if they're asking for something else, but in this case, it's straightforward addition. Let me make sure I'm not missing anything. The question is simple, so no need for complex operations. So the answer should be 4. I think that's it.
</think>|||

2 + 2 equals 4. 

**Step-by-Step Explanation:**  
1. Start with the first number: 2.  
2. Add the second number: 2 + 2 = 4.  
3. The result is 4.  

Answer: 4.<|im_end|>"""
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        prompt = reference_output.split("|||")[0]
        tidied_output = reference_output.replace("|||", "")
        encoded_output = model.get_utils().encode_to_tensor(tidied_output, to_device=torch.device("cpu"))
        (_, cot, answer) = model.do_split(encoded_output,  prompt)
        assert cot.strip() == reference_output.split("|||")[1].replace("</think>", "").strip()
        assert answer.strip() == reference_output.split("|||")[2].strip()

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_do_split_DeepSeek_R1_Distill_Qwen_1_5B(self, mock_model, mock_config):
        reference_output = \
"""<｜begin▁of▁sentence｜><｜User｜>Question: A car travels 60 miles in 1.5 hours. What is its average speed?
Let's think step by step.<｜Assistant｜><think>|||
First, I need to calculate the average speed of the car. [...]
</think>|||
The answer is 40 mph.
"""
        model = CoTModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir=TEST_CACHE_DIR)
        prompt = reference_output.split("|||")[0]
        tidied_output = reference_output.replace("|||", "")
        encoded_output = model.get_utils().encode_to_tensor(tidied_output, to_device=torch.device("cpu"))
        (_, cot, answer) = model.do_split(encoded_output,  prompt)
        assert cot.strip() == reference_output.split("|||")[1].replace("</think>", "").strip()
        assert answer.strip() == reference_output.split("|||")[2].strip()

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_do_split_Gemma2_2B_it(self, mock_model, mock_config):
        question = "A car travels 60 miles in 1.5 hours. What is its average speed?"
        reference_output = \
"""<bos><start_of_turn>user
Question: A car travels 60 miles in 1.5 hours. What is its average speed?
Let's think step by step. Please write the string "Answer: " before the final answer.<end_of_turn>
<start_of_turn>model
|||Here's how to solve it step-by-step:

1. **Start with the numbers:** We have 2 + 2.
2. **Add them together:**  2 + 2 = 4. 


**Answer: 4**<eos>"""
        model = CoTModel("google/gemma-2-2b-it", cache_dir=TEST_CACHE_DIR)
        prompt = reference_output.split("|||")[0]
        cot_and_output = reference_output.split("|||")[1]
        tidied_output = reference_output.replace("|||", "")
        encoded_output = model.get_utils().encode_to_tensor(tidied_output, to_device=torch.device("cpu"))
        (_, cot, answer) = model.do_split(encoded_output,  prompt)
        assert cot.strip() == cot_and_output.split("Answer:", 1)[0].strip()
        assert answer.strip() == cot_and_output.split("Answer:", 1)[1].replace("<eos>", "").strip()

    @patch('model.AutoConfig.from_pretrained')
    @patch('model.AutoModelForCausalLM.from_pretrained')
    def test_split_logprobs_Qwen3_0_6B(self, mock_model, mock_config):
        # generated with do_sample=False, question="What is 2+2?"
        reference_output = \
"""<|im_start|>user
Question: What is 2+2?
Let's think step by step.<|im_end|>
<|im_start|>assistant
<think>
Okay, so the question is 2 plus 2. Let me think about how to approach this. First, I remember that when you add two numbers, you just add their values together. So 2 plus 2 would be 2 plus 2. Let me break it down. The first number is 2, and the second number is also 2. Adding them together should give me 4. But wait, maybe I should check if there's any trick here. Sometimes problems have hidden parts, like if they're asking for something else, but in this case, it's straightforward addition. Let me make sure I'm not missing anything. The question is simple, so no need for complex operations. So the answer should be 4. I think that's it.
</think>

2 + 2 equals 4. 

**Step-by-Step Explanation:**  
1. Start with the first number: 2.  
2. Add the second number: 2 + 2 = 4.  
3. The result is 4.  

Answer: 4.<|im_end|>"""
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        encoded_output = model.get_utils().encode_to_tensor(reference_output, to_device=torch.device("cpu"))
        (prompt, cot, answer) = model.do_split(encoded_output, reference_output.split("\nOkay")[0])

        assert prompt == "<|im_start|>user\nQuestion: What is 2+2?\nLet's think step by step.<|im_end|>\n<|im_start|>assistant"
        assert cot == "Okay, so the question is 2 plus 2. Let me think about how to approach this. First, I remember that when you add two numbers, you just add their values together. So 2 plus 2 would be 2 plus 2. Let me break it down. The first number is 2, and the second number is also 2. Adding them together should give me 4. But wait, maybe I should check if there's any trick here. Sometimes problems have hidden parts, like if they're asking for something else, but in this case, it's straightforward addition. Let me make sure I'm not missing anything. The question is simple, so no need for complex operations. So the answer should be 4. I think that's it."
        assert answer == "2 + 2 equals 4. \n\n**Step-by-Step Explanation:**  \n1. Start with the first number: 2.  \n2. Add the second number: 2 + 2 = 4.  \n3. The result is 4.  \n\nAnswer: 4.<|im_end|>"
        assert prompt + "\n<think>\n" + cot + "\n</think>\n\n" + answer == reference_output