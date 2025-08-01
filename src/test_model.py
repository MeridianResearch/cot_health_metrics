import torch
import pytest
from unittest.mock import Mock, patch
from model import Model, CoTModel, ModelResponse

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

# Fixtures for common test data
@pytest.fixture
def sample_model_response():
    """Fixture providing a sample ModelResponse"""
    return ModelResponse(
        question_id="test_001",
        question="What is 2+2?",
        prompt="Question: What is 2+2?\nLet's think step by step.\n<think>",
        cot="Let me think about this step by step. 2+2 equals 4.",
        answer="4",
        raw_output="<think>Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
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

class TestCoTModelReal:
    def test_make_prompt_Qwen3_0_6B(self):
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        prompt = model.make_prompt("test_001", "What is 2+2?")
        assert prompt == "<|im_start|>user\nQuestion: What is 2+2?\nLet's think step by step.<|im_end|>\n<|im_start|>assistant\n"

    def test_make_prompt_DeepSeek_R1_Distill_Qwen_1_5B(self):
        model = CoTModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir=TEST_CACHE_DIR)
        prompt = model.make_prompt("test_001", "What is 2+2?")
        assert prompt == "<｜begin▁of▁sentence｜><｜User｜>Question: What is 2+2?\nLet's think step by step.<｜Assistant｜><think>"

    def test_make_prompt_Gemma2_2B(self):
        model = CoTModel("google/gemma-2-2b", cache_dir=TEST_CACHE_DIR)
        prompt = model.make_prompt("test_001", "What is 2+2?")
        assert prompt == "<start_of_turn>user\nQuestion: What is 2+2?\nLet's think step by step.<end_of_turn>\n<start_of_turn>model\n"

    def test_tokenizer_decode_Qwen3_0_6B(self):
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)
        response = "Question: What is 2+2?\nLet's think step by step.<think>" \
            + "Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
        tokens = model.utils.encode_to_tensor(response)
        output = model.utils.decode_to_string(tokens[0])
        assert output == response

    def test_tokenizer_decode_Gemma2_2B(self):
        model = CoTModel("google/gemma-2-2b", cache_dir=TEST_CACHE_DIR)
        response = "Question: What is 2+2?\nLet's think step by step.<think>" \
            + "Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
        tokens = model.utils.encode_to_tensor(response)
        output = model.utils.decode_to_string(tokens[0])
        assert output == response

    def test_do_split_Qwen3_0_6B_not_enough_pieces(self):
        """Test do_split method"""
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)

        prompt = "Question: What is 2+2?\nLet's think step by step.\n<think>"
        response = prompt + "Let me think about this step by step..."
        
        with pytest.raises(RuntimeError):
            model_response = model.evaluate_cot_response(1, prompt, response)

    def test_do_split_Qwen3_0_6B(self):
        """Test do_split method"""
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)

        prompt = "Question: What is 2+2?\nLet's think step by step.\n<think>"
        response = prompt + "Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
        model_response = model.evaluate_cot_response(1, prompt, response)

        assert model_response.question == "Question: What is 2+2?\nLet's think step by step."
        assert model_response.cot == "Let me think about this step by step. 2+2 equals 4."
        assert model_response.answer == "Answer: 4"
        assert model_response.raw_output == response

    @pytest.mark.xfail(reason="Test is expected to fail due to model loading issues")
    def test_do_split_Gemma2_2B(self):
        """Test do_split method"""
        model = CoTModel("google/gemma-2-2b", cache_dir=TEST_CACHE_DIR)

        prompt = "Question: What is 2+2?\nLet's think step by step."
        response = prompt #+ "Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
        model_response = model.evaluate_cot_response(1, prompt, response)

        assert model_response.question == "<bos>Question: What is 2+2?\nLet's think step by step."
        assert model_response.cot == "Let me think about this step by step. 2+2 equals 4."
        assert model_response.answer == "4"
        assert model_response.raw_output == response

    def test_real_example_DeepSeek_R1_Distill_Qwen_1_5B(self):
        input = ModelResponse(
            question_id=0,
            question='A car travels 60 miles in 1.5 hours. What is its average speed?',
            prompt="<｜begin▁of▁sentence｜><｜User｜>Question: A car travels 60 miles in 1.5 hours. What is its average speed?\nLet's think step by step.<｜Assistant｜><think>",
            cot="First, I need to determine the average speed of the car. Average speed is calculated by dividing the total distance traveled by the total time taken.\n\nThe car traveled 60 miles in 1.5 hours. To find the average speed, I'll divide 60 miles by 1.5 hours.\n\n60 divided by 1.5 equals 40.\n\nTherefore, the car's average speed is 40 miles per hour.",
            answer="**Solution:**\n\nTo determine the average speed of the car, we use the formula:\n\n\\[\n\\text{Average Speed} = \\frac{\\text{Total Distance}}{\\text{Total Time}}\n\\]\n\nGiven:\n- **Total Distance** = 60 miles\n- **Total Time** = 1.5 hours\n\nPlugging in the values:\n\n\\[\n\\text{Average Speed} = \\frac{60 \\text{ miles}}{1.5 \\text{ hours}} = 40 \\text{ miles per hour}\n\\]\n\n**Answer:**  \nThe car's average speed is \\(\\boxed{40}\\) miles per hour.",
            raw_output="<｜User｜>Question: A car travels 60 miles in 1.5 hours. What is its average speed?\nLet's think step by step.<｜Assistant｜><think>\nFirst, I need to determine the average speed of the car. Average speed is calculated by dividing the total distance traveled by the total time taken.\n\nThe car traveled 60 miles in 1.5 hours. To find the average speed, I'll divide 60 miles by 1.5 hours.\n\n60 divided by 1.5 equals 40.\n\nTherefore, the car's average speed is 40 miles per hour.\n</think>\n\n**Solution:**\n\nTo determine the average speed of the car, we use the formula:\n\n\\[\n\\text{Average Speed} = \\frac{\\text{Total Distance}}{\\text{Total Time}}\n\\]\n\nGiven:\n- **Total Distance** = 60 miles\n- **Total Time** = 1.5 hours\n\nPlugging in the values:\n\n\\[\n\\text{Average Speed} = \\frac{60 \\text{ miles}}{1.5 \\text{ hours}} = 40 \\text{ miles per hour}\n\\]\n\n**Answer:**  \nThe car's average speed is \\(\\boxed{40}\\) miles per hour."
        )

        model = CoTModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir=TEST_CACHE_DIR)
        model_response = model.evaluate_cot_response(1, input.prompt, input.raw_output)

        assert model_response.prompt == input.prompt
        assert model_response.cot == input.cot
        assert model_response.answer == input.answer


class TestCoTModelGenerate:
    live: bool = True

    def test_generate_cot_response_full_Qwen3_0_6B(self):
        question = "What is 2+2?"
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
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR) if self.live \
            else MockModel(model_name="Qwen/Qwen3-0.6B", model_response=reference_output)
        model_response = model.generate_cot_response_full(1, question, do_sample=False)
        assert model_response.question == question
        assert model_response.prompt == reference_output.split("|||")[0].replace("<think>", "")
        assert model_response.cot == reference_output.split("|||")[1].replace("</think>", "")
        assert model_response.answer == reference_output.split("|||")[2]
        assert model_response.raw_output == reference_output.replace("|||", "")