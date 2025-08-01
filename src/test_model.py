import pytest
from unittest.mock import Mock, patch
from model import Model, CoTModel, ModelResponse

TEST_CACHE_DIR = "/tmp/cache-test"

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