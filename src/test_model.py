import pytest
from unittest.mock import Mock, patch
from model import Model, CoTModel, ModelResponse

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
        model = Model("test_model", "/tmp/cache")
        assert model.model_name == "test_model"
        assert model.cache_dir == "/tmp/cache"

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

    def test_do_split(self):
        """Test do_split method"""
        model = CoTModel("Qwen/Qwen3-0.6B", cache_dir="/tmp/cache-test")

        prompt = "Question: What is 2+2?\nLet's think step by step.\n<think>" \
            + "Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
        response = model.evaluate_cot_response(1, prompt)

        assert response.question == "Question: What is 2+2?\nLet's think step by step."
        assert response.cot == "Let me think about this step by step. 2+2 equals 4."
        assert response.answer == "Answer: 4"
        assert response.raw_output == prompt