import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import project modules
from metric import Metric, DummyMetric


class TestMetric:
    """Test cases for base Metric class"""
    
    def test_metric_initialization(self):
        """Test Metric initialization"""
        mock_model = Mock()
        mock_alternative_model = Mock()
        
        metric = Metric("test_metric", mock_model, mock_alternative_model)
        assert metric.metric_name == "test_metric"
        assert metric.model == mock_model
        assert metric.alternative_model == mock_alternative_model
    
    def test_evaluate_not_implemented(self):
        """Test that evaluate raises NotImplementedError"""
        mock_model = Mock()
        metric = Metric("test_metric", mock_model)
        
        with pytest.raises(NotImplementedError):
            metric.evaluate(Mock())
    
    def test_str_representation(self):
        """Test string representation of Metric"""
        mock_model = Mock()
        mock_model.model_name = "test_model"
        metric = Metric("test_metric", mock_model)
        
        assert str(metric) == "Metric(model_name=test_model)"


class TestDummyMetric:
    """Test cases for DummyMetric class"""
    
    def test_dummy_metric_initialization(self):
        """Test DummyMetric initialization"""
        mock_model = Mock()
        metric = DummyMetric(mock_model)
        
        assert metric.metric_name == "DummyMetric"
        assert metric.model == mock_model
    
    def test_dummy_metric_evaluate(self):
        """Test DummyMetric evaluate method"""
        mock_model = Mock()
        mock_model.model_name = "test_model"
        metric = DummyMetric(mock_model)
        
        mock_response = Mock()
        mock_response.prompt = "Test prompt"
        mock_response.cot = "Test reasoning"
        mock_response.answer = "Test answer"
        
        # Should return 0 and print debug info
        result = metric.evaluate(mock_response)
        assert result == 0


# Fixtures for metric tests
@pytest.fixture
def mock_model():
    """Fixture providing a mock model for metric testing"""
    model = Mock()
    model.model_name = "test_model"
    return model


@pytest.fixture
def sample_model_response():
    """Fixture providing a sample ModelResponse for metric testing"""
    from model import ModelResponse
    return ModelResponse(
        question_id="test_001",
        question="What is 2+2?",
        prompt="Question: What is 2+2?\nLet's think step by step.\n<think>",
        cot="Let me think about this step by step. 2+2 equals 4.",
        answer="4",
        raw_output="<think>Let me think about this step by step. 2+2 equals 4.</think>\nAnswer: 4"
    )


@pytest.mark.parametrize("metric_class,expected_score", [
    (DummyMetric, 0),
])
def test_metric_scores(mock_model, sample_model_response, metric_class, expected_score):
    """Test that metrics return expected scores"""
    metric = metric_class(mock_model)
    score = metric.evaluate(sample_model_response)
    assert score == expected_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 