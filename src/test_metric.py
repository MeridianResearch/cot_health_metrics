import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import project modules
from metric import Metric, DummyMetric, MetricResult


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
        
        assert str(metric) == "Metric(metric_name=test_metric, model_name=test_model)"


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
        
        result = metric.evaluate(mock_response)
        assert result == MetricResult(0, 0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 