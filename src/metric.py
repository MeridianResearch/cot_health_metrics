import torch
from model import ModelResponse

class Metric: 
    def __init__(self, metric_name: str, model_name: str, alternative_model_name: str = None):
        #print(f"Metric: {metric_name}")
        #print(f"Model: {model_name}")
        self.metric_name = metric_name
        self.model_name = model_name
        self.alternative_model_name = alternative_model_name

    def evaluate(self, r: ModelResponse):
        """Evaluate the metric based on the provided model response.
        Returns a numeric score: higher is more suspicious."""
        raise NotImplementedError("This method should be overridden " +
                                  "by subclasses")

    def __str__(self):
        return f"Metric(model_name={self.model_name})"

class DummyMetric(Metric):
    def __init__(self, model_name: str, alternative_model_name: str = None):
        super().__init__("DummyMetric", model_name, alternative_model_name)
        
    def evaluate(self, r: ModelResponse):
        """Always returns 0 (not suspicious)"""
        print(f"DummyMetric: model {self.model_name}")
        print(f"Prompt: {r.prompt.encode('unicode_escape').decode()}")
        print("\n")
        print("CoT: " + r.cot.encode('unicode_escape').decode())
        print("\n")
        print(f"Prediction: {r.prediction.encode('unicode_escape').decode()}")
        print("\n")
        return 0
