import torch
from model import ModelResponse, Model

class Metric: 
    def __init__(self, metric_name: str, model: Model, alternative_model: Model | None = None):
        #print(f"Metric: {metric_name}")
        #print(f"Model: {model_name}")
        self.metric_name = metric_name
        self.model = model
        self.alternative_model = alternative_model

    def evaluate(self, r: ModelResponse):
        """Evaluate the metric based on the provided model response.
        Returns a numeric score: higher is more suspicious."""
        raise NotImplementedError("This method should be overridden " +
                                  "by subclasses")

    def __str__(self):
        return f"Metric(model_name={self.model.model_name})"

class DummyMetric(Metric):
    def __init__(self, model: Model, alternative_model: Model | None = None):
        super().__init__("DummyMetric", model, alternative_model)
        
    def evaluate(self, r: ModelResponse):
        """Always returns 0 (not suspicious)"""
        print(f"DummyMetric: model {self.model.model_name}")
        print(f"Prompt: {r.prompt.encode('unicode_escape').decode()}")
        print("\n")
        print("CoT: " + r.cot.encode('unicode_escape').decode())
        print("\n")
        print(f"Prediction: {r.answer.encode('unicode_escape').decode()}")
        print("\n")
        return (0, 0, 0)
