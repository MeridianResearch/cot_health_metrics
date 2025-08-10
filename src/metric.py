import torch
from dataclasses import dataclass
from model import ModelResponse, Model

@dataclass
class SampleGroundTruth:
    cot: str
    answer: str

class Metric: 
    def __init__(self, metric_name: str, model: Model, alternative_model: Model | None = None):
        self.metric_name = metric_name
        self.model = model
        self.alternative_model = alternative_model

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        """Evaluate the metric based on the provided model response.
        Returns a numeric score: higher is more suspicious."""
        raise NotImplementedError("This method should be overridden " +
                                  "by subclasses")

    def evaluate_batch(self, responses: list[ModelResponse], ground_truth: list[SampleGroundTruth] | None = None):
        raise NotImplementedError("This method should be overridden " +
                                  "by subclasses")

    def __str__(self):
        return f"Metric(metric_name={self.metric_name}, model_name={self.model.model_name})"

class SingleMetric(Metric):
    def __init__(self, metric_name: str, model: Model, alternative_model: Model | None = None):
        super().__init__(metric_name, model, alternative_model)

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        raise NotImplementedError("This method should be overridden " +
                                  "by subclasses")

    def evaluate_batch(self, responses: list[ModelResponse], ground_truth: list[SampleGroundTruth] | None = None):
        scores = []
        for i, r in enumerate(responses):
            score, score_original, score_intervention = self.evaluate(r, ground_truth[i] if ground_truth else None)
            scores.append((score, score_original, score_intervention))
        return scores

class DummyMetric(Metric):
    def __init__(self, model: Model, alternative_model: Model | None = None):
        super().__init__("DummyMetric", model, alternative_model)
        
    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        """Always returns 0 (not suspicious)"""
        print(f"DummyMetric: model {self.model.model_name}")
        print(f"Prompt: {r.prompt.encode('unicode_escape').decode()}")
        print("\n")
        print("CoT: " + r.cot.encode('unicode_escape').decode())
        print("\n")
        print(f"Prediction: {r.answer.encode('unicode_escape').decode()}")
        print("\n")
        return (0, 0, 0)
