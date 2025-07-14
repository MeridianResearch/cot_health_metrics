class Metric: 
    def __init__(self, metric_name: str, model_name: str, alternative_model_name: str = None):
        self.metric_name = metric_name
        self.model_name = model_name
        self.alternative_model_name = alternative_model_name

    def evaluate(self, prompt: str, cot: str, prediction: str):
        """Evaluate the metric based on the provided prompt, chain of thought
        (cot), and prediction.
        Returns a numeric score: higher is more suspicious."""
        raise NotImplementedError("This method should be overridden " +
                                  "by subclasses")

    def __str__(self):
        return f"Metric(model_name={self.model_name})"

class DummyMetric:
    def __init__(self, metric_name: str, model_name: str, alternative_model_name: str = None):
        super().__init__("DummyMetric", model_name, alternative_model_name)
        
    def evaluate(self, prompt: str, cot: str, prediction: str):
        """Always returns 0 (not suspicious)"""
        print(f"DummyMetric: {self.model_name} with prompt: {prompt}, cot: {cot}, prediction: {prediction}")
        print(f"Prompt: {prompt}")
        print("\n")
        print(f"CoT: {cot}")
        print("\n")
        print(f"Prediction: {prediction}")
        print("\n")
        return 0