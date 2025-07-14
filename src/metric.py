class Metric: 
    def __init__(self, model_name: str, alternative_model_name: str = None):
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
