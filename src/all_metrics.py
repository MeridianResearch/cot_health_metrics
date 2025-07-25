from metric import Metric, DummyMetric
from metric_reliance import RelianceMetric
from metric_paraphrasability import ParaphrasabilityMetric
from metric_transferability import TransferabilityMetric
from metric_internalized import InternalizedMetric

METRICS = {
    "Dummy": DummyMetric,
    "Reliance": RelianceMetric,
    "Paraphrasability": ParaphrasabilityMetric,
    "Transferability": TransferabilityMetric,
    "Internalized": InternalizedMetric
}

def construct_metric(metric_name, model_name, alternative_model_name=None):
    metric_class = METRICS[metric_name]
    metric = metric_class(
        model_name=model_name,
        alternative_model_name=alternative_model_name)
    return metric