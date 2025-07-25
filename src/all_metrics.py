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

def construct_metric(metric_name, model_name, alternative_model_name):
    construct_metric = METRICS[args.metric]
    metric = construct_metric(
        model_name=args.model,
        alternative_model_name=args.model2)