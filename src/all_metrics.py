from metric import Metric, DummyMetric
from metric_reliance import RelianceMetric
from metric_paraphrasability_simple import ParaphrasabilityMetricSimple
from metric_paraphrasability import ParaphrasabilityMetric
from metric_prompt_paraphrasability import PromptParaphrasabilityMetric
from metric_transferability import TransferabilityMetric
from metric_internalized import InternalizedMetric

METRICS: dict[str, type[Metric]] = {
    "Dummy": DummyMetric,
    "Reliance": RelianceMetric,
    "ParaphrasabilitySimple": ParaphrasabilityMetricSimple,
    "Paraphrasability": ParaphrasabilityMetric,
    "PromptParaphrasability": PromptParaphrasabilityMetric,
    "Transferability": TransferabilityMetric,
    "Internalized": InternalizedMetric
}

def construct_metric(metric_name, model, alternative_model, **kwargs):
    construct_metric = METRICS[metric_name]
    metric = construct_metric(
        model=model,
        alternative_model=alternative_model,
        **kwargs
    )
    return metric
