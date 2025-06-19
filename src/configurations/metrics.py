from src.configurations.enums import MetricName
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, TypeAlias
from functools import partial
from utilsforecast.losses import mase, msse, mae, mse


ForecastMetric: TypeAlias = Any


@dataclass(frozen=True)
class MetricSpec:
    """
    Describes how to calculate a metric.
    """

    factory: Callable[..., ForecastMetric]
    default_params: Dict[str, Any] = field(default_factory=dict)


METRIC_REGISTRY: dict[MetricName, MetricSpec] = {
    MetricName.MASE: MetricSpec(
        factory=lambda **p: partial(mase, **p), default_params={"seasonality": 7}
    ),
    MetricName.MSSE: MetricSpec(
        factory=lambda **p: partial(msse, **p), default_params={"seasonality": 7}
    ),
    MetricName.MAE: MetricSpec(
        factory=lambda **p: partial(mae, **p),
    ),
    MetricName.MSE: MetricSpec(factory=lambda **p: partial(mse, **p)),
}


@dataclass(frozen=True)
class MetricConfig:
    """
    A dataclass to store the metrics used for evaluation.
    """

    names: list[MetricName] = field(default_factory=list)
    seasonality: int = 7

    @property
    def metrics(self) -> Dict[MetricName, MetricSpec]:
        """
        Returns a dictionary of metric names and their corresponding MetricSpec.
        """

        metrics = {}
        for name in self.names:
            metric_spec = METRIC_REGISTRY[name]
            metrics[name] = metric_spec.factory(**metric_spec.default_params)

        return metrics
