from src.configurations.utils.enums import MetricName, Frequency
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, TypeAlias, Optional
from functools import partial
from utilsforecast.losses import mase, msse, mae, mse, rmse, scaled_mqloss
from src.configurations.model.quantile import QuantileUtils

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
    MetricName.RMSE: MetricSpec(factory=lambda **p: partial(rmse, **p)),
    MetricName.SCALED_MQLOSS: MetricSpec(
        factory=lambda **p: partial(scaled_mqloss, **p),
        default_params={"seasonality": 7, "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9]},
    ),
}


@dataclass
class MetricConfig:
    """
    A dataclass to store the metrics used for evaluation.
    """

    names: list[MetricName] = field(default_factory=list)
    seasonality: int = field(default=None, repr=False)
    quantiles: Optional[list[float]] = field(default=None, repr=False)
    metrics: Dict[MetricName, MetricSpec] = field(init=False)

    def __post_init__(self):
        self.seasonality_provided = self.seasonality is not None

    def set_seasonality(self, freq: Optional[Frequency] = None):
        """
        Sets the seasonality for the metrics based on the frequency of the dataset.
        """

        if freq == Frequency.DAILY:
            self.seasonality = 7
        elif freq == Frequency.WEEKLY:
            self.seasonality = 52
        elif freq == Frequency.MONTHLY:
            self.seasonality = 12
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

    def set_metrics(self):
        """
        Sets the metrics based on the names provided in the configuration.
        """
        metrics: Dict[MetricName, MetricSpec] = {}
        for name in self.names:
            spec = METRIC_REGISTRY[name]

            # Make a shallow copy of default_params so we don't mutate the registry
            params = spec.default_params.copy()

            # If seasonality is a supported param, override it
            if "seasonality" in params:
                params["seasonality"] = self.seasonality
            # If quantiles are provided, add them to the params
            if "quantiles" in params:
                params["quantiles"] = QuantileUtils.create_quantiles(self.quantiles)
            # Instantiate the metric with the (possibly overridden) params
            metrics[name] = spec.factory(**params)

            self.metrics = metrics
