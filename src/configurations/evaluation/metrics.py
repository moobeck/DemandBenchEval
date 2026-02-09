from src.configurations.utils.enums import MetricName, FrequencyType
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, TypeAlias, Optional
from functools import partial
from utilsforecast.losses import mase, msse, mae, mse, rmse, scaled_mqloss, rmsse, scaled_quantile_loss
from src.configurations.evaluation.utils import spec, scaled_spec, apis, sapis, scaled_bias, scaled_mae, scaled_rmse, sabs_bias
from src.configurations.forecasting.quantile import QuantileConfig
from src.configurations.forecasting.utils.quantile import QuantileUtils

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
    MetricName.RMSSE: MetricSpec(
        factory=lambda **p: partial(rmsse, **p), default_params={"seasonality": 7}
    ),
    MetricName.MAE: MetricSpec(
        factory=lambda **p: partial(mae, **p),
    ),
    MetricName.MSE: MetricSpec(factory=lambda **p: partial(mse, **p)),
    MetricName.RMSE: MetricSpec(factory=lambda **p: partial(rmse, **p)),
    MetricName.SMQL: MetricSpec(
        factory=lambda **p: partial(scaled_mqloss, **p),
        default_params={"seasonality": 7, "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9]},
    ),
    MetricName.SPEC: MetricSpec(factory=lambda **p: partial(spec, **p)),
    MetricName.SSPEC: MetricSpec(
        factory=lambda **p: partial(scaled_spec, **p),
    ),
    MetricName.APIS: MetricSpec(
        factory=lambda **p: partial(apis, **p),
    ),
    MetricName.SAPIS: MetricSpec(
        factory=lambda **p: partial(sapis, **p),
    ),
    MetricName.SMAE: MetricSpec(
        factory=lambda **p: partial(scaled_mae, **p),
    ),
    MetricName.SRMSE: MetricSpec(
        factory=lambda **p: partial(scaled_rmse, **p),
    ),
    MetricName.SBIAS: MetricSpec(
        factory=lambda **p: partial(scaled_bias, **p),
    ),
    MetricName.SABSBIAS: MetricSpec(
        factory=lambda **p: partial(sabs_bias, **p),
    ),
    MetricName.SQL_60: MetricSpec(
        factory=lambda **p: partial(scaled_quantile_loss, q=0.6, **p),
        default_params={"seasonality": 7},
    ),
    MetricName.SQL_70: MetricSpec(
        factory=lambda **p: partial(scaled_quantile_loss, q=0.7, **p),
        default_params={"seasonality": 7},
    ),
    MetricName.SQL_80: MetricSpec(
        factory=lambda **p: partial(scaled_quantile_loss, q=0.8, **p),
        default_params={"seasonality": 7},
    ),
    MetricName.SQL_90: MetricSpec(
        factory=lambda **p: partial(scaled_quantile_loss, q=0.9, **p),
        default_params={"seasonality": 7},
    ),


}

PROBABILISTIC_METRICS = {
    MetricName.SMQL,
    MetricName.SQL_60,
    MetricName.SQL_70,
    MetricName.SQL_80,
    MetricName.SQL_90,
}


@dataclass
class MetricConfig:
    """
    A dataclass to store the metrics used for evaluation.
    """

    names: list[MetricName] = field(default_factory=list)
    seasonality: Optional[int] = field(default=None, repr=False)
    quantiles: Optional[QuantileConfig] = field(default=None, repr=False)
    metrics: Dict[MetricName, MetricSpec] = field(init=False)

    @property
    def contains_probabilistic(self) -> bool:
        """
        Checks if the metric configuration contains probabilistic metrics.
        """
        return any(name in PROBABILISTIC_METRICS for name in self.names)

    def set_seasonality(self, freq: Optional[FrequencyType] = None):
        """
        Sets the seasonality for the metrics based on the frequency of the dataset.
        """

        if freq == FrequencyType.DAILY:
            self.seasonality = 7
        elif freq == FrequencyType.WEEKLY:
            self.seasonality = 52
        elif freq == FrequencyType.MONTHLY:
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
