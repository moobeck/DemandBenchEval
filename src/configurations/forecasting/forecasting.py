from dataclasses import dataclass, field
from typing import List, Dict, Any
from ..utils.enums import ModelName, Framework, Frequency
from ..data.forecast_column import ForecastColumnConfig
from ..data.input_column import InputColumnConfig
from demandbench.datasets import Dataset
from .models.mixture import MixtureLossFactory
from .utils.quantile import QuantileLossFactory, QuantileUtils
from .models.model import MODEL_REGISTRY
from .quantile import QuantileConfig
from .models.model import ForecastModel

from neuralforecast.losses.pytorch import MAE, MQLoss

import logging


@dataclass(frozen=True)
class NeuralForecastConfig:
    """
    Configuration for neural forecasting models.
    """

    mixture: Dict[str, Any] = field(default_factory=dict)
    quantile: QuantileConfig = field(default_factory=QuantileConfig)
    gpus: int = 1
    cpus: int = 1
    num_samples: int = 1
    input_size: int = 1


@dataclass
class FoundationModelConfig:
    num_samples: int = 1
    quantile: QuantileConfig = field(default_factory=QuantileConfig)


@dataclass(frozen=True)
class StatisticalForecastConfig:
    quantile: QuantileConfig = field(default_factory=QuantileConfig)


@dataclass
class ForecastConfig:
    names: List[ModelName]
    freq: Frequency = Frequency.DAILY
    horizon: int = 14
    lags: List[int] = field(default_factory=list)
    model_config: Dict[Framework, Dict[str, Any]] = field(default_factory=dict)
    lags_config: Dict[str, Dict[str, int]] = field(default_factory=dict)
    columns_config: ForecastColumnConfig = field(default_factory=ForecastColumnConfig)

    @property
    def neural(self) -> NeuralForecastConfig:
        """
        Get the neural network configuration from the model_config.
        """
        neural_cfg = self.model_config.get(Framework.NEURAL, {})
        return NeuralForecastConfig(
            mixture=neural_cfg.get("mixture", {}),
            quantile=QuantileConfig(**neural_cfg.get("quantile", {})),
            gpus=neural_cfg.get("gpus", 1),
            cpus=neural_cfg.get("cpus", 1),
            num_samples=neural_cfg.get("num_samples", 1),
            input_size=len(self.lags),
        )

    @property
    def foundation(self) -> FoundationModelConfig:
        """
        Get the foundation model configuration from the model_config.
        """
        fm_cfg = self.model_config.get(Framework.FM, {})
        return FoundationModelConfig(
            num_samples=fm_cfg.get("num_samples", 1),
            quantile=QuantileConfig(**fm_cfg.get("quantile", {})),
        )

    @property
    def models(self) -> Dict[Framework, Dict[ModelName, ForecastModel]]:

        frameworks = {
            Framework.STATS: {},
            Framework.NEURAL: {},
            Framework.FM: {},
        }

        logging.info(f"Forecast models to be instantiated: {self.names}")

        for name in self.names:
            # map your config.ModelName to ModelKey
            key = ModelName(name.value)
            spec = MODEL_REGISTRY[key]

            logging.info(f"Instantiating model: {name} of framework {spec.framework}")

            # merge defaults with trainer-level params
            params = spec.default_params.copy()
            if spec.framework == Framework.STATS:
                params["season_length"] = Frequency.get_season_length(self.freq)
            elif spec.framework == Framework.NEURAL:
                params["h"] = self.horizon

                params["config"] = {
                    "stat_exog_list": self.columns_config.static,
                    "futr_exog_list": [
                        col
                        for col in self.columns_config.future_exogenous
                        if col not in self.columns_config.static
                    ],
                    "past_exog_list": [
                        col
                        for col in self.columns_config.past_exogenous
                        if col not in self.columns_config.static
                    ],
                    "input_size": self.neural.input_size,
                }

                mixture_config = self.neural.mixture
                quantile_config = self.neural.quantile
                params["gpus"] = self.neural.gpus
                params["cpus"] = self.neural.cpus
                params["num_samples"] = self.neural.num_samples

                if mixture_config:
                    loss_function = MixtureLossFactory.create_loss(mixture_config)
                    quantiles = QuantileUtils.create_quantiles(quantile_config)
                    loss_function = MQLoss(
                        level=QuantileUtils.quantiles_to_level(quantiles)
                    )
                    params["loss"] = loss_function
                elif quantile_config:
                    loss_function = QuantileLossFactory.create_loss(quantile_config)
                    params["loss"] = loss_function
                else:
                    params["loss"] = MAE()

            elif spec.framework == Framework.FM:

                params["stat_exog_list"] = self.columns_config.static
                params["futr_exog_list"] = self.columns_config.future_exogenous
                params["hist_exog_list"] = self.columns_config.past_exogenous

            model_instance = spec.factory(**params)

            if model_instance is not None:  # Skip unavailable models
                frameworks[spec.framework][name] = model_instance

        return frameworks

    def set_freq(self, dataset: Dataset, input_column: InputColumnConfig):
        """
        Set the frequency of the forecast configuration based on the dataset.
        """

        frequencies = dataset.features[input_column.frequency].unique()

        # check if any of the daily frequency identifiers are present
        if Frequency.get_alias(Frequency.DAILY, "demandbench") in frequencies:
            self.freq = Frequency.DAILY
        elif Frequency.get_alias(Frequency.WEEKLY, "demandbench") in frequencies:
            self.freq = Frequency.WEEKLY
        else:
            raise ValueError(
                f"Unsupported frequency found in the dataset: {frequencies}. "
                "Only 'daily' and 'weekly' frequencies are supported."
            )

    def set_lags(self, dataset_name: str = None):
        """
        Set the lags for the forecast configuration based on frequency and dataset.
        """
        if not self.lags:
            # Determine dataset name
            ds_name = dataset_name or (
                self.model_config.get("dataset_name") if self.model_config else None
            )
            lags_cfg = self.lags_config.get(ds_name, {}) if ds_name else {}

            if self.freq == Frequency.DAILY:
                n_lags = lags_cfg.get("daily", 50)
                self.lags = range(1, n_lags + 1)
            elif self.freq == Frequency.WEEKLY:
                n_lags = lags_cfg.get("weekly", 15)
                self.lags = range(1, n_lags + 1)
            else:
                raise ValueError(f"Unsupported frequency: {self.freq}")

            tabpfn_config = self.model_config.get(Framework.FM, {}).get("TABPFN")
            if tabpfn_config and "n_lags" in tabpfn_config:
                tabpfn_config["n_lags"] = len(self.lags)

    def set_horizon(self):
        """
        Set the horizon based on the frequency defined in the dataset.
        """
        if self.freq == Frequency.DAILY:
            self.horizon = 7
        elif self.freq == Frequency.WEEKLY:
            self.horizon = 4
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

    def set_columns(self, columns_config: ForecastColumnConfig):
        """
        Set the input columns for the forecast configuration based on the dataset.
        """
        self.columns_config = columns_config
