from dataclasses import dataclass, field
from typing import Callable, Dict, Any, TypeAlias
import logging
from statsforecast.models import AutoARIMA, AutoTheta, AutoETS, AutoCES
from mlforecast.auto import AutoCatboost, AutoLightGBM, AutoRandomForest
from .enums import ModelName, Framework, Frequency
from .forecast_column import ForecastColumnConfig
from .input_column import InputColumnConfig
from neuralforecast.auto import (
    AutoVanillaTransformer,
    AutoMLP,
    AutoLSTM,
    AutoTimesNet,
    AutoFEDformer,
    AutoNHITS,
    AutoTiDE,
    AutoDeepAR,
    AutoBiTCN,
)
from dataclasses import dataclass, field
from typing import List, Dict, Any
from demandbench.datasets import Dataset
try:
    from src.forecasting.toto_wrapper import TOTOWrapper
    TOTO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TOTO_AVAILABLE = False
    class TOTOWrapper:
        pass

from src.forecasting.tabpfn_wrapper import TabPFNWrapper
from .mixture import MixtureLossFactory
from neuralforecast.losses.pytorch import MAE
import os
ForecastModel: TypeAlias = Any

@dataclass(frozen=True)
class ModelSpec:
    """
    Describes how to build a model and which framework it belongs to.
    """

    factory: Callable[..., ForecastModel]
    framework: Framework
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_space: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DefaultParams:

    STATS = {}
    ML = {}
    NEURAL = {}
    FM = {}


MODEL_REGISTRY: dict[ModelName, ModelSpec] = {
    ModelName.ARIMA: ModelSpec(
        factory=lambda **p: AutoARIMA(alias="arima", **p),
        framework=Framework.STATS,
        default_params=DefaultParams.STATS,
    ),
    ModelName.THETA: ModelSpec(
        factory=lambda **p: AutoTheta(alias="theta", **p),
        framework=Framework.STATS,
        default_params=DefaultParams.STATS,
    ),
    ModelName.ETS: ModelSpec(
        factory=lambda **p: AutoETS(alias="ets", **p),
        framework=Framework.STATS,
        default_params=DefaultParams.STATS,
    ),
    ModelName.CES: ModelSpec(
        factory=lambda **p: AutoCES(alias="ces", **p),
        framework=Framework.STATS,
        default_params=DefaultParams.STATS,
    ),
    ModelName.LGBM: ModelSpec(
        factory=lambda **p: AutoLightGBM(**p),
        framework=Framework.ML,
        default_params=DefaultParams.ML,
    ),
    ModelName.CATBOOST: ModelSpec(
        factory=lambda **p: AutoCatboost(**p),
        framework=Framework.ML,
        default_params=DefaultParams.ML,
    ),
    ModelName.RF: ModelSpec(
        factory=lambda **p: AutoRandomForest(**p),
        framework=Framework.ML,
        default_params=DefaultParams.ML,
    ),
    ModelName.TOTO: ModelSpec(
        factory=lambda **p: TOTOWrapper(alias="Toto", **p) if TOTO_AVAILABLE else None,
        framework=Framework.FM,
        default_params=DefaultParams.FM,
    ),
    ModelName.TABPFN: ModelSpec(
        factory=lambda **p: TabPFNWrapper(alias="TabPFN", **p),
        framework=Framework.FM,
        default_params=DefaultParams.FM,
    ),
    ModelName.TRANSFORMER: ModelSpec(
        factory=lambda **p: AutoVanillaTransformer(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
    ModelName.MLP: ModelSpec(
        factory=lambda **p: AutoMLP(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
    ModelName.LSTM: ModelSpec(
        factory=lambda **p: AutoLSTM(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
    ModelName.TIMESNET: ModelSpec(
        factory=lambda **p: AutoTimesNet(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
    ModelName.FEDFORMER: ModelSpec(
        factory=lambda **p: AutoFEDformer(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
    ModelName.TIDE: ModelSpec(
        factory=lambda **p: AutoTiDE(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
    ModelName.NHITS: ModelSpec(
        factory=lambda **p: AutoNHITS(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
    ModelName.DEEPAR: ModelSpec(
        factory=lambda **p: AutoDeepAR(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
    ModelName.BITCN: ModelSpec(
        factory=lambda **p: AutoBiTCN(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
    ),
}



@dataclass(frozen=True)
class NeuralForecastConfig:
    """
    Configuration for neural forecasting models.
    """
    mixture: Dict[str, Any] = field(default_factory=dict)
    gpus: int = 1
    cpus: int = 1
    num_samples: int = 1
    input_size: int = 1


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
    def neuralconfig(self) -> NeuralForecastConfig:
        """
        Get the neural network configuration from the model_config.
        """
        neural_cfg = self.model_config.get(Framework.NEURAL, {})
        return NeuralForecastConfig(
            mixture=neural_cfg.get("mixture", {}),
            gpus=neural_cfg.get("gpus", 1),
            cpus=neural_cfg.get("cpus", 1),
            num_samples=neural_cfg.get("num_samples", 1),
            input_size=len(self.lags)
        )


    @property
    def models(self) -> Dict[Framework, Dict[ModelName, ForecastModel]]:

        frameworks = {
            Framework.STATS: {},
            Framework.ML: {},
            Framework.NEURAL: {},
            Framework.FM: {},
        }

        for name in self.names:
            # map your config.ModelName to ModelKey
            key = ModelName(name.value)
            spec = MODEL_REGISTRY[key]

            # merge defaults with trainer-level params
            params = spec.default_params.copy()
            if spec.framework == Framework.STATS:
                params["season_length"] = Frequency.get_season_length(self.freq)
            elif spec.framework == Framework.NEURAL:
                params["h"] = self.horizon
                
                params["config"] = {
                    "stat_exog_list": self.columns_config.static,
                    "future_exog_list": [col for col in self.columns_config.exogenous if col not in self.columns_config.static],
                }

                mixture_config = self.neuralconfig.mixture
                params["gpus"] = self.neuralconfig.gpus
                params["cpus"] = self.neuralconfig.cpus
                params["num_samples"] = self.neuralconfig.num_samples
                if mixture_config:
                    loss_function = MixtureLossFactory.create_loss(mixture_config)
                    params["loss"] = loss_function

            elif spec.framework == Framework.FM:
                if key == ModelName.TOTO and TOTO_AVAILABLE:
                    params.update(self.model_config.get(Framework.FM, {}).get("TOTO", {}))
                elif key == ModelName.TABPFN:
                    params.update(self.model_config.get(Framework.FM, {}).get("TABPFN", {}))

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
            ds_name = dataset_name or (self.model_config.get("dataset_name") if self.model_config else None)
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



        