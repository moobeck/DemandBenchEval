from dataclasses import dataclass, field
from typing import Callable, Dict, Any, TypeAlias
from statsforecast.models import AutoARIMA, AutoTheta, AutoETS, AutoCES
from mlforecast.auto import AutoCatboost, AutoLightGBM, AutoRandomForest
from .enums import ModelName, Framework, Frequency
from .input_column import InputColumnConfig
from neuralforecast.auto import (
    AutoVanillaTransformer,
    AutoMLP,
    AutoLSTM,
    AutoTimesNet,
    AutoFEDformer,
    AutoNHITS,
    AutoTiDE,
)
from dataclasses import dataclass, field
from typing import List, Dict
from demandbench.datasets import Dataset
from src.forecasting.toto_wrapper import TOTOWrapper


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
    NEURAL = {"h": 14, "backend": "ray", "num_samples": 100}
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
        factory=lambda **p: TOTOWrapper(alias="Toto", **p),
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
}


@dataclass
class ForecastConfig:
    names: List[ModelName]
    freq: Frequency = Frequency.DAILY
    horizon: int = 14
    lags: List[int] = field(default_factory=list)
    date_features: List[str] = field(default_factory=list)
    model_config: Dict[Framework, Dict[str, Any]] = field(default_factory=dict)

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
            elif spec.framework == Framework.FM:
                if key == ModelName.TOTO:
                    params["num_samples"]  = self.model_config[Framework.FM]["TOTO"]["num_samples"]
            frameworks[spec.framework][name] = spec.factory(**params)

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
