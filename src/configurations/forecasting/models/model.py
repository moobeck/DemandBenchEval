from dataclasses import dataclass, field
from typing import Callable, Dict, Any, TypeAlias
from statsforecast.models import AutoARIMA, AutoTheta, AutoETS, AutoCES
from ...utils.enums import ModelName, Framework
from neuralforecast.auto import (
    AutoVanillaTransformer,
    AutoMLP,
    AutoLSTM,
    AutoTimesNet,
    AutoFEDformer,
    AutoNHITS,
    AutoTiDE,
    AutoDeepAR,
    AutoNBEATSx,
    AutoBiTCN,
    AutoGRU,
    AutoTFT,
    AutoTCN,
    AutoPatchTST,
    AutoxLSTM,
)
from src.forecasting.models.foundation import Moirai, Chronos, TabPFN
from dataclasses import dataclass, field
from typing import Dict, Any
from optuna.samplers import TPESampler


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
    model: ForecastModel = None


@dataclass(frozen=True)
class DefaultParams:

    STATS = {}
    NEURAL = {"backend": "optuna", "search_alg": TPESampler()}
    FM = {}


MODEL_REGISTRY: dict[ModelName, ModelSpec] = {
    ModelName.ARIMA: ModelSpec(
        factory=lambda **p: AutoARIMA(alias="arima", **p),
        framework=Framework.STATS,
        default_params=DefaultParams.STATS,
        model=AutoARIMA,
    ),
    ModelName.THETA: ModelSpec(
        factory=lambda **p: AutoTheta(alias="theta", **p),
        framework=Framework.STATS,
        default_params=DefaultParams.STATS,
        model=AutoTheta,
    ),
    ModelName.ETS: ModelSpec(
        factory=lambda **p: AutoETS(alias="ets", **p),
        framework=Framework.STATS,
        default_params=DefaultParams.STATS,
        model=AutoETS,
    ),
    ModelName.CES: ModelSpec(
        factory=lambda **p: AutoCES(alias="ces", **p),
        framework=Framework.STATS,
        default_params=DefaultParams.STATS,
        model=AutoCES,
    ),
    ModelName.TRANSFORMER: ModelSpec(
        factory=lambda **p: AutoVanillaTransformer(**p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoVanillaTransformer,
    ),
    ModelName.MLP: ModelSpec(
        factory=lambda **p: AutoMLP(alias="mlp", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoMLP,
    ),
    ModelName.LSTM: ModelSpec(
        factory=lambda **p: AutoLSTM(alias="lstm", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoLSTM,
    ),
    ModelName.TIMESNET: ModelSpec(
        factory=lambda **p: AutoTimesNet(alias="timesnet", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoTimesNet,
    ),
    ModelName.FEDFORMER: ModelSpec(
        factory=lambda **p: AutoFEDformer(alias="fedformer", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoFEDformer,
    ),
    ModelName.TIDE: ModelSpec(
        factory=lambda **p: AutoTiDE(alias="tide", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoTiDE,
    ),
    ModelName.NHITS: ModelSpec(
        factory=lambda **p: AutoNHITS(alias="nhits", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoNHITS,
    ),
    ModelName.NBEATS: ModelSpec(
        factory=lambda **p: AutoNBEATSx(alias="nbeats", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoNBEATSx,
    ),
    ModelName.DEEPAR: ModelSpec(
        factory=lambda **p: AutoDeepAR(alias="deepar", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoDeepAR,
    ),
    ModelName.BITCN: ModelSpec(
        factory=lambda **p: AutoBiTCN(alias="bitcn", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoBiTCN,
    ),
    ModelName.GRU: ModelSpec(
        factory=lambda **p: AutoGRU(alias="gru", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoGRU,
    ),
    ModelName.TCN: ModelSpec(
        factory=lambda **p: AutoTCN(alias="tcn", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoTCN,
    ),
    ModelName.TFT: ModelSpec(
        factory=lambda **p: AutoTFT(alias="tft", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoTFT,
    ),
    ModelName.PATCHTST: ModelSpec(
        factory=lambda **p: AutoPatchTST(alias="patchtst", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoPatchTST,
    ),
    ModelName.XLSTM: ModelSpec(
        factory=lambda **p: AutoxLSTM(alias="xlstm", **p),
        framework=Framework.NEURAL,
        default_params=DefaultParams.NEURAL,
        model=AutoxLSTM,
    ),
    ModelName.MOIRAI: ModelSpec(
        factory=lambda **p: Moirai(**p),
        framework=Framework.FM,
        default_params=DefaultParams.FM,
        model=Moirai,
    ),
    ModelName.CHRONOS: ModelSpec(
        factory=lambda **p: Chronos(**p),
        framework=Framework.FM,
        default_params=DefaultParams.FM,
        model=Chronos,
    ),
    ModelName.TABPFN: ModelSpec(
        factory=lambda **p: TabPFN(**p),
        framework=Framework.FM,
        default_params=DefaultParams.FM,
        model=TabPFN,
    ),
}
