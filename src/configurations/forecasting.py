from dataclasses import dataclass, field    
from typing import Callable, Dict, Any, TypeAlias
from statsforecast.models import AutoARIMA, AutoTheta, AutoETS
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from configurations.enums import ModelName, Framework


ForecastModel: TypeAlias = Any



@dataclass(frozen=True)
class ModelSpec:
    """
    Describes how to build a model and which framework it belongs to.
    """
    factory: Callable[..., ForecastModel]
    framework: Framework
    default_params: Dict[str, Any] = field(default_factory=dict)



MODEL_REGISTRY: dict[ModelName, ModelSpec] = {
    ModelName.ARIMA: ModelSpec(
        factory=lambda **p: AutoARIMA(alias="arima", **p),
        framework=Framework.STATS,
        default_params={"season_length": 7}
    ),
    ModelName.THETA: ModelSpec(
        factory=lambda **p: AutoTheta(alias="theta", **p),
        framework=Framework.STATS,
        default_params={"season_length": 7}
    ),
    ModelName.ETS: ModelSpec(
        factory=lambda **p: AutoETS(alias="ets", **p),
        framework=Framework.STATS,
        default_params={"season_length": 7}
    ),
    ModelName.LGBM: ModelSpec(
        factory=lambda **p: LGBMRegressor(**p),
        framework=Framework.ML,
        default_params={}  # no seasonality param
    ),
    ModelName.RF: ModelSpec(
        factory=lambda **p: RandomForestRegressor(**p),
        framework=Framework.ML,
        default_params={} # no seasonality param
    ),
}

# trainer.py
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass(frozen=True)
class ForecastConfig:
    names: List[ModelName]
    freq: str = "D"        
    season_length: int = 7
    horizon: int = 14
    lags: List[int] = field(default_factory=list)
    date_features: List[str] = field(default_factory=list)

    @property
    def models(self) -> Dict[Framework, Dict[ModelName, ForecastModel]]:

        frameworks = {Framework.STATS: {}, Framework.ML: {}}

        for name in self.names:
            # map your config.ModelName to ModelKey
            key = ModelName(name.value)
            spec = MODEL_REGISTRY[key]

            # merge defaults with trainer-level params
            params = spec.default_params.copy()
            # only ML models need lags/date_features
            if spec.framework == Framework.STATS :
                params["season_length"] = self.season_length

            # instantiate
            frameworks[spec.framework][name] = spec.factory(**params)

        return frameworks

        