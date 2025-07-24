from dataclasses import dataclass, field
from typing import Callable, Dict, Any, TypeAlias
import logging
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
try:
    from src.forecasting.toto_wrapper import TOTOWrapper
    TOTO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TOTO_AVAILABLE = False
    class TOTOWrapper:
        pass

from src.forecasting.tabpfn_wrapper import TabPFNWrapper
from .mixture import TGMM
from neuralforecast.losses.pytorch import MAE
ForecastModel: TypeAlias = Any


def create_loss(mixture_config: Dict[str, Any]) -> Any:
    """
    Create a loss function based on the mixture_config.

    Args:
        mixture_config (Dict[str, Any]): Configuration dictionary for the mixture model.

    Returns:
        An instance of a loss function (TGMM or MAE).

    Notes:
        - If mixture_config is empty or None, returns MAE loss.
        - If mixture_config contains TGMM settings, returns a TGMM instance.
    """
    logger = logging.getLogger(__name__)
    
    if not mixture_config:
        logger.info("No mixture configuration found, using default MAE loss")
        return MAE()
    
    if "TGMM" in mixture_config:
        tgmm_config = mixture_config["TGMM"]
        logger.info(f"Creating TGMM loss with configuration: {tgmm_config}")
        
        # Extract TGMM parameters with defaults
        n_components = tgmm_config.get("num_components", 1)
        horizon_correlation = tgmm_config.get("horizon_correlation", True)
        weighted = tgmm_config.get("weighted", True)
        
        # Create and return TGMM instance
        tgmm_loss = TGMM(
            n_components=n_components,
            horizon_correlation=horizon_correlation,
            weighted=weighted
        )
        logger.info(f"Successfully created TGMM loss with {n_components} components")
        return tgmm_loss
    else:
        # If mixture_config exists but doesn't contain TGMM, default to MAE
        logger.warning(f"Mixture configuration found but no supported mixture type: {list(mixture_config.keys())}. Using MAE loss.")
        return MAE()


def validate_mixture_config(mixture_config: Dict[str, Any]) -> bool:
    """
    Validate the mixture model configuration.
    
    Args:
        mixture_config: Dictionary containing mixture model configuration
        
    Returns:
        True if configuration is valid, False otherwise
        
    Raises:
        ValueError: If configuration contains invalid values
    """
    if not mixture_config:
        return True
        
    if "TGMM" in mixture_config:
        tgmm_config = mixture_config["TGMM"]
        
        # Validate num_components
        num_components = tgmm_config.get("num_components", 1)
        if not isinstance(num_components, int) or num_components < 1:
            raise ValueError("num_components must be a positive integer")
            
        # Validate boolean parameters
        for param in ["horizon_correlation", "weighted"]:
            if param in tgmm_config and not isinstance(tgmm_config[param], bool):
                raise ValueError(f"{param} must be a boolean value")
                
        return True
    
    # If we reach here, mixture_config has unsupported mixture types
    supported_types = ["TGMM"]
    available_types = list(mixture_config.keys())
    raise ValueError(
        f"Unsupported mixture type(s): {available_types}. "
        f"Supported types: {supported_types}"
    )


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
}


@dataclass
class ForecastConfig:
    names: List[ModelName]
    freq: Frequency = Frequency.DAILY
    horizon: int = 14
    lags: List[int] = field(default_factory=list)
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
                
                # Check for mixture model configuration in NEURAL framework
                neural_config = self.model_config.get(Framework.NEURAL, {})
                if "MIXTURE" in neural_config:
                    mixture_config = neural_config["MIXTURE"]
                    logger = logging.getLogger(__name__)
                    logger.info(f"Found MIXTURE configuration in NEURAL framework for model {key}: {mixture_config}")
                    
                    # Validate the mixture configuration
                    validate_mixture_config(mixture_config)
                    
                    loss_function = create_loss(mixture_config)
                    params["loss"] = loss_function
                    logger.info(f"Added mixture loss to {key} model parameters")
                
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

    def set_lags(self):
        """
        Set the lags for the forecast configuration the frequency defined in the dataset.
        """
        if not self.lags:
            # Use the default lags based on frequency
            if self.freq == Frequency.DAILY:
                self.lags = range(1, 50)
                
            elif self.freq == Frequency.WEEKLY:
                self.lags = range(1, 15)
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