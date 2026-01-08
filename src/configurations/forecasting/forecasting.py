import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch
from ..utils.enums import ModelName, Framework, FrequencyType
from ..data.forecast_column import ForecastColumnConfig
from .models.mixture import MixtureLossFactory
from .utils.quantile import QuantileLossFactory, QuantileUtils
from .models.model import MODEL_REGISTRY
from .quantile import QuantileConfig, DEFAULT_QUANTILE_CONFIG
from .models.model import ForecastModel
from neuralforecast.auto import AutoxLSTM

from neuralforecast.losses.pytorch import MAE, MQLoss
from ray import tune

import logging
from src.forecasting.utils.optuna_logging import make_trial_logger


@dataclass(frozen=True)
class NeuralForecastConfig:
    """
    Configuration for neural forecasting models.
    """

    mixture: Dict[str, Any] = field(default_factory=dict)
    quantile: QuantileConfig = field(default=DEFAULT_QUANTILE_CONFIG)
    gpus: int = 1
    cpus: int = 1
    num_samples: int = 1


@dataclass
class FoundationModelConfig:
    num_samples: int = 1
    quantile: QuantileConfig = field(default=DEFAULT_QUANTILE_CONFIG)


@dataclass(frozen=True)
class StatisticalForecastConfig:
    quantile: QuantileConfig = field(default=DEFAULT_QUANTILE_CONFIG)


@dataclass
class ForecastConfig:
    names: List[ModelName]
    freq: FrequencyType = FrequencyType.DAILY
    horizon: int = 14
    model_config: Dict[Framework, Dict[str, Any]] = field(default_factory=dict)
    columns_config: ForecastColumnConfig = field(default_factory=ForecastColumnConfig)

    @property
    def neural(self) -> NeuralForecastConfig:
        """
        Get the neural network configuration from the model_config.
        """
        neural_cfg = self.model_config.get(Framework.NEURAL, {})
        return NeuralForecastConfig(
            mixture=neural_cfg.get("mixture", {}),
            quantile=DEFAULT_QUANTILE_CONFIG,
            gpus=neural_cfg.get("gpus", 1),
            cpus=neural_cfg.get("cpus", 1),
            num_samples=neural_cfg.get("num_samples", 1),
        )

    @property
    def foundation(self) -> FoundationModelConfig:
        """
        Get the foundation model configuration from the model_config.
        """
        fm_cfg = self.model_config.get(Framework.FM, {})
        return FoundationModelConfig(
            num_samples=fm_cfg.get("num_samples", 1),
            quantile=DEFAULT_QUANTILE_CONFIG,
        )
    
    @property
    def statistical(self) -> StatisticalForecastConfig:
        """
        Get the statistical model configuration from the model_config.
        """
        stats_cfg = self.model_config.get(Framework.STATS, {})
        return StatisticalForecastConfig(
            quantile=DEFAULT_QUANTILE_CONFIG,
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
            # Optional: log Optuna trials for inspection by setting OPTUNA_LOG_DIR
            optuna_log_dir = os.getenv("OPTUNA_LOG_DIR")
            if optuna_log_dir and params.get("backend") == "optuna":
                log_path = os.path.join(optuna_log_dir, f"{name.value}.jsonl")
                callbacks = params.get("callbacks", [])
                callbacks.append(make_trial_logger(log_path))
                params["callbacks"] = callbacks
            if spec.framework == Framework.STATS:
                params["season_length"] = FrequencyType.get_season_length(self.freq)
            elif spec.framework == Framework.NEURAL:
                params["h"] = self.horizon

                config = spec.model.get_default_config(
                    h=self.horizon, backend="not_specified"
                )

                config.update({
                    "stat_exog_list": self.columns_config.static,
                    "futr_exog_list": [
                        col
                        for col in self.columns_config.future_exogenous
                        if col not in self.columns_config.static
                    ],
                    "hist_exog_list": [
                        col
                        for col in self.columns_config.past_exogenous
                        if col not in self.columns_config.static
                    ],
                })

                backend = params.get("backend")

                if backend == "optuna":
                    # Short-series-friendly search spaces.
                    horizon = self.horizon
                    safe_input_common = sorted(
                        {
                            max(4, horizon // 2),
                            max(6, horizon),
                            min(14, 2 * horizon),
                        }
                    )
                    max_step_common = max(1, min(horizon // 2, 3))

                    if spec.model is AutoxLSTM:
                        max_step_xlstm = max(1, min(horizon // 2, 4))
                        config.update(
                            {
                                "input_size": tune.choice(safe_input_common),
                                "step_size": tune.randint(1, max_step_xlstm + 1),
                                "encoder_hidden_size": tune.choice([32, 64]),
                                "decoder_hidden_size": tune.choice([32, 64]),
                                "encoder_dropout": tune.uniform(0.0, 0.3),
                                "learning_rate": tune.loguniform(1e-4, 1e-2),
                                "encoder_n_blocks": tune.randint(1, 3),
                                "max_steps": tune.choice([200, 400]),
                                "batch_size": tune.choice([32, 64]),
                                "windows_batch_size": tune.choice([128, 256]),
                                "scaler_type": tune.choice(["standard", "robust"]),
                                "start_padding_enabled": True,
                            }
                        )
                    else:
                        if "input_size" in config:
                            config["input_size"] = tune.choice(safe_input_common)
                        if "step_size" in config:
                            config["step_size"] = tune.randint(
                                1, max_step_common + 1
                            )
                        # Force padding even if the default config lacks this key.
                        config["start_padding_enabled"] = True

                    config = spec.model._ray_config_to_optuna(config)

                params["config"] = config

                mixture_config = self.neural.mixture
                quantile_config = self.neural.quantile
                params["gpus"] = min(self.neural.gpus, torch.cuda.device_count())
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

    def set_columns(self, columns_config: ForecastColumnConfig):
        """
        Set the input columns for the forecast configuration based on the dataset.
        """
        self.columns_config = columns_config

    def set_freq(self, freq: FrequencyType):
        """
        Set the frequency for the forecast configuration.
        """
        self.freq = freq

    def set_horizon(self, horizon: int):
        """
        Set the forecast horizon for the forecast configuration.
        """
        self.horizon = horizon
