from dataclasses import dataclass

import yaml
import logging
from typing import Any, List

from src.configurations.core.system import SystemConfig
from src.configurations.core.file_path import FilePathConfig
from src.configurations.data.forecast_column import (
    ForecastColumnConfig,
    DEFAULT_FORECASTING_COLUMNS,
)
from src.configurations.evaluation.cross_validation import (
    CrossValidationConfig,
    DEFAULT_CROSS_VALIDATION_CONFIG,
)
from src.configurations.forecasting.forecasting import ForecastConfig
from src.configurations.evaluation.metrics import MetricConfig
from src.configurations.forecasting.quantile import DEFAULT_QUANTILE_CONFIG
from src.configurations.utils.wandb import WandbConfig
from src.configurations.utils.enums import (
    FileFormat,
    ModelName,
    MetricName,
)
from src.constants import DEFAULT_MODEL_FRAMEWORK_CONFIG
from src.constants.tasks import TASKS, Task
from src.configurations.data.preprocessing import (
    PreprocessingConfig,
    DEFAULT_PREPROCESSING_CONFIG,
)


@dataclass(frozen=True)
class GlobalConfig:
    system: SystemConfig
    filepaths: FilePathConfig
    preprocessing: PreprocessingConfig
    forecast_columns: ForecastColumnConfig
    cross_validation: CrossValidationConfig
    forecast: ForecastConfig
    metrics: MetricConfig
    wandb: WandbConfig
    tasks: List[Task]

    def set_task(self, task: Task):
        """
        Sets the configuration for a given task.
        """
        self.filepaths.set_file_paths(task.dataset_name)
        self.forecast_columns.set_columns(task)
        self.forecast.set_columns(self.forecast_columns)
        self.forecast.set_freq(task.frequency)
        self.forecast.set_horizon(task.horizon)
        self.metrics.set_seasonality(freq=task.frequency)
        self.metrics.set_metrics()

        self.cross_validation.set_task(task)

    @classmethod
    def build(cls, public_config: dict, private_config: dict) -> "GlobalConfig":
        """
        Builds a GlobalConfig instance from public and private configuration dictionaries.
        """
        system_config = cls._build_system_config(public_config.get("system", {}))
        filepaths_config = cls._build_filepaths_config(
            public_config.get("filepaths", {})
        )
        preprocessing_config = DEFAULT_PREPROCESSING_CONFIG
        forecast_columns_config = DEFAULT_FORECASTING_COLUMNS
        cross_validation_config = DEFAULT_CROSS_VALIDATION_CONFIG
        forecast_config = cls._build_forecast_config(public_config.get("forecast", {}))
        metrics_config = cls._build_metrics_config(public_config.get("metrics", {}))
        wandb_config = cls._build_wandb_config(
            public_config,
            private_config.get("wandb", {}),
        )
        tasks = cls._build_tasks(public_config.get("tasks", []))

        logging.info(
            f"Building GlobalConfig with tasks: {[task.name for task in tasks]}"
        )

        return GlobalConfig(
            system=system_config,
            filepaths=filepaths_config,
            preprocessing=preprocessing_config,
            forecast_columns=forecast_columns_config,
            cross_validation=cross_validation_config,
            forecast=forecast_config,
            metrics=metrics_config,
            wandb=wandb_config,
            tasks=tasks,
        )

    @classmethod
    def _build_system_config(cls, system_dict: dict) -> SystemConfig:
        """Builds the SystemConfig from the provided dictionary."""
        if not system_dict:
            logging.warning("No system settings provided in the public config.")
        return SystemConfig(
            GPU=system_dict.get("GPU", 0),
            RANDOM_SEED=system_dict.get("RANDOM_SEED", 42),
        )

    @classmethod
    def _build_filepaths_config(cls, filepaths_dict: dict) -> FilePathConfig:
        """Builds the FilePathConfig from the provided dictionary."""
        if not filepaths_dict:
            logging.warning("No file paths provided in the public config.")
        return FilePathConfig(
            processed_data_dir=filepaths_dict.get(
                "processed_data_dir", "data/processed"
            ),
            sku_stats_dir=filepaths_dict.get("sku_stats_dir", "data/sku_stats"),
            cv_results_dir=filepaths_dict.get("cv_results_dir", "data/cv_results"),
            stats_insample_cv_results_dir=filepaths_dict.get(
                "stats_insample_cv_results_dir", "data/stats_insample_cv_results"
            ),
            stats_insample_results_dir=filepaths_dict.get(
                "stats_insample_results_dir", "data/stats_insample_results"
            ),
            eval_results_dir=filepaths_dict.get(
                "eval_results_dir", "data/eval_results"
            ),
            eval_plots_dir=filepaths_dict.get("eval_plots_dir", "data/eval_plots"),
            file_format=FileFormat[filepaths_dict.get("file_format", "FEATHER")],
        )

    @classmethod
    def _build_forecast_config(cls, forecast_dict: dict) -> ForecastConfig:
        """Builds the ForecastConfig from the provided dictionary."""
        if not forecast_dict:
            logging.warning("No forecast settings provided in the public config.")
        return ForecastConfig(
            names=[ModelName[name] for name in forecast_dict.get("models", [])],
            model_config=DEFAULT_MODEL_FRAMEWORK_CONFIG,
        )

    @classmethod
    def _build_metrics_config(cls, metrics_dict: dict) -> MetricConfig:
        """Builds the MetricConfig from the provided dictionary."""
        if not metrics_dict:
            logging.warning("No metrics settings provided in the public config.")
        return MetricConfig(
            names=[MetricName[metric] for metric in metrics_dict],
            quantiles=DEFAULT_QUANTILE_CONFIG,
        )

    @classmethod
    def _build_wandb_config(cls, public_config: dict, wandb_dict: dict) -> WandbConfig:
        """Builds the WandbConfig from the provided settings."""

        log_wandb = public_config.get("log_wandb")
        if log_wandb is None:
            log_wandb = public_config.get("system", {}).get("log_wandb", False)

        if not wandb_dict and log_wandb:
            logging.warning("No W&B settings provided in the private config.")
        wandb_dict = dict(wandb_dict)  # Copy to avoid mutating original
        wandb_dict["log_wandb"] = log_wandb
        return WandbConfig(
            api_key=wandb_dict.get("api_key"),
            entity=wandb_dict.get("entity"),
            project=wandb_dict.get("project", "bench-forecast"),
            log_wandb=wandb_dict.get("log_wandb", False),
        )

    @classmethod
    def _build_tasks(cls, task_names: list) -> List[Task]:
        """Builds the list of Task objects from task names."""
        return [TASKS[name] for name in task_names]

    @staticmethod
    def load_dict(path) -> dict[str, Any]:
        """
        Loads a YAML configuration file and returns it as a dictionary.
        If the file is not found, logs a warning and returns an empty dict.
        """
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.warning(f"Config file not found: {path}. Returning empty config.")
            return {}
