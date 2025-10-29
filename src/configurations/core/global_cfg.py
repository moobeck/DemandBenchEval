from dataclasses import dataclass

import yaml
import logging
from typing import Any

from src.configurations.core.system import SystemConfig
from src.configurations.core.file_path import FilePathConfig
from src.configurations.data.datasets import DatasetConfig
from src.configurations.data.input_column import InputColumnConfig
from src.configurations.data.preprocessing import PreprocessingConfig
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.evaluation.cross_validation import (
    CrossValidationConfig,
    CrossValWindowConfig,
    CrossValDatasetConfig,
)
from src.configurations.forecasting.forecasting import ForecastConfig
from src.configurations.evaluation.metrics import MetricConfig
from src.configurations.forecasting.quantile import QuantileConfig
from src.configurations.utils.wandb import WandbConfig
from src.configurations.utils.enums import (
    DatasetName,
    FileFormat,
    ModelName,
    Framework,
    MetricName,
    TargetScalerType,
)
from demandbench.datasets import Dataset


@dataclass(frozen=True)
class GlobalConfig:
    system: SystemConfig
    filepaths: FilePathConfig
    datasets: DatasetConfig
    input_columns: InputColumnConfig
    preprocessing: PreprocessingConfig
    forecast_columns: ForecastColumnConfig
    cross_validation: CrossValidationConfig
    forecast: ForecastConfig
    metrics: MetricConfig
    wandb: WandbConfig

    def set_dataset(self, dataset_name: DatasetName, dataset: Dataset):
        """
        Sets the dataset for the configuration.
        """
        self.filepaths.set_file_paths(dataset_name)
        self.forecast_columns.set_columns(dataset)
        self.forecast.set_freq(dataset, self.input_columns)
        self.forecast.set_horizon()
        self.forecast.set_lags()
        self.forecast.set_columns(self.forecast_columns)

        if not self.metrics.seasonality_provided:
            self.metrics.set_seasonality(freq=self.forecast.freq)
        self.metrics.set_metrics()

        self.cross_validation.set_dataset_config(dataset_name)

    @classmethod
    def build(cls, public_config: dict, private_config: dict) -> "GlobalConfig":

        system = public_config.get("system", {})
        if not system:
            logging.warning("No system settings provided in the public config.")

        filepaths = public_config.get("filepaths", {})
        if not filepaths:
            logging.warning("No file paths provided in the public config.")
        dataset_names = public_config.get("datasets", [])
        input_colums = public_config.get("input_columns", {})
        if not input_colums:
            logging.warning("No input columns provided in the public config.")
        preprocessing = public_config.get("preprocessing", {})
        if not preprocessing:
            logging.warning("No preprocessing settings provided in the public config.")
        forecast_columns = public_config.get("forecast_columns", {})
        if not forecast_columns:
            logging.warning("No forecast columns provided in the public config.")
        cross_validation_data = public_config.get("cross_validation", {})
        if not cross_validation_data:
            logging.warning(
                "No cross-validation settings provided in the public config."
            )
        forecast = public_config.get("forecast", {})
        if not forecast:
            logging.warning("No forecast settings provided in the public config.")
        metrics = public_config.get("metrics", {})
        if not metrics:
            logging.warning("No metrics settings provided in the public config.")
        lags = public_config.get("lags", [])
        if not lags:
            logging.warning("No lags provided in the public config.")

        log_wandb = public_config.get("log_wandb", False)
        wandb: dict = private_config.get("wandb", {})  #
        if not wandb and log_wandb:
            logging.warning("No W&B settings provided in the private config.")
        wandb.update({"log_wandb": log_wandb})

        return GlobalConfig(
            system=SystemConfig(
                GPU=system.get("GPU", 0), RANDOM_SEED=system.get("RANDOM_SEED", 42)
            ),
            filepaths=FilePathConfig(
                processed_data_dir=filepaths.get(
                    "processed_data_dir", "data/processed"
                ),
                sku_stats_dir=filepaths.get("sku_stats_dir", "data/sku_stats"),
                cv_results_dir=filepaths.get("cv_results_dir", "data/cv_results"),
                eval_results_dir=filepaths.get("eval_results_dir", "data/eval_results"),
                eval_plots_dir=filepaths.get("eval_plots_dir", "data/eval_plots"),
                file_format=FileFormat[filepaths.get("file_format", "FEATHER")],
            ),
            datasets=DatasetConfig(names=[DatasetName[name] for name in dataset_names]),
            input_columns=InputColumnConfig(
                time_series_index=input_colums["time_series_index"],
                date=input_colums["date"],
                target=input_colums["target"],
                frequency=input_colums["frequency"],
            ),
            preprocessing=PreprocessingConfig(
                target_transform=TargetScalerType[preprocessing["target_transform"]],
            ),
            forecast_columns=ForecastColumnConfig(
                time_series_index=forecast_columns["time_series_index"],
                date=forecast_columns["date"],
                product_index=forecast_columns.get("product_index", "productID"),
                store_index=forecast_columns.get("store_index", "storeID"),
                target=forecast_columns["target"],
                cutoff=forecast_columns["cutoff"],
            ),
            cross_validation=CrossValidationConfig(
                data={
                    DatasetName[name]: CrossValDatasetConfig(
                        test=CrossValWindowConfig(
                            n_windows=cv["test"]["n_windows"],
                            step_size=cv["test"]["step_size"],
                            refit=cv["test"]["refit"],
                        ),
                        val=CrossValWindowConfig(
                            n_windows=cv["val"]["n_windows"],
                            step_size=cv["val"]["step_size"],
                            refit=cv["val"]["refit"],
                        ),
                    )
                    for name, cv in cross_validation_data.items()
                }
            ),
            forecast=ForecastConfig(
                names=[ModelName[name] for name in forecast["models"]],
                model_config={
                    Framework[fw]: forecast["model_config"][fw]
                    for fw in forecast["model_config"]
                },
                lags_config=lags,
            ),
            metrics=MetricConfig(
                names=[
                    MetricName[name] for name in public_config["metrics"]["metrics"]
                ],
                seasonality=public_config["metrics"].get("seasonality", None),
                quantiles=QuantileConfig(
                    **public_config["metrics"].get("quantiles", None)
                ),
            ),
            wandb=WandbConfig(
                api_key=(wandb.get("api_key") if wandb else None),
                entity=(wandb.get("entity") if wandb else None),
                project=(wandb.get("project", "bench-forecast") if wandb else None),
                log_wandb=wandb.get("log_wandb", False) if wandb else False,
            ),
        )

    @staticmethod
    def load_dict(path) -> dict[str, Any]:
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.warning(f"Config file not found: {path}. Returning empty config.")
            return {}
