from dataclasses import dataclass

from src.configurations.file_path import FilePathConfig
from src.configurations.input_column import InputColumnConfig
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.cross_validation import CrossValidationConfig
from src.configurations.forecasting import ForecastConfig
from src.configurations.metrics import MetricConfig
from src.configurations.wandb import WandbConfig


@dataclass(frozen=True)
class GlobalConfig:
    filepaths: FilePathConfig
    input_columns: InputColumnConfig
    forecast_columns: ForecastColumnConfig
    cross_validation: CrossValidationConfig
    forecast: ForecastConfig
    metrics: MetricConfig
    wandb: WandbConfig
    seed: int
