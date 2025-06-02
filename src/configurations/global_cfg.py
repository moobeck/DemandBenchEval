from dataclasses import dataclass

from src.configurations.file_path import FilePathConfig
from src.configurations.datasets import DatasetConfig
from src.configurations.input_column import InputColumnConfig
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.cross_validation import CrossValidationConfig
from src.configurations.forecasting import ForecastConfig
from src.configurations.metrics import MetricConfig
from src.configurations.wandb import WandbConfig
from src.configurations.enums import DatasetName
from demandbench.datasets import Dataset


@dataclass(frozen=True)
class GlobalConfig:
    filepaths: FilePathConfig
    datasets: DatasetConfig
    input_columns: InputColumnConfig
    forecast_columns: ForecastColumnConfig
    cross_validation: CrossValidationConfig
    forecast: ForecastConfig
    metrics: MetricConfig
    wandb: WandbConfig
    seed: int

    def set_dataset(self, dataset_name: DatasetName, dataset: Dataset):
        """
        Sets the dataset for the configuration.
        """
        self.filepaths.set_file_paths(dataset_name)
        self.forecast_columns.set_exogenous(dataset)
