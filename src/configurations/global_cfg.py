from dataclasses import dataclass

from src.configurations.system import SystemConfig
from src.configurations.file_path import FilePathConfig
from src.configurations.datasets import DatasetConfig
from src.configurations.input_column import InputColumnConfig
from src.configurations.preprocessing import PreprocessingConfig
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.cross_validation import CrossValidationConfig
from src.configurations.forecasting import ForecastConfig
from src.configurations.metrics import MetricConfig
from src.configurations.wandb import WandbConfig
from src.configurations.enums import DatasetName
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
