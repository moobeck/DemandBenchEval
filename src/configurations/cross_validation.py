from dataclasses import dataclass
from typing import Dict, TypedDict, List, Literal
from .enums import DatasetName, Frequency
import pandas as pd


@dataclass(frozen=True)
class CrossValWindowConfig:
    n_windows: int
    step_size: int
    refit: bool


class CrossValDatasetConfig(TypedDict):
    test: CrossValWindowConfig
    val: CrossValWindowConfig


@dataclass()
class CrossValidationConfig:
    data: Dict[DatasetName, CrossValDatasetConfig]
    test: CrossValWindowConfig = None
    val: CrossValWindowConfig = None

    def set_dataset_config(self, dataset_name: DatasetName) -> None:
        """
        Sets the dataset configuration for cross-validation.
        """

        dataset_config = self.data.get(dataset_name)
        if dataset_config:
            self.test = dataset_config.get("test")
            self.val = dataset_config.get("val")

        else:
            raise ValueError(
                f"No cross-validation config found for dataset: {dataset_name}"
            )

    def get_cutoff_date(
        self, max_date: pd.Timestamp, freq: Frequency, split: Literal["test", "val"]
    ) -> pd.Timestamp:
        """
        Calculate the cutoff date for training data based on frequency and split type.
        """
        if split == "test":
            config = self.test
        elif split == "val":
            config = self.val
        else:
            raise ValueError(f"Unsupported split: {split}")

        n_windows = config.n_windows
        step_size = config.step_size

        if freq == Frequency.DAILY:
            offset = pd.Timedelta(days=n_windows * step_size)
        elif freq == Frequency.WEEKLY:
            offset = pd.Timedelta(weeks=n_windows * step_size)
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
        return max_date - offset
