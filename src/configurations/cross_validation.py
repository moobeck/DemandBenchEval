from dataclasses import dataclass
from typing import Dict, TypedDict
from .enums import DatasetName

class CrossValWindowConfig(TypedDict):
    n_windows: int
    step_size: int
    refit: int

class CrossValDatasetConfig(TypedDict):
    test: CrossValWindowConfig
    val: CrossValWindowConfig

@dataclass(frozen=True)
class CrossValidationConfig:
    data: Dict[DatasetName, CrossValDatasetConfig]
    dataset_config: CrossValDatasetConfig = None

    def set_dataset_config(self, dataset_name: DatasetName) -> None:
        """
        Sets the dataset configuration for cross-validation.
        """
        self.dataset_config = self.data.get(dataset_name)


