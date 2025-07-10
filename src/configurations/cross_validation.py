from dataclasses import dataclass
import logging
from typing import Dict, TypedDict
from .enums import DatasetName


@dataclass(frozen=True)
class CrossValWindowConfig:
    n_windows: int
    step_size: int
    refit: int

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
            self.test = dataset_config.get('test')
            self.val = dataset_config.get('val')

        else: 
            logging.warning(f"No cross-validation config found for dataset: {dataset_name}")