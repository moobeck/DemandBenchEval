from demandbench.datasets import Dataset, load_m5, load_favorita, load_bakery
from src.configurations.enums import DatasetName

import logging

class DatasetFactory:
    """
    Factory class to create datasets based on the DatasetName enum.
    """

    @staticmethod
    def create_dataset(dataset_name: DatasetName) -> Dataset:
        """
        Create a dataset instance based on the provided dataset name.
        """
        logging.info(f"Creating dataset for: {dataset_name}")

        if dataset_name == DatasetName.M5:
            return load_m5()
        elif dataset_name == DatasetName.FAVORITA:
            return load_favorita()
        elif dataset_name == DatasetName.BAKERY:
            return load_bakery()
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
