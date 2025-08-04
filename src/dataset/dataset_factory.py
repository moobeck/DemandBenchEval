from demandbench.datasets import (
    Dataset,
    load_m5,
    load_favorita,
    load_rohlik,
    load_rossmann,
    load_bakery,
    load_yaz,
    load_pharmacy,
    load_hoteldemand,
    load_onlineretail,
    
)
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
        elif dataset_name == DatasetName.ROHLIK:
            return load_rohlik()
        elif dataset_name == DatasetName.ROSSMANN:
            return load_rossmann()
        elif dataset_name == DatasetName.BAKERY:
            return load_bakery()
        elif dataset_name == DatasetName.YAZ:
            return load_yaz()
        elif dataset_name == DatasetName.PHARMACY:
            return load_pharmacy()
        elif dataset_name == DatasetName.HOTEL:
            return load_hoteldemand()
        elif dataset_name == DatasetName.ONLINERETAIL:
            return load_onlineretail()
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
