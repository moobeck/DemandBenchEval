from demandbench.datasets import (
    Dataset,
    load_m5,
    load_favorita,
    load_rohlik,
    load_rossmann,
    load_bakery,
    load_yaz,
    load_pharmacy,
    load_freshretail50k,
    load_hoteldemand,
    load_onlineretail,
    load_onlineretail2,
    load_hierarchicalsales,
    load_australianretail,
    load_carparts,
    load_kaggledemand,
    load_productdemand,
    load_pharmacy2,
    load_vn1,
    load_kagglewalmart,
    load_fossil,
)

from src.configurations.utils.enums import DatasetName


import logging


class DatasetLoader:
    """
    Factory class to load dataset instances based on dataset names.
    """

    @staticmethod
    def load(dataset_name: DatasetName) -> Dataset:
        """
        Load and return the dataset instance corresponding to the given dataset name.
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
        elif dataset_name == DatasetName.FRESHRETAIL50K:
            return load_freshretail50k()
        elif dataset_name == DatasetName.HOTEL:
            return load_hoteldemand()
        elif dataset_name == DatasetName.ONLINERETAIL:
            return load_onlineretail()
        elif dataset_name == DatasetName.ONLINERETAIL2:
            return load_onlineretail2()
        elif dataset_name == DatasetName.HIERARCHICALSALES:
            return load_hierarchicalsales()
        elif dataset_name == DatasetName.AUSTRALIANRETAIL:
            return load_australianretail()
        elif dataset_name == DatasetName.CARPARTS:
            return load_carparts()
        elif dataset_name == DatasetName.KAGGLEDEMAND:
            return load_kaggledemand()
        elif dataset_name == DatasetName.PRODUCTDEMAND:
            return load_productdemand()
        elif dataset_name == DatasetName.PHARMACY2:
            return load_pharmacy2()
        elif dataset_name == DatasetName.VN1:
            return load_vn1()
        elif dataset_name == DatasetName.KAGGLEWALMART:
            return load_kagglewalmart()
        elif dataset_name == DatasetName.FOSSIL:
            return load_fossil()
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
