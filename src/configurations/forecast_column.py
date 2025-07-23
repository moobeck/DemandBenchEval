from dataclasses import dataclass, field
from typing import List
from demandbench.datasets import Dataset
from demandbench.datasets.metadata import VariableType


@dataclass
class ForecastColumnConfig:
    """
    A dataclass to store the names of the columns used in the output DataFrame.
    """

    date: str = "date"
    sku_index: str = "skuID"
    target: str = "demand"
    store_index: str = "storeID"
    product_index: str = "productID"
    base_exogenous: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    exogenous: List[str] = field(default_factory=list)
    static: List[str] = field(default_factory=list)
    cutoff: str = "cutoff"

    @property
    def ts_base_cols(self) -> List[str]:
        """
        Returns the three core columns for time-series: SKU, date, and target.
        """
        return [self.sku_index, self.date, self.target]
    

    def set_base_exogenous(self, dataset: Dataset):
        """
        Sets the base exogenous features for dataset.
        """
        self.base_exogenous = []
        if len(dataset.metadata.stores) > 1:
            self.base_exogenous.append(self.store_index)
        if len(dataset.metadata.products) > 1:
            self.base_exogenous.append(self.product_index)


    def set_exogenous(self, dataset: Dataset):
        """
        Sets the exogenous features for dataset.
        """
        self.exogenous = self.base_exogenous + [
            feature.name for feature in dataset.metadata.exo_features if feature.name not in self.base_exogenous
        ]

    def set_static(self, dataset: Dataset):
        """
        Sets the static features for dataset.
        """
        self.static = [
            feature.name for feature in dataset.metadata.sku_features
            if feature.name in self.exogenous
        ]

    def set_categorical(self, dataset: Dataset):
        """
        Sets the categorical features for dataset.
        """
        self.categorical = [
            feature.name for feature in dataset.metadata.features
            if feature.var_type == VariableType.CATEGORICAL and feature.name in self.exogenous
        ]

    def set_columns(self, dataset: Dataset):
        """
        Sets the forecast columns based on the dataset metadata.
        This includes setting exogenous, static, and categorical features.
        """
        self.set_base_exogenous(dataset)
        self.set_exogenous(dataset)
        self.set_static(dataset)
        self.set_categorical(dataset)


    def add_exogenous(self, columns: List[str]):
        """
        Adds additional exogenous features to the existing list.
        """
        self.exogenous += columns

    def add_static(self, columns: List[str]):
        """
        Adds additional static features to the existing list.
        """
        self.static += columns

    def rename_static(self, mapping: dict):
        """
        Renames static columns based on the provided mapping.
        Mapping should be a dictionary where keys are old names and values are new names.
        """
        self.static = [mapping.get(col, col) for col in self.static]