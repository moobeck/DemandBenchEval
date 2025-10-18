from dataclasses import dataclass, field
from typing import List
from demandbench.datasets import Dataset


@dataclass
class ForecastColumnConfig:
    """
    A dataclass to store the names of the columns used in the output DataFrame.
    """

    date: str = "date"
    time_series_index: str = "skuID"
    target: str = "demand"
    store_index: str = "storeID"
    product_index: str = "productID"
    base_exogenous: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    past_exogenous: List[str] = field(default_factory=list)
    future_exogenous: List[str] = field(default_factory=list)
    exogenous: List[str] = field(default_factory=list)
    static: List[str] = field(default_factory=list)
    cutoff: str = "cutoff"

    @property
    def ts_base_cols(self) -> List[str]:
        """
        Returns the three core columns for time-series: SKU, date, and target.
        """
        return [self.time_series_index, self.date, self.target]

    def set_base_exogenous(self):
        """
        Sets the base exogenous features for dataset.
        """
        self.base_exogenous = [self.store_index, self.product_index]

    def set_past_exogenous(self, dataset: Dataset):
        """
        Sets the past exogenous features for dataset.
        """
        self.past_exogenous = [
            feature
            for feature in dataset.metadata.past_exo_features
            if feature not in self.base_exogenous
        ]

    def set_future_exogenous(self, dataset: Dataset):
        """
        Sets the future exogenous features for dataset.
        """
        self.future_exogenous = [
            feature
            for feature in dataset.metadata.future_exo_features
            if feature not in self.base_exogenous
        ]

    def set_exogenous(self, dataset: Dataset):
        """
        Sets the exogenous features for dataset.
        """
        self.exogenous = self.base_exogenous + [
            feature
            for feature in dataset.metadata.exo_features
            if feature not in self.base_exogenous
        ]

    def set_static(self, dataset: Dataset):
        """
        Sets the static features for dataset.
        """
        self.static = dataset.metadata.static_features

    def set_categorical(self, dataset: Dataset):
        """
        Sets the categorical features for dataset.
        """
        self.categorical = [
            feature
            for feature in dataset.metadata.categorical_features
            if feature in self.exogenous
        ]

    def set_columns(self, dataset: Dataset):
        """
        Sets the forecast columns based on the dataset metadata.
        This includes setting exogenous, static, and categorical features.
        """
        self.set_base_exogenous()
        self.set_past_exogenous(dataset)
        self.set_future_exogenous(dataset)
        self.set_exogenous(dataset)
        self.set_static(dataset)
        self.set_categorical(dataset)

    def add_exogenous(self, columns: List[str], future: bool):
        """
        Adds additional exogenous features to the existing list.
        """
        if future:
            self.future_exogenous += columns
        else:
            self.past_exogenous += columns

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
