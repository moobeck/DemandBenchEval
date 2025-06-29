from dataclasses import dataclass, field
from typing import List  #
from demandbench.datasets import Dataset


@dataclass
class ForecastColumnConfig:
    """
    A dataclass to store the names of the columns used in the output DataFrame.
    """

    date: str = "date"
    sku_index: str = "skuID"
    target: str = "demand"
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

    def set_exogenous(self, dataset: Dataset):
        """
        Sets the exogenous features for dataset.
        """
        self.exogenous = self.base_exogenous + [
            col for col in dataset.features.columns if col.startswith("feature_")
        ]

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