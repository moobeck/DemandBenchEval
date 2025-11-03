from typing import List
from demandbench.datasets import Dataset
from typing import Literal


from src.constants.tasks import Task, HierarchyType

ID_COLUMN_MAPPING = {
    HierarchyType.PRODUCT_STORE: "timeSeriesID",
    HierarchyType.PRODUCT: "productID",
    HierarchyType.STORE: "storeID",
}


class ForecastColumnConfig:
    """Mutable feature registry built on top of immutable column names."""

    def __init__(self) -> None:

        self.date: str = "date"
        self.time_series_index: str = "timeSeriesID"
        self.target: str = "target"
        self.store_index: str = "storeID"
        self.product_index: str = "productID"
        self.cutoff: str = "cutoff"

        self.categorical: List[str] = []
        self.past_exogenous: List[str] = []
        self.future_exogenous: List[str] = []
        self.exogenous: List[str] = []
        self.static: List[str] = []

    @property
    def ts_base_cols(self) -> List[str]:
        """
        Returns the three core columns for time-series: SKU, date, and target.
        """
        return [self.time_series_index, self.date, self.target]

    def set_past_exogenous(self, dataset: Dataset):
        """
        Sets the past exogenous features for dataset.
        """

        self.past_exogenous = list(dataset.metadata.past_exo_features or [])

    def set_future_exogenous(self, dataset: Dataset):
        """
        Sets the future exogenous features for dataset.
        """
        self.future_exogenous = list(dataset.metadata.future_exo_features or [])

    def set_static(self, dataset: Dataset):
        """
        Sets the static features for dataset.
        """
        self.static = list(dataset.metadata.static_features or [])

    def set_exogenous(self):
        """
        Sets the exogenous features for dataset.
        """
        self.exogenous = self.past_exogenous + self.future_exogenous + self.static

    def set_categorical(self, dataset: Dataset):
        """
        Sets the categorical features for dataset.
        """
        self.categorical = [
            feature
            for feature in dataset.metadata.categorical_features
            if feature in self.exogenous
        ]

    def set_columns(self, task: Task):
        """
        Sets the forecast columns based on the dataset metadata.
        This includes setting exogenous, static, and categorical features.
        """

        self.set_id_column(task)
        self.set_past_exogenous(task.dataset)
        self.set_future_exogenous(task.dataset)
        self.set_static(task.dataset)
        self.set_exogenous()
        self.set_categorical(task.dataset)

    def set_id_column(self, task: Task):
        """
        Sets the time series index column based on the task's hierarchy.
        """
        self.time_series_index = ID_COLUMN_MAPPING[task.hierarchy]

    @staticmethod
    def _add_columns(columns: List[str], target_list: List[str]):
        """
        Helper method to add columns to a target list if they are not already present.
        """
        for col in columns:
            if col not in target_list:
                target_list.append(col)

    def add_features(
        self,
        new_columns: List[str],
        feature_type: Literal["past_exogenous", "future_exogenous", "static"],
    ):
        """
        Adds new features to the specified feature type list.

        Parameters:
        -----------
        new_columns : List[str]
            List of new column names to add.
        feature_type : Literal["past_exogenous", "future_exogenous", "static"]
            The type of feature list to which the new columns should be added.
        """
        if feature_type == "past_exogenous":
            self._add_columns(new_columns, self.past_exogenous)
        elif feature_type == "future_exogenous":
            self._add_columns(new_columns, self.future_exogenous)
        elif feature_type == "static":
            self._add_columns(new_columns, self.static)
        else:
            raise ValueError(
                f"Invalid feature_type: {feature_type}. Must be one of 'past_exogenous', 'future_exogenous', or 'static'."
            )

        self.set_exogenous()

    def remove_features(
        self,
        remove_columns: List[str],
        feature_type: Literal[
            "past_exogenous", "future_exogenous", "static", None
        ] = None,
    ):
        """
        Removes features from the specified feature type list or from all exogenous features if feature_type is None.

        Parameters:
        -----------
        remove_columns : List[str]
            List of column names to remove.
        feature_type : Literal["past_exogenous", "future_exogenous", "static", None]
            The type of feature list from which the columns should be removed. If None, removes from all exogenous features.
        """
        if feature_type == "past_exogenous":
            self.past_exogenous = [
                col for col in self.past_exogenous if col not in remove_columns
            ]
        elif feature_type == "future_exogenous":
            self.future_exogenous = [
                col for col in self.future_exogenous if col not in remove_columns
            ]
        elif feature_type == "static":
            self.static = [col for col in self.static if col not in remove_columns]
        elif feature_type is None:
            self.past_exogenous = [
                col for col in self.past_exogenous if col not in remove_columns
            ]
            self.future_exogenous = [
                col for col in self.future_exogenous if col not in remove_columns
            ]
            self.static = [col for col in self.static if col not in remove_columns]
        else:
            raise ValueError(
                f"Invalid feature_type: {feature_type}. Must be one of 'past_exogenous', 'future_exogenous', 'static', or None."
            )

        self.set_exogenous()

    def replace_features(self, mapping: dict):
        """
        Replaces features based on a provided mapping dictionary.
        Keys in the mapping represent existing feature names to be replaced,
        and values represent the new feature names.
        The feature type is inferred based on where the key exists.

        Parameters:
        -----------
        mapping : dict
            A dictionary mapping existing feature names to new feature names.
        """
        for old_feature, new_feature in mapping.items():
            if old_feature in self.past_exogenous:
                self.past_exogenous = [
                    new_feature if col == old_feature else col
                    for col in self.past_exogenous
                ]
            if old_feature in self.future_exogenous:
                self.future_exogenous = [
                    new_feature if col == old_feature else col
                    for col in self.future_exogenous
                ]
            if old_feature in self.static:
                self.static = [
                    new_feature if col == old_feature else col for col in self.static
                ]

        self.set_exogenous()


DEFAULT_FORECASTING_COLUMNS = ForecastColumnConfig()
