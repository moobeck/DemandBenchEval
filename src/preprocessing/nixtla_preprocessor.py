import pandas as pd
from typing import List, Literal, Union
from datetime import datetime
from utilsforecast.preprocessing import fill_gaps
from demandbench.datasets import Dataset

from src.configurations.input_column import InputColumnConfig
from src.configurations.file_path import FilePathConfig
from src.configurations.forecast_column import ForecastColumnConfig
import logging


class NixtlaPreprocessor:
    """
    A class to load, merge, and convert feather datasets into the Nixtla NeuralForecast-ready format.
    """

    def __init__(
        self,
        dataset: Dataset,
        input_columns: InputColumnConfig,
        forecast_columns: ForecastColumnConfig,
    ):

        self._input_columns = input_columns
        self._dataset = dataset
        self._forecast_columns = forecast_columns

        self.df_merged = None

    def merge(self):

        self.df_merged = self._dataset.get_merged_data().to_pandas()

    def remove_skus(self, skus: Union[List[str], Literal["not_at_min_date"]]):
        """
        Remove specific SKUs from the merged DataFrame.

        Parameters:
        - skus (List[str] | "not_at_min_date"):
            - If a list of SKUs is provided, those SKUs will be removed.
            - If "not_at_min_date" is passed, all SKUs that do NOT have data starting
            at the minimum date in the DataFrame will be removed.
        """

        logging.info(f"Removing SKUs: {skus}")

        if self.df_merged is None:
            raise ValueError("Data not merged. Call merge() first.")

        sku_col = self._input_columns.sku_index
        date_col = "date"  # Adjust if your actual date column has a different name

        if skus == "not_at_min_date":
            min_date = self.df_merged[date_col].min()

            # Find SKUs that have entries on the minimum date
            skus_to_keep = self.df_merged[self.df_merged[date_col] == min_date][
                sku_col
            ].unique()

            # Keep only rows with those SKUs
            self.df_merged = self.df_merged[self.df_merged[sku_col].isin(skus_to_keep)]
        else:
            self.df_merged = self.df_merged[~self.df_merged[sku_col].isin(skus)]

        return self.df_merged

    def prepare_nixtla(self) -> pd.DataFrame:
        """Prepare a pandas DataFrame for Nixtla with required columns."""

        logging.info("Preparing data for Nixtla")

        if self.df_merged is None:
            raise ValueError("Data not merged. Call merge() first.")

        df = self.df_merged[
            [
                self._input_columns.sku_index,
                self._forecast_columns.date,
                self._input_columns.target,
                *(self._forecast_columns.exogenous),
            ]
        ]

        # Fill gaps in the time series
        df = fill_gaps(
            df,
            freq="D",
            id_col=self._input_columns.sku_index,
            time_col=self._forecast_columns.date,
        )

        # Fill missing values
        df = df.ffill()

        # rename columns to Nixtla standard
        df = df.rename(
            columns={
                self._input_columns.sku_index: self._forecast_columns.sku_index,
                self._input_columns.target: self._forecast_columns.target,
            }
        )
        return df
