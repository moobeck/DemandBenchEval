import polars as pl
import pandas as pd
from typing import List
from datetime import datetime
from utilsforecast.preprocessing import fill_gaps


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
        file_paths: FilePathConfig,
        input_columns: InputColumnConfig,
        forecast_columns: ForecastColumnConfig,
    ):

        self._input_columns = input_columns
        self._file_paths = file_paths
        self._forecast_columns = forecast_columns

        self.df_features = None
        self.df_target = None
        self.df_merged = None

    def _load_feather(self, paths: List[str]) -> pl.DataFrame:
        """Load multiple feather files into a single Polars DataFrame."""
        frames = [pl.read_ipc(path) for path in paths]
        return pl.concat(frames, how="vertical")

    def load_data(self):
        """Loads features and targets from provided feather file paths."""

        logging.info(
            f"Loading features from {self._file_paths.train_data_features} and {self._file_paths.val_data_features}"
        )
        logging.info(
            f"Loading targets from {self._file_paths.train_data_target} and {self._file_paths.val_data_target}"
        )

        self.df_features = self._load_feather(
            [self._file_paths.train_data_features, self._file_paths.val_data_features]
        )
        self.df_target = self._load_feather(
            [self._file_paths.train_data_target, self._file_paths.val_data_target]
        )

    def merge(self):

        logging.info(
            f"Merging features and targets on {self._input_columns.dp_index} column"
        )

        """Merge features and target on index column."""
        if self.df_features is None or self.df_target is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # ensure unique index for join
        self.df_merged = self.df_features.join(
            self.df_target.select([self._input_columns.dp_index, "target"]),
            on=self._input_columns.dp_index,
            how="left",
        )

    def _compute_dates(self):
        # 1. Define the “origin” so that dateID = 1 → 2000-01-01
        origin = datetime(2000, 1, 1)

        # 2. Build `ds` by adding (dateID – 1) days to that origin
        self.df_merged = self.df_merged.with_columns(
            [
                (
                    # cast Python datetime → Polars Date
                    pl.lit(origin).cast(pl.Date)
                    # add one duration per row equal to dateID−1
                    + pl.duration(
                        days=(pl.col(self._input_columns.date).cast(pl.Int64) - 1)
                    )
                ).alias(self._forecast_columns.date),
            ]
        )

    def remove_skus(self, skus: List[str]):
        """Remove specific SKUs from the merged DataFrame."""

        logging.info(f"Removing SKUs: {skus}")

        if self.df_merged is None:
            raise ValueError("Data not merged. Call merge() first.")
        self.df_merged = self.df_merged.filter(
            ~pl.col(self._input_columns.sku_index).is_in(skus)
        )
        return self.df_merged

    def prepare_nixtla(self) -> pd.DataFrame:
        """Prepare a pandas DataFrame for Nixtla with required columns."""

        logging.info("Preparing data for Nixtla")

        if self.df_merged is None:
            raise ValueError("Data not merged. Call merge() first.")

        # compute ds
        self._compute_dates()

        df = self.df_merged.select(
            [
                self._input_columns.sku_index,
                self._forecast_columns.date,
                self._input_columns.target,
                *(self._forecast_columns.exogenous),
            ]
        ).to_pandas()

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
