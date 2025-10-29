import pandas as pd
from typing import List, Literal, Union
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from utilsforecast.preprocessing import fill_gaps
from demandbench.datasets import Dataset
from src.preprocessing.scaler.scaler import TargetScalerFactory
from src.preprocessing.encoder.date_encoder import DateEncoder
from src.preprocessing.encoder.category_encoder import CategoryEncoder
from src.preprocessing.encoder.statistical_encoder import StatisticalFeaturesEncoder
from src.configurations.utils.enums import Frequency
from src.configurations.data.input_column import InputColumnConfig
from src.configurations.data.preprocessing import PreprocessingConfig
from src.configurations.forecasting.forecasting import ForecastConfig
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.evaluation.cross_validation import CrossValidationConfig
import logging

from mlforecast import MLForecast


class Preprocessor:
    """
    A class to load, merge, and convert feather datasets into the format required
    """

    def __init__(
        self,
        dataset: Dataset,
        input_columns: InputColumnConfig,
        preprocessing: PreprocessingConfig,
        forecast_columns: ForecastColumnConfig,
        forecast: ForecastConfig,
        cross_validation: CrossValidationConfig,
    ):

        self._input_columns = input_columns
        self._preprocessing = preprocessing
        self._dataset = dataset
        self._forecast_columns = forecast_columns
        self._forecast = forecast
        self._cross_validation = cross_validation

        self.df_merged = None

    def merge(self):
        """
        Merge the dataset into a single DataFrame.
        """

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

        time_series_col = self._input_columns.time_series_index
        date_col = self._input_columns.date

        if skus == "not_at_min_date":
            min_date = self.df_merged[date_col].min()

            # Find SKUs that have entries on the minimum date
            skus_to_keep = self.df_merged[self.df_merged[date_col] == min_date][
                time_series_col
            ].unique()

            # Keep only rows with those SKUs
            self.df_merged = self.df_merged[
                self.df_merged[time_series_col].isin(skus_to_keep)
            ]
        else:
            self.df_merged = self.df_merged[~self.df_merged[time_series_col].isin(skus)]

        return self.df_merged

    def _filter_by_frequency(self, df: pd.DataFrame):
        """
        Filter the merged DataFrame by a specific frequency.
        """
        frequency_alias = Frequency.get_alias(self._forecast.freq, "demandbench")

        df = (
            df[df[self._input_columns.frequency] == frequency_alias]
            .copy()
            .reset_index(drop=True)
        )

        return df

    def prepare_forecasting_data(self) -> pd.DataFrame:
        """Prepare a pandas DataFrame for forecasting."""

        logging.info("Preparing data for forecasting")

        if self.df_merged is None:
            raise ValueError("Data not merged. Call merge() first.")

        df = self._filter_by_frequency(self.df_merged)

        # Rename columns to match names expected by the forecasting config
        df = df.rename(
            columns={
                self._input_columns.time_series_index: self._forecast_columns.time_series_index,
                self._input_columns.target: self._forecast_columns.target,
            }
        )

        selected_columns = list(
            set(
                (
                    *self._forecast_columns.ts_base_cols,
                    *self._forecast_columns.exogenous,
                )
            )
        )

        df = df[selected_columns].copy()

        # Fill gaps in the time series
        frequency_alias = Frequency.get_alias(self._forecast.freq, "pandas")

        df = fill_gaps(
            df,
            freq=frequency_alias,
            id_col=self._forecast_columns.time_series_index,
            time_col=self._forecast_columns.date,
        )

        # Fill missing values
        df = df.ffill()
        # Fill remaining NaNs with 0
        df = df.fillna(0)

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame through scaling and encoding.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the target variable to scale and
            features to encode.

        Returns:
        - pd.DataFrame: DataFrame with the scaled target variable.
        """
        logging.info("Scaling target variable")

        freq = self._forecast.freq
        cross_validation = self._cross_validation

        global_min_max_scaler = MinMaxScaler(
            feature_range=(0, 1),
        )

        non_cat_exog = [
            col
            for col in self._forecast_columns.exogenous
            if col not in self._forecast_columns.categorical
        ]
        if non_cat_exog:
            df[non_cat_exog] = global_min_max_scaler.fit_transform(df[non_cat_exog])

        local_scaler = TargetScalerFactory.create_scaler(
            self._preprocessing, cross_validation, freq, self._forecast
        )

        date_encoder = DateEncoder(freq=freq)
        self._forecast_columns.add_features(
            date_encoder.out_columns, feature_type="future_exogenous"
        )

        ml_forecast = MLForecast(
            models=[],
            freq=self._forecast.freq,
            target_transforms=[local_scaler],
            date_features=date_encoder.get_encoders(),
        )

        df = ml_forecast.preprocess(
            df,
            id_col=self._forecast_columns.time_series_index,
            time_col=self._forecast_columns.date,
            target_col=self._forecast_columns.target,
            static_features=self._forecast_columns.static,
        )

        category_encoder = CategoryEncoder(
            cv_cfg=cross_validation,
            freq=freq,
            forecast_columns=self._forecast_columns,
            horizon=self._forecast.horizon,
        )

        df = category_encoder.fit_transform(df)

        self._forecast_columns.replace_features(
            dict(
                zip(
                    self._forecast_columns.categorical,
                    category_encoder.out_columns,
                )
            )
        )

        stats_encoder = StatisticalFeaturesEncoder(
            cv_cfg=cross_validation,
            freq=freq,
            forecast_columns=self._forecast_columns,
            forecast=self._forecast,
        )
        df = stats_encoder.fit_transform(df)

        self._forecast_columns.remove_features(
            [self._forecast_columns.time_series_index], feature_type="static"
        )
        self._forecast_columns.add_features(
            stats_encoder.out_columns, feature_type="static"
        )

        return df
