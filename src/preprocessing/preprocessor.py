import pandas as pd
from typing import List, Literal, Union
from sklearn.preprocessing import MinMaxScaler
from utilsforecast.preprocessing import fill_gaps
from demandbench.datasets import Dataset
from src.preprocessing.scaler.scaler import TargetScalerFactory
from src.preprocessing.encoder.date_encoder import DateEncoder
from src.preprocessing.encoder.category_encoder import CategoryEncoder
from src.preprocessing.encoder.statistical_encoder import StatisticalFeaturesEncoder
from src.configurations.utils.enums import FrequencyType
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
        preprocessing: PreprocessingConfig,
        forecast_columns: ForecastColumnConfig,
        forecast: ForecastConfig,
        cross_validation: CrossValidationConfig,
    ):

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

        # Some datasets (e.g., Favorita) keep store-level identifiers only in the
        # feature chunks. If required columns are missing after the default merge,
        # backfill them from `dataset.features`.
        required_cols = set(
            list(self._forecast_columns.ts_base_cols) + list(self._forecast_columns.exogenous)
        )
        missing_cols = [col for col in required_cols if col not in self.df_merged.columns]

        if missing_cols and hasattr(self._dataset, "features"):
            feature_cols = [
                col for col in missing_cols if col in self._dataset.features.columns
            ]
            if feature_cols:
                # Join using lightweight keys available in both sides.
                join_keys = [
                    col
                    for col in ["idx", self._forecast_columns.date, "frequency"]
                    if col in self.df_merged.columns
                    and col in self._dataset.features.columns
                ]
                if not join_keys:
                    logging.warning(
                        "Missing columns %s but no common join keys between merged data "
                        "and features. Skipping backfill.",
                        feature_cols,
                    )
                    return

                features_df = (
                    self._dataset.features.select(join_keys + feature_cols)
                    .to_pandas()
                )
                before_cols = set(self.df_merged.columns)
                self.df_merged = self.df_merged.merge(
                    features_df, on=join_keys, how="left"
                )
                added = set(self.df_merged.columns) - before_cols
                logging.info(
                    "Added missing columns from feature chunks: %s",
                    sorted(added & set(feature_cols)),
                )
            else:
                logging.warning(
                    "Required columns %s are missing and not present in features.",
                    missing_cols,
                )

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

        # Print all columns in the DataFrame
        logging.info(f"Columns in the DataFrame: {self.df_merged.columns.tolist()}")

        if self.df_merged is None:
            raise ValueError("Data not merged. Call merge() first.")

        time_series_col = self._forecast_columns.time_series_index
        date_col = self._forecast_columns.date

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

    def prepare_forecasting_data(self) -> pd.DataFrame:
        """Prepare a pandas DataFrame for forecasting."""

        logging.info("Preparing data for forecasting")

        if self.df_merged is None:
            raise ValueError("Data not merged. Call merge() first.")

        selected_columns = list(
            set(
                (
                    *self._forecast_columns.ts_base_cols,
                    *self._forecast_columns.exogenous,
                )
            )
        )

        df = self.df_merged[selected_columns].copy()

        # Ensure the configured id column exists; if missing but an equivalent
        # identifier is present (e.g., storeID/productID), duplicate it to the
        # expected name instead of changing the configured column.
        id_col = self._forecast_columns.time_series_index
        candidate_ids = [
            self._forecast_columns.store_index,
            self._forecast_columns.product_index,
            "timeSeriesID",
        ]
        if id_col not in df.columns:
            for candidate in candidate_ids:
                if candidate in df.columns:
                    logging.warning(
                        "Time-series id column '%s' not found; copying from '%s'.",
                        id_col,
                        candidate,
                    )
                    df[id_col] = df[candidate]
                    break
            else:
                raise KeyError(
                    f"Time-series id column '{id_col}' not found in data columns: "
                    f"{list(df.columns)}"
                )

        # Fill gaps in the time series
        frequency_alias = FrequencyType.get_alias(self._forecast.freq, "pandas")

        df = fill_gaps(
            df,
            freq=frequency_alias,
            id_col=self._forecast_columns.time_series_index,
            time_col=self._forecast_columns.date,
        )

        # Fill missing values per series to avoid cross-series leakage on ffill/bfill.
        id_col = self._forecast_columns.time_series_index
        date_col = self._forecast_columns.date
        df = df.sort_values([id_col, date_col])
        df = df.groupby(id_col).ffill()
        df = df.groupby(id_col).bfill()

        # Only fill remaining numeric NaNs with 0; leave categoricals untouched.
        numeric_cols = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(0)

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

        non_numeric_exog = [
            col for col in non_cat_exog if not pd.api.types.is_numeric_dtype(df[col])
        ]

        if non_numeric_exog:
            logging.warning(
                "Detected non-numeric exogenous columns; treating as categorical: %s",
                non_numeric_exog,
            )
            for col in non_numeric_exog:
                if col not in self._forecast_columns.categorical:
                    self._forecast_columns.categorical.append(col)
            non_cat_exog = [col for col in non_cat_exog if col not in non_numeric_exog]

        if non_cat_exog:
            df[non_cat_exog] = global_min_max_scaler.fit_transform(df[non_cat_exog])

        local_scaler = TargetScalerFactory.create_scaler(
            self._preprocessing, cross_validation, freq, self._forecast
        )

        date_encoder = DateEncoder(freq=freq)
        date_encoders = date_encoder.get_encoders()
        # Register date-derived features after they are populated
        self._forecast_columns.add_features(
            date_encoder.out_columns, feature_type="future_exogenous"
        )

        # Ensure static features are truly static; otherwise treat them as dynamic
        if self._forecast_columns.static:
            id_col = self._forecast_columns.time_series_index
            varying_static = []
            for col in list(self._forecast_columns.static):
                if col not in df.columns:
                    continue
                if df.groupby(id_col)[col].nunique(dropna=False).gt(1).any():
                    varying_static.append(col)

            if varying_static:
                logging.warning(
                    "Static features change over time; moving to past_exogenous: %s",
                    varying_static,
                )
                self._forecast_columns.remove_features(
                    varying_static, feature_type="static"
                )
                self._forecast_columns.add_features(
                    varying_static, feature_type="past_exogenous"
                )

        ml_forecast = MLForecast(
            models=[],
            freq=self._forecast.freq,
            target_transforms=[local_scaler],
            date_features=date_encoders,
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
