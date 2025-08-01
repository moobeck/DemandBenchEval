from abc import ABC, abstractmethod
from typing import Any, List
from typing import Iterable
import pandas as pd
from statsforecast import StatsForecast
from mlforecast.auto import AutoMLForecast, AutoLightGBM
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from src.configurations.enums import Framework
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.forecasting import ForecastConfig
from src.configurations.enums import Frequency
from src.forecasting.foundation_model_base import FoundationModelWrapper
from src.configurations.cross_validation import CrossValDatasetConfig

import logging


class ForecastEngine(ABC):
    @abstractmethod
    def cross_validation(self, **kwargs: Any) -> pd.DataFrame:
        """
        Perform cross-validation for the forecasting engine.

        Args:
            **kwargs: Additional arguments for cross-validation.

        Returns:
            pd.DataFrame: Cross-validation results.
        """
        pass

    @staticmethod
    def _combine_results(
        dfs: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Concatenate and dedupe columns.
        """
        df_reset = [df.reset_index(drop=True) for df in dfs]

        combined = pd.concat(df_reset, axis=1).reset_index()
        # Drop any duplicated forecast columns, keep first
        return combined.loc[:, ~combined.columns.duplicated()].copy()


class FoundationModelEngine(ForecastEngine):
    """
    Engine for Foundation Models (FM) like TabPFN and TOTO.
    These models have special requirements and interfaces.
    """

    def __init__(self, models: List[FoundationModelWrapper], freq: str, **kwargs):
        """
        Initialize Foundation Model Engine.

        Parameters:
        -----------
        models : List[FoundationModelWrapper]
            List of foundation model instances (TabPFN, TOTO, etc.)
        freq : str
            Frequency string for time series
        """
        self.models = {model.alias: model for model in models}
        self.freq = freq
        self.kwargs = kwargs

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        cv_config: CrossValDatasetConfig,
        forecast_columns: ForecastColumnConfig = None,
        forecast_config: ForecastConfig = None,
    ) -> pd.DataFrame:
        """
        Perform cross-validation for foundation models.
        Foundation models use a different approach than traditional ML models.
        """

        results = []

        n_windows = cv_config.test.n_windows
        step_size = cv_config.test.step_size

        # Calculate time splits
        if forecast_config.freq == Frequency.DAILY:
            offset = pd.Timedelta(days=n_windows * step_size)
        elif forecast_config.freq == Frequency.WEEKLY:
            offset = pd.Timedelta(weeks=n_windows * step_size)
        else:
            raise ValueError(f"Unsupported frequency: {forecast_config.freq}")

        for model_name, model in self.models.items():
            logging.info(f"Cross-validating foundation model: {model_name}")
            window_results = []

            # For each cross-validation window
            for window in range(n_windows):
                # Calculate cutoff point for this window
                cutoff_offset = (
                    pd.Timedelta(days=window * step_size)
                    if forecast_config.freq == Frequency.DAILY
                    else pd.Timedelta(weeks=window * step_size)
                )
                cutoff = df[forecast_columns.date].max() - offset + cutoff_offset

                # Split data into train and test
                train_data = df[df[forecast_columns.date] <= cutoff]

                # Generate predictions
                model_df = model.predict(
                    X=train_data,
                    forecast_columns=forecast_columns,
                    horizon=h,  # Use h instead of forecast_config.horizon
                    freq=forecast_config.freq,
                )

                # Add cutoff column
                model_df["cutoff"] = cutoff

                # Merge with actual values from test data
                # First get the unique SKU IDs and dates from predictions
                merged_df = pd.merge(
                    model_df,
                    df[
                        [
                            forecast_columns.sku_index,
                            forecast_columns.date,
                            forecast_columns.target,
                        ]
                    ],
                    on=[forecast_columns.sku_index, forecast_columns.date],
                    how="left",
                )

                window_results.append(merged_df)

            # Combine all windows
            if window_results:
                model_results = pd.concat(window_results, ignore_index=True)
                results.append(model_results)

        # Combine results from all models
        combined_results = self._combine_results(results)

        return combined_results


class StatsForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine = StatsForecast(*args, **kw, verbose=True)

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        cv_config: CrossValDatasetConfig,
        id_col: str = None,
        target_col: str = None,
        time_col: str = None,
    ):
        n_windows = cv_config.test.n_windows
        step_size = cv_config.test.step_size
        refit = cv_config.test.refit

        return self._engine.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
            id_col=id_col,
            target_col=target_col,
            time_col=time_col,
        )


class AutoMLForecastEngine(ForecastEngine):
    def __init__(self, num_samples: int, *args, **kw):
        self._engine: AutoMLForecast = AutoMLForecast(*args, **kw)
        self.num_samples = num_samples


    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        cv_config: CrossValDatasetConfig,
        forecast_columns: ForecastColumnConfig = None,
        forecast_config: ForecastConfig = None,
        id_col: str = None,
        target_col: str = None,
        time_col: str = None,
    ):

        

        # Filter out the n_windows to get the df used to fit the model
        n_windows_val = cv_config.val.n_windows
        step_size_val = cv_config.val.step_size
        refit_val = cv_config.val.refit

        n_windows_test = cv_config.test.n_windows
        step_size_test = cv_config.test.step_size
        refit_test = cv_config.test.refit

        cutoff = self._engine.get_cutoff_date(
            max_date=df[forecast_columns.date].max(),
            freq=forecast_config.freq,
            split='val'
        )
        df_fit = df[df[forecast_columns.date] <= cutoff]

        # Fit the model with the filtered df
        self._engine = self._engine.fit(
            df=df_fit,
            h=h,
            n_windows=n_windows_val,
            step_size=step_size_val,
            num_samples=self.num_samples,
            refit=refit_val,
            id_col=id_col,
            target_col=target_col,
            time_col=time_col,
        )

        # Now get the models to do the cross-validation
        dfs = []
        models: Iterable[MLForecast] = self._engine.models_.values()
        for model in models:

            # Get the cross-validation results for each model
            df = model.cross_validation(
                df=df,
                n_windows=n_windows_test,
                step_size=step_size_test,
                refit=refit_test,
                h=h,
                max_horizon=forecast_config.horizon,
                id_col=id_col,
                target_col=target_col,
                time_col=time_col
            )

            dfs.append(df)
        # Combine the results from all models
        combined = self._combine_results(dfs)

        return combined


class NeuralForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine = NeuralForecast(*args, **kw)

    def cross_validation(
        self,
        df: pd.DataFrame,
        cv_config: CrossValDatasetConfig,
        id_col: str = None,
        target_col: str = None,
        time_col: str = None,
        static_df: pd.DataFrame = None,
    ):
        return self._engine.cross_validation(
            df=df,
            static_df=static_df,
            n_windows=cv_config.test.n_windows,
            step_size=cv_config.test.step_size,
            refit=cv_config.test.refit,
            verbose=True,
            id_col=id_col,
            target_col=target_col,
            time_col=time_col,
        )
