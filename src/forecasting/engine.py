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
from src.configurations.cross_validation import CrossValDatasetConfig, CrossValidationConfig

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
    
    
    def create_time_offset(self, freq: Frequency, periods: int) -> pd.Timedelta:
        """
        Create time offset based on frequency and number of periods.
        
        Parameters:
        -----------
        freq : Frequency
            Data frequency (DAILY or WEEKLY)
        periods : int
            Number of periods
            
        Returns:
        --------
        pd.Timedelta
            Time offset
        """
        if freq == Frequency.DAILY:
            return pd.Timedelta(days=periods)
        elif freq == Frequency.WEEKLY:
            return pd.Timedelta(weeks=periods)
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
    
    def validate_cv_params(self, n_windows: int, step_size: int, h: int) -> None:
        """
        Validate cross-validation parameters.
        
        Parameters:
        -----------
        n_windows : int
            Number of validation windows
        step_size : int
            Step size between windows
        h : int
            Forecast horizon
            
        Raises:
        -------
        ValueError
            If parameters are invalid
        """
        if n_windows <= 0:
            raise ValueError("n_windows must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if h <= 0:
            raise ValueError("forecast horizon (h) must be positive")
    
    def extract_cv_config(self, cv_config: CrossValDatasetConfig, split: str = 'test') -> tuple[int, int, bool]:
        """
        Extract cross-validation configuration for a specific split.
        
        Parameters:
        -----------
        cv_config : CrossValDatasetConfig
            Cross-validation configuration (TypedDict)
        split : str
            Split type ('test' or 'val')
            
        Returns:
        --------
        tuple[int, int, bool]
            n_windows, step_size, refit
        """
        if split == 'test':
            config = cv_config['test']
            return config.n_windows, config.step_size, config.refit
        elif split == 'val':
            config = cv_config['val']
            return config.n_windows, config.step_size, config.refit
        else:
            raise ValueError(f"Unsupported split type: {split}")

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

        # Extract cross-validation configuration
        n_windows, step_size, _ = self.extract_cv_config(cv_config, 'test')
        
        # Validate parameters
        self.validate_cv_params(n_windows, step_size, h)

        # Calculate time splits
        offset = self.create_time_offset(forecast_config.freq, n_windows * step_size)

        for model_name, model in self.models.items():
            logging.info(f"Cross-validating foundation model: {model_name}")
            window_results = []

            # For each cross-validation window
            for window in range(n_windows):
                # Calculate cutoff point for this window
                cutoff_offset = self.create_time_offset(forecast_config.freq, window * step_size)
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
        # Extract and validate cross-validation configuration
        n_windows, step_size, refit = self.extract_cv_config(cv_config, 'test')
        self.validate_cv_params(n_windows, step_size, h)

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

        # Extract cross-validation configuration
        n_windows_val, step_size_val, refit_val = self.extract_cv_config(cv_config, 'val')
        n_windows_test, step_size_test, refit_test = self.extract_cv_config(cv_config, 'test')
        
        # Validate cross-validation parameters
        self.validate_cv_params(n_windows_val, step_size_val, h)
        self.validate_cv_params(n_windows_test, step_size_test, h)

        # Create temporary CrossValidationConfig instance to use its get_cutoff_date method
        temp_cv_config = CrossValidationConfig(data={})
        temp_cv_config.val = cv_config['val'] 
        
        # Calculate cutoff date using CrossValidationConfig method
        cutoff = temp_cv_config.get_cutoff_date(
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
                time_col=time_col,
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
        # Extract cross-validation configuration
        n_windows, step_size, refit = self.extract_cv_config(cv_config, 'test')
        
        return self._engine.cross_validation(
            df=df,
            static_df=static_df,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
            verbose=True,
            id_col=id_col,
            target_col=target_col,
            time_col=time_col,
        )
