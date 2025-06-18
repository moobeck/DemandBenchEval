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
        self.models = {model.__class__.__name__: model for model in models}
        self.freq = freq
        self.kwargs = kwargs
        
    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int,
        step_size: int,
        refit: bool = False,
        forecast_columns: ForecastColumnConfig = None,
        forecast_config: ForecastConfig = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Perform cross-validation for foundation models.
        Foundation models use a different approach than traditional ML models.
        """
        logging.info(f"Starting foundation model cross-validation with {len(self.models)} models...")
        
        results = []
        
        # Calculate time splits
        if forecast_config.freq == Frequency.DAILY:
            offset = pd.Timedelta(days=n_windows * step_size)
        elif forecast_config.freq == Frequency.WEEKLY:
            offset = pd.Timedelta(weeks=n_windows * step_size)
        else:
            raise ValueError(f"Unsupported frequency: {forecast_config.freq}")
        
        # Get unique time series IDs
        unique_ids = df[forecast_columns.sku_index].unique()
        
        for model_name, model in self.models.items():
            logging.info(f"Cross-validating foundation model: {model_name}")
            
            model_results = []
            
            # For each time series ID
            for series_id in unique_ids:
                series_df = df[df[forecast_columns.sku_index] == series_id].copy()
                series_df = series_df.sort_values(forecast_columns.date)
                
                # For each cross-validation window
                for window in range(n_windows):
                    # Calculate cutoff point
                    cutoff_offset = pd.Timedelta(days=window * step_size) if forecast_config.freq == Frequency.DAILY else pd.Timedelta(weeks=window * step_size)
                    cutoff = series_df[forecast_columns.date].max() - offset + cutoff_offset
                    
                    # Split data
                    train_data = series_df[series_df[forecast_columns.date] <= cutoff]
                    test_data = series_df[
                        (series_df[forecast_columns.date] > cutoff) & 
                        (series_df[forecast_columns.date] <= cutoff + pd.Timedelta(days=h) if forecast_config.freq == Frequency.DAILY else cutoff + pd.Timedelta(weeks=h))
                    ]
                    
                    if len(train_data) < 10 or len(test_data) == 0:
                        continue
                    
                    try:
                        # Prepare features
                        y_train = train_data[forecast_columns.target]
                        X_train = None
                        X_test = None
                        static_features = None
                        
                        # Handle exogenous variables and static features
                        if forecast_columns.exogenous:
                            exog_cols = [col for col in forecast_columns.exogenous if col in train_data.columns]
                            if exog_cols:
                                X_train = train_data[exog_cols]
                                X_test = test_data[exog_cols] if len(test_data) > 0 else None
                        
                        if forecast_columns.static:
                            static_cols = [col for col in forecast_columns.static if col in train_data.columns]
                            if static_cols:
                                static_features = train_data[static_cols].iloc[0].to_dict()
                        
                        # Fit and predict
                        if model.supports_exogenous:
                            model.fit(y_train, X=X_train, static_features=static_features)
                            predictions = model.predict(len(test_data), X=X_test, static_features=static_features)
                        else:
                            model.fit(y_train)
                            predictions = model.predict(len(test_data))
                        
                        # Handle multivariate predictions
                        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                            # For multivariate, take first series (primary target)
                            predictions = predictions[:, 0]
                        
                        # Create result DataFrame
                        for i, (idx, row) in enumerate(test_data.iterrows()):
                            if i < len(predictions):
                                model_results.append({
                                    forecast_columns.sku_index: series_id,
                                    forecast_columns.date: row[forecast_columns.date],
                                    forecast_columns.target: row[forecast_columns.target],
                                    f'{model_name}': predictions[i],
                                    'cutoff': cutoff
                                })
                    
                    except Exception as e:
                        logging.warning(f"Failed to process series {series_id}, window {window} with {model_name}: {e}")
                        continue
            
            if model_results:
                model_df = pd.DataFrame(model_results)
                results.append(model_df)
        
        # Combine results from all models
        if results:
            # Merge all model results
            combined = results[0]
            for result_df in results[1:]:
                merge_cols = [forecast_columns.sku_index, forecast_columns.date, 'cutoff']
                combined = combined.merge(
                    result_df,
                    on=merge_cols,
                    how='outer',
                    suffixes=('', '_y')
                )
                # Drop duplicate target columns
                target_y = f'{forecast_columns.target}_y'
                if target_y in combined.columns:
                    combined = combined.drop(columns=[target_y])
            
            return combined
        else:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=[
                forecast_columns.sku_index,
                forecast_columns.date,
                forecast_columns.target,
                'cutoff'
            ])


class StatsForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine = StatsForecast(*args, **kw, verbose=True)

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int,
        step_size: int,
        refit: bool = False,
        **kwargs,
    ):
        return self._engine.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
            **kwargs,
        )


class AutoMLForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine: AutoMLForecast = AutoMLForecast(*args, **kw)

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

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int,
        step_size: int,
        refit: bool = False,
        forecast_columns: ForecastColumnConfig = None,
        forecast_config: ForecastConfig = None,
        **kwargs,
    ):

        if refit:
            raise ValueError("refit=True is not supported for AutoMLForecastEngine.")

        # Filter out the n_windows to get the df used to fit the model

        # Calculate the offset based on the frequency ('D'. 'W', raise error if not supported)
        if forecast_config.freq == Frequency.DAILY:
            offset = pd.Timedelta(days=n_windows * step_size)
        elif forecast_config.freq == Frequency.WEEKLY:
            offset = pd.Timedelta(weeks=n_windows * step_size)
        else:
            raise ValueError(f"Unsupported frequency: {forecast_config.freq}")

        cutoff = df[forecast_columns.date].max() - offset
        df_fit = df[df[forecast_columns.date] <= cutoff]

        # Fit the model with the filtered df
        self._engine = self._engine.fit(
            df=df_fit,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            num_samples=10,
            refit=refit,
            **kwargs,
        )

        # Now get the models to do the cross-validation
        dfs = []
        models: Iterable[MLForecast] = self._engine.models_.values()
        for model in models:

            # Get the cross-validation results for each model
            df = model.cross_validation(
                df=df,
                n_windows=n_windows,
                step_size=step_size,
                refit=refit,
                h=h,
                max_horizon=forecast_config.horizon,
                **kwargs,
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
        n_windows: int,
        step_size: int,
        refit: bool = False,
        **kwargs,
    ):
        return self._engine.cross_validation(
            df=df,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
            verbose=True,
            **kwargs,
        )
