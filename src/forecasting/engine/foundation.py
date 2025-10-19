from typing import Any, List
import pandas as pd
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.forecasting import ForecastConfig
from src.configurations.evaluation.cross_validation import CrossValDatasetConfig
from src.forecasting.models.foundation.utils import GluonTSForecaster
from src.utils.quantile import QuantileUtils
from src.forecasting.engine.abstract import ForecastEngine
import logging

FoundationModelWrapper = Any


class FoundationModelEngine(ForecastEngine):
    """
    Engine for Foundation Models (FM)
    """

    def __init__(self, models: List[FoundationModelWrapper], freq: str, **kwargs):
        """
        Initialize Foundation Model Engine.

        Parameters:
        -----------
        models : List[FoundationModelWrapper]
            List of foundation model instances
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
        static_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Perform cross-validation for foundation models.
        """

        results = []

        step_size = cv_config.test.step_size
        n_windows = cv_config.test.n_windows
        quantiles = QuantileUtils.create_quantiles(forecast_config.foundation.quantile)

        for model_name, model in self.models.items():
            logging.info(f"Cross-validating foundation model: {model_name}")

            # Give type hint that model is GluonTSForecaster
            model: GluonTSForecaster

            fcst = model.cross_validation(
                df=df,
                static_df=static_df,
                horizon=h,
                step_size=step_size,
                quantiles=quantiles,
                n_windows=n_windows,
                id_col=forecast_columns.time_series_index,
                target_col=forecast_columns.target,
                time_col=forecast_columns.date,
            )
            results.append(fcst)

        # Combine results from all models
        combined_results = self._combine_results(results)

        return combined_results
