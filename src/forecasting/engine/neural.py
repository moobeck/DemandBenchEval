import pandas as pd
from neuralforecast import NeuralForecast
from src.configurations.evaluation.cross_validation import CrossValidationConfig
from src.forecasting.engine.abstract import ForecastEngine
from src.configurations.data.forecast_column import ForecastColumnConfig
from typing import List


class NeuralForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine = NeuralForecast(*args, **kw)

    @property
    def models(self) -> List:
        """
        Expose the underlying NeuralForecast models (used for logging/search results).
        """
        return getattr(self._engine, "models", [])


    @staticmethod
    def cv_inputs() -> List[str]:
        """
        Return the list of input parameter names required for cross-validation.
        """
        return [
            "df",
            "cv_config",
            "forecast_columns",
            "static_df",
        ]

    def cross_validation(
        self,
        df: pd.DataFrame,
        cv_config: CrossValidationConfig,
        forecast_columns: ForecastColumnConfig,
        static_df: pd.DataFrame = None,
    ):
        return self._engine.cross_validation(
            df=df,
            static_df=static_df,
            n_windows=cv_config.n_windows,
            step_size=cv_config.step_size,
            verbose=True,
            id_col=forecast_columns.time_series_index,
            target_col=forecast_columns.target,
            time_col=forecast_columns.date,
            refit=cv_config.refit,
        )
