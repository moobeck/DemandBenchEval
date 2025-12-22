import pandas as pd
from statsforecast import StatsForecast
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.evaluation.cross_validation import CrossValidationConfig
from src.forecasting.engine.abstract import ForecastEngine
from typing import List

class StatsForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine = StatsForecast(*args, **kw, verbose=True)

    def _get_model_names(self):
        return [str(model) for model in self._engine.models]

    @staticmethod
    def cv_inputs() -> List[str]:
        """
        Return the list of input parameter names required for cross-validation.
        """
        return [
            "df",
            "h",
            "cv_config",
            "forecast_columns",
        ]

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        cv_config: CrossValidationConfig,
        forecast_columns: ForecastColumnConfig,
    ):
        n_windows = cv_config.n_windows
        step_size = cv_config.step_size

        cv_results = self._engine.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            id_col=forecast_columns.time_series_index,
            target_col=forecast_columns.target,
            time_col=forecast_columns.date,
            refit=cv_config.refit,
        )

        return cv_results
