import pandas as pd
from statsforecast import StatsForecast
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.forecasting import ForecastConfig
from src.configurations.evaluation.cross_validation import CrossValidationConfig
from src.utils.quantile import QuantileUtils
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
            "forecast_config",
        ]

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        cv_config: CrossValidationConfig,
        forecast_columns: ForecastColumnConfig,
        forecast_config: ForecastConfig,
    ):
        n_windows = cv_config.n_windows
        step_size = cv_config.step_size

        quantiles = QuantileUtils.create_quantiles(
            forecast_config.statistical.quantile
        )
        levels = QuantileUtils.quantiles_to_level(quantiles)

        cv_results = self._engine.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            id_col=forecast_columns.time_series_index,
            target_col=forecast_columns.target,
            time_col=forecast_columns.date,
            refit=cv_config.refit,
            level=levels,
        )

        return cv_results
