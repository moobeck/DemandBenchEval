import pandas as pd
from statsforecast import StatsForecast
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.evaluation.cross_validation import CrossValDatasetConfig
from src.forecasting.engine.abstract import ForecastEngine


class StatsForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine = StatsForecast(*args, **kw, verbose=True)

    def _get_model_names(self):
        return [str(model) for model in self._engine.models]

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        cv_config: CrossValDatasetConfig,
        forecast_columns: ForecastColumnConfig,
    ):
        n_windows = cv_config.test.n_windows
        step_size = cv_config.test.step_size
        refit = cv_config.test.refit

        cv_results = self._engine.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
            id_col=forecast_columns.time_series_index,
            target_col=forecast_columns.target,
            time_col=forecast_columns.date,
            fitted=True,
        )

        return cv_results

