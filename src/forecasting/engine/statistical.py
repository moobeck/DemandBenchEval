import pandas as pd
from statsforecast import StatsForecast
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.evaluation.cross_validation import CrossValDatasetConfig
from src.configurations.forecasting.forecasting import ForecastConfig
from src.forecasting.engine.abstract import ForecastEngine
from src.forecasting.engine.utils.quantile_forecaster import QuantileForecaster
from src.utils.quantile import QuantileUtils


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
        forecast_config: ForecastConfig,
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

        quantiles = QuantileUtils.create_quantiles(forecast_config.stats.quantile)

        cv_results = self._add_quantiles(cv_results, quantiles, forecast_columns)
        return cv_results

    def _insample_forecast(self) -> pd.DataFrame:
        return self._engine.cross_validation_fitted_values()

    def _add_quantiles(
        self, df: pd.DataFrame, quantiles: list, forecast_columns: ForecastColumnConfig
    ) -> pd.DataFrame:
        """Add quantile columns to the forecast results DataFrame."""
        in_sample_fcst = self._insample_forecast()

        quantile_forecaster = QuantileForecaster(
            models=self._get_model_names(), forecast_columns=forecast_columns
        )

        return quantile_forecaster.add_quantiles(df, in_sample_fcst, quantiles)
