import pandas as pd
from neuralforecast import NeuralForecast
from src.configurations.evaluation.cross_validation import CrossValidationConfig
from src.forecasting.engine.abstract import ForecastEngine


class NeuralForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine = NeuralForecast(*args, **kw)

    def cross_validation(
        self,
        df: pd.DataFrame,
        cv_config: CrossValidationConfig,
        id_col: str = None,
        target_col: str = None,
        time_col: str = None,
        static_df: pd.DataFrame = None,
    ):
        return self._engine.cross_validation(
            df=df,
            static_df=static_df,
            n_windows=cv_config.n_windows,
            step_size=cv_config.step_size,
            refit=cv_config.refit,
            verbose=True,
            id_col=id_col,
            target_col=target_col,
            time_col=time_col,
        )
