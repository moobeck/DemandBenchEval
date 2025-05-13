
import pandas as pd
from utilsforecast.evaluation import evaluate
from configurations.metrics import MetricConfig
from configurations.forecast_column import ForecastColumnConfig


class Evaluator:
    """
    A class to evaluate the performance of a forecasting model using various metrics.
    """

    def __init__(
            self, 
            metric_config: MetricConfig,
            forecast_columns: ForecastColumnConfig,
    ): 
        
        self._metric_config = metric_config
        self._forecast_columns = forecast_columns

        self.metrics = self._metric_config.metrics


    def evaluate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        model_cols = [
            col for col in df.columns
            if col not in [
                self._forecast_columns.sku_index,
                self._forecast_columns.date,
                self._forecast_columns.target,
                self._forecast_columns.cutoff,
            ]
        ]

        return evaluate(
            df=df,
            models=model_cols,
            target_col=self._forecast_columns.target,
            time_col=self._forecast_columns.date,
            id_col=self._forecast_columns.sku_index,
            metrics=list(self.metrics.values()),
            **kwargs
        )


        

    