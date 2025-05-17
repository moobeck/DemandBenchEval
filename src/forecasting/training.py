import logging
import pandas as pd
from statsforecast import StatsForecast
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.forecasting import ForecastConfig
from src.configurations.enums import Framework


class ForecastTrainer:
    """
    Orchestrates model fitting
    for multiple models from multiple frameworks
    """

    def __init__(
        self,
        forecast_config: ForecastConfig,
        forecast_columns: ForecastColumnConfig = None,
    ):

        self._forecast_config = forecast_config
        self._forecast_columns = forecast_columns
        self.frameworks = {
            Framework.STATS: StatsForecast(
                models=list(forecast_config.models[Framework.STATS].values()),
                freq=self._forecast_config.freq,
            ),
            Framework.ML: MLForecast(
                models=list(forecast_config.models[Framework.ML].values()),
                freq=self._forecast_config.freq,
                lags=self._forecast_config.lags,
                date_features=self._forecast_config.date_features,
            ),
            Framework.NEURAL: NeuralForecast(
                models=list(forecast_config.models[Framework.NEURAL].values()),
                freq=self._forecast_config.freq,
            ),
        }

    def cross_validate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Perform cross-validation on the given DataFrame.
        """
        # Placeholder for cross-validation logic

        logging.info("Starting cross-validation...")

        cv_dfs = []

        for framework, forecast_engine in self.frameworks.items():
            # Perform cross-validation for each framework
            logging.info(f"Cross-validating with {framework.name}...")

            ts_cols = self._forecast_columns.ts_base_cols

            cv_kwargs = kwargs.copy()

            if framework == Framework.ML:
                cv_kwargs["static_features"] = self._forecast_columns.static
                ts_cols += self._forecast_columns.static
            elif framework == Framework.NEURAL:
                cv_kwargs["static_df"] = df[
                    [self._forecast_columns.sku_index, self._forecast_columns.date]
                    + self._forecast_columns.static
                ]

            cv_df: pd.DataFrame = forecast_engine.cross_validation(
                df=df[ts_cols],
                h=self._forecast_config.horizon,
                id_col=self._forecast_columns.sku_index,
                target_col=self._forecast_columns.target,
                time_col=self._forecast_columns.date,
                **cv_kwargs,
            ).set_index(
                [self._forecast_columns.sku_index, self._forecast_columns.date],
                drop=True,
            )

            # Append the cross-validation DataFrame to the list
            cv_dfs.append(cv_df)

        df_combined = pd.concat(cv_dfs, axis=1).reset_index()

        return df_combined.loc[:, ~df_combined.columns.duplicated()].copy()
