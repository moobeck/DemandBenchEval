from abc import ABC, abstractmethod
from typing import Any, List
from typing import Iterable
import pandas as pd
from statsforecast import StatsForecast
from mlforecast.auto import AutoMLForecast
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from src.configurations.enums import Framework
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.forecasting import ForecastConfig



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


class StatsForecastEngine(ForecastEngine):
    def __init__(self, *args, **kw):
        self._engine = StatsForecast(*args, **kw)

    def cross_validation(
        self,
        df: pd.DataFrame,
        cv_windows: int,
        step_size: int,
        refit: bool = False,
        **kwargs,
    ):
        return self._engine.cross_validation(
            df=df,
            cv_windows=cv_windows,
            step_size=step_size,
            refit=refit,
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
        combined = pd.concat(dfs, axis=1).reset_index()
        # Drop any duplicated forecast columns, keep first
        return combined.loc[:, ~combined.columns.duplicated()].copy()

    def cross_validation(
        self,
        df: pd.DataFrame,
        cv_windows: int,
        step_size: int,
        refit: bool = False,
        forecast_columns: ForecastColumnConfig = None,
        forecast_config: ForecastConfig = None,
        **kwargs,
    ):

        if refit:
            raise ValueError("refit=True is not supported for AutoMLForecastEngine.")

        # Filter out the cv_windows to get the df used to fit the model

        # Calculate the offset based on the frequency ('D'. 'W', raise error if not supported)
        if forecast_config.freq == "D":
            offset = pd.Timedelta(days=cv_windows * step_size)
        elif forecast_config.freq == "W":
            offset = pd.Timedelta(weeks=cv_windows * step_size)
        else:
            raise ValueError(f"Unsupported frequency: {forecast_config.freq}")

        cutoff = df[forecast_columns.date].max() - offset
        df = df[df[forecast_columns.date] <= cutoff]

        # Fit the model with the filtered df
        self._engine = self._engine.fit(
            df=df,
            cv_windows=cv_windows,
            step_size=step_size,
        )

        # Now get the models to do the cross-validation
        dfs = []
        models: Iterable[MLForecast] = self._engine.models_.values()
        for model in models:

            # Get the cross-validation results for each model
            df = model.cross_validation(
                df=df,
                cv_windows=cv_windows,
                step_size=step_size,
                refit=refit,
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
        cv_windows: int,
        step_size: int,
        refit: bool = False,
        **kwargs,
    ):
        return self._engine.cross_validation(
            df=df,
            cv_windows=cv_windows,
            step_size=step_size,
            refit=refit,
        )
