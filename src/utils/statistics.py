import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.forecasting import ForecastConfig
from src.configurations.evaluation.cross_validation import CrossValidationConfig
from src.configurations.utils.enums import FrequencyType


class SKUStatistics:
    """
    Computes quantile, mean, std, min, and max statistics for each SKU in training data,
    using cross-validation logic from ForecastColumnConfig and CrossValidationConfig.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        forecast_columns: ForecastColumnConfig,
        cross_validation: CrossValidationConfig,
        forecast: ForecastConfig,
        freq: FrequencyType,
        quantiles: Optional[List[float]] = None,
    ):
        self.df = df.copy()
        self._forecast_columns = forecast_columns
        self._cv_cfg = cross_validation
        self._forecast = forecast
        self._freq = freq
        self.quantiles = (
            quantiles if quantiles is not None else [0.1, 0.25, 0.5, 0.75, 0.9]
        )

    def compute_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for each SKU in the training data.
        Returns a DataFrame: index=sku, columns=[mean, std, min, max, quantile_xx]
        """

        cutoff = self._cv_cfg.get_cutoff_date(
            max_date=self.df[self._forecast_columns.date].max(),
            freq=self._freq,
            horizon=self._forecast.horizon,
        )
        df_train = self.df[self.df[self._forecast_columns.date] <= cutoff]
        rows = []
        for sku, group in df_train.groupby(self._forecast_columns.time_series_index):
            target = group[self._forecast_columns.target].dropna().values
            if len(target) == 0:
                continue
            row = {
                self._forecast_columns.time_series_index: sku,
                "mean": np.mean(target),
                "std": np.std(target),
                "min": np.min(target),
                "max": np.max(target),
            }
            for q in self.quantiles:
                row[f"quantile_{q}"] = np.quantile(target, q)
            rows.append(row)
        return pd.DataFrame(rows).set_index(self._forecast_columns.time_series_index)
