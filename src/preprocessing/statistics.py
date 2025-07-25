

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.cross_validation import CrossValidationConfig
from src.configurations.enums import Frequency

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
        freq: Frequency,
        quantiles: Optional[List[float]] = None
    ):
        self.df = df.copy()
        self._forecast_columns = forecast_columns
        self._cv_cfg = cross_validation
        self._freq = freq
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.25, 0.5, 0.75, 0.9]

    def get_training_data(self) -> pd.DataFrame:
        """
        Get the training data based on cross-validation config and frequency.
        """
        n_windows = self._cv_cfg.test.n_windows
        step_size = self._cv_cfg.test.step_size

        if self._freq == Frequency.DAILY:
            offset = pd.Timedelta(days=n_windows * step_size)
        elif self._freq == Frequency.WEEKLY:
            offset = pd.Timedelta(weeks=n_windows * step_size)
        else:
            raise ValueError(f"Unsupported frequency: {self._freq}")

        cutoff = self.df[self._forecast_columns.date].max() - offset
        df_train = self.df[self.df[self._forecast_columns.date] <= cutoff]
        return df_train

    def compute_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for each SKU in the training data.
        Returns a DataFrame: index=sku, columns=[mean, std, min, max, quantile_xx]
        """
        df_train = self.get_training_data()
        rows = []
        for sku, group in df_train.groupby(self._forecast_columns.sku_index):
            target = group[self._forecast_columns.target].dropna().values
            if len(target) == 0:
                continue
            row = {
                self._forecast_columns.sku_index: sku,
                "mean": np.mean(target),
                "std": np.std(target),
                "min": np.min(target),
                "max": np.max(target),
            }
            for q in self.quantiles:
                row[f"quantile_{q}"] = np.quantile(target, q)
            rows.append(row)
        return pd.DataFrame(rows).set_index(self._forecast_columns.sku_index)
