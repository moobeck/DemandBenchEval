"""
Abstract base class for foundation model wrappers.

This module defines the common interface that all foundation model wrappers
must implement to work with the forecasting framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from src.configurations.forecast_column import ForecastColumnConfig


class FoundationModelWrapper(ABC):
    """
    Abstract base class for foundation model wrappers.

    This ensures a consistent interface across all foundation models
    and eliminates the need for hasattr checks in the engine.
    """

    # Class attributes that subclasses should set
    supports_exogenous: bool = False
    supports_multivariate: bool = False
    is_probabilistic: bool = False
    alias = None  # Should be set by subclasses to identify the model

    def fit(
        self,
        y: Union[pd.Series, pd.DataFrame, np.ndarray],
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> "FoundationModelWrapper":
        """
        Fit the foundation model to training data.

        Parameters:
        -----------
        y : Series, DataFrame, or array
            Target time series data
        X : DataFrame or array, optional
            Exogenous time-varying features
        static_features : DataFrame or dict, optional
            Static (time-invariant) features

        Returns:
        --------
        self : FoundationModelWrapper
            Returns self for method chaining
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
        forecast_columns: ForecastColumnConfig,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate point forecasts.

        Parameters:
        -----------
        horizon : int
            Number of periods to forecast
        X : DataFrame or array, optional
            Future exogenous variables
        static_features : DataFrame or dict, optional
            Static features for prediction

        Returns:
        --------
        np.ndarray
            Point forecasts
        """
        pass
