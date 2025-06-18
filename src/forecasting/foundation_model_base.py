"""
Abstract base class for foundation model wrappers.

This module defines the common interface that all foundation model wrappers
(TabPFN, TOTO, etc.) must implement to work with the forecasting framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np
import pandas as pd


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
    
    @abstractmethod
    def fit(
        self, 
        y: Union[pd.Series, pd.DataFrame, np.ndarray],
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        static_features: Optional[Union[pd.DataFrame, dict]] = None
    ) -> 'FoundationModelWrapper':
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
        horizon: int,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        static_features: Optional[Union[pd.DataFrame, dict]] = None
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
    
    @abstractmethod
    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'FoundationModelWrapper':
        """Set model parameters."""
        pass 