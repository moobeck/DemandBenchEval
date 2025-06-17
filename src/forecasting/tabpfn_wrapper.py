"""
TabPFN Wrapper for MLForecast Integration

This module provides a wrapper class for TabPFN Regressor that is compatible 
with the MLForecast framework used in this benchmarking system.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from tabpfn import TabPFNRegressor
from typing import Any, Optional
import warnings


class TabPFNWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for TabPFN Regressor that is compatible with MLForecast.
    
    TabPFN is a foundation model for tabular data that works particularly well
    on small to medium-sized datasets (up to ~10,000 samples).
    """
    
    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = 8,
        max_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize TabPFN wrapper.
        
        Parameters:
        -----------
        device : str, default="cpu"
            Device to use for inference ("cpu" or "cuda")
        n_estimators : int, default=8
            Number of estimators for TabPFN (default for regression)
        max_samples : int, default=10000
            Maximum number of samples to use for training (TabPFN limitation)
        random_state : int, optional
            Random state for reproducibility
        **kwargs : dict
            Additional parameters passed to TabPFNRegressor
        """
        self.device = device
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Initialize the actual TabPFN model
        self.model_ = None
        self._is_fitted = False
        
    def fit(self, X, y, **fit_params):
        """
        Fit TabPFN model to training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        **fit_params : dict
            Additional fit parameters (ignored)
            
        Returns:
        --------
        self : TabPFNWrapper
            Returns self for method chaining
        """
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Handle TabPFN's sample size limitation
        if len(X) > self.max_samples:
            warnings.warn(
                f"Dataset has {len(X)} samples, but TabPFN works best with <={self.max_samples} samples. "
                f"Randomly sampling {self.max_samples} samples for training.",
                UserWarning
            )
            # Random sampling
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            indices = np.random.choice(len(X), size=self.max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Initialize and fit TabPFN model
        try:
            self.model_ = TabPFNRegressor(
                device=self.device,
                **self.kwargs
            )
            self.model_.fit(X, y)
            self._is_fitted = True
            
        except Exception as e:
            # Fallback to CPU if GPU fails
            if self.device != "cpu":
                warnings.warn(
                    f"TabPFN failed on {self.device}, falling back to CPU: {str(e)}",
                    UserWarning
                )
                self.model_ = TabPFNRegressor(device="cpu", **self.kwargs)
                self.model_.fit(X, y)
                self._is_fitted = True
            else:
                raise e
                
        return self
    
    def predict(self, X):
        """
        Make predictions using the fitted TabPFN model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction
            
        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values
        """
        if not self._is_fitted:
            raise ValueError("This TabPFNWrapper instance is not fitted yet.")
            
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        
        return self.model_.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'device': self.device,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'random_state': self.random_state
        }
        if deep:
            params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key in ['device', 'n_estimators', 'max_samples', 'random_state']:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self


def create_tabpfn_regressor(**params):
    """
    Factory function to create TabPFN regressor compatible with MLForecast.
    
    Parameters:
    -----------
    **params : dict
        Parameters to pass to TabPFNWrapper
        
    Returns:
    --------
    TabPFNWrapper
        Configured TabPFN wrapper instance
    """
    return TabPFNWrapper(**params) 