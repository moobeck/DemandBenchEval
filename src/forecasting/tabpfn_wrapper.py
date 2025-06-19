"""
TabPFN Wrapper for MLForecast Integration

This module provides a wrapper class for TabPFN Regressor that is compatible 
with the MLForecast framework used in this benchmarking system.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from tabpfn import TabPFNRegressor
from typing import Any, Optional, Union
import warnings
from .foundation_model_base import FoundationModelWrapper


class TabPFNWrapper(BaseEstimator, RegressorMixin, FoundationModelWrapper):
    """
    Wrapper for TabPFN Regressor that is compatible with MLForecast.

    TabPFN is a foundation model for tabular data that works particularly well
    on small to medium-sized datasets (up to ~10,000 samples).
    """

    # Foundation model capabilities
    supports_exogenous = (
        False  # TabPFN doesn't support exogenous features in time series context
    )
    supports_multivariate = False  # TabPFN is univariate
    is_probabilistic = False  # TabPFN provides point predictions

    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = 8,
        max_samples: int = 10000,
        random_state: Optional[int] = None,
        scaling: bool = True,
        **kwargs,
    ):
        """
        Initialize TabPFN wrapper.

        Parameters:
        -----------
        device : str, default="cpu"
            Device to use for inference ("cpu" or "cuda")
        n_estimators : int, default=8
            Number of estimators in the TabPFN ensemble. TabPFN aggregates predictions
            from multiple forward passes with slightly different input data prompts.
        max_samples : int, default=10000
            Maximum number of samples to use for training (TabPFN limitation)
        random_state : int, optional
            Random state for reproducibility
        scaling : bool, default=True
            Whether to apply min-max scaling to features and targets
            (highly recommended for TabPFN performance)
        **kwargs : dict
            Additional parameters passed to TabPFNRegressor
        """
        self.device = device
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.scaling = scaling
        self.kwargs = kwargs

        # Initialize the actual TabPFN model and scalers
        self.model_ = None
        self._is_fitted = False
        self.scaler_X_ = None
        self.scaler_y_ = None

    def fit(
        self,
        y: Union[pd.Series, pd.DataFrame, np.ndarray],
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        static_features: Optional[Union[pd.DataFrame, dict]] = None,
        **fit_params,
    ):
        """
        Fit TabPFN model to training data.

        Parameters:
        -----------
        y : Series, DataFrame, or array
            Target time series data
        X : DataFrame or array, optional
            Exogenous features (ignored by TabPFN)
        static_features : DataFrame or dict, optional
            Static features (ignored by TabPFN)
        **fit_params : dict
            Additional fit parameters (ignored)

        Returns:
        --------
        self : TabPFNWrapper
            Returns self for method chaining
        """
        # TabPFN doesn't support exogenous/static features in time series context
        # We only use the target data (y)

        # Convert to numpy arrays if needed
        if hasattr(y, "values"):
            y_data = y.values
        else:
            y_data = y

        y_data = np.asarray(y_data, dtype=np.float32)

        # For TabPFN, we need to create features from the time series
        # Simple approach: use lagged values as features
        n_lags = min(10, len(y_data) // 4)  # Use up to 10 lags or 1/4 of data length

        if len(y_data) <= n_lags:
            # Too short for lag features, use basic approach
            X_data = np.arange(len(y_data)).reshape(-1, 1).astype(np.float32)
            y_target = y_data
        else:
            # Create lag features
            X_data = []
            y_target = []
            for i in range(n_lags, len(y_data)):
                X_data.append(y_data[i - n_lags : i])
                y_target.append(y_data[i])

            X_data = np.array(X_data, dtype=np.float32)
            y_target = np.array(y_target, dtype=np.float32)

        # Store feature dimension for prediction
        self.n_features_ = X_data.shape[1]

        # Apply scaling if enabled (following Nixtla pipeline approach)
        if self.scaling:
            # Initialize scalers following Nixtla pattern (consistent with TOTO)
            self.scaler_X_ = MinMaxScaler()
            self.scaler_y_ = MinMaxScaler()

            # Fit and transform features
            X_data = self.scaler_X_.fit_transform(X_data)

            # Fit and transform target
            y_target = self.scaler_y_.fit_transform(y_target.reshape(-1, 1)).ravel()

        # Handle TabPFN's sample size limitation
        if len(X_data) > self.max_samples:
            warnings.warn(
                f"Dataset has {len(X_data)} samples, but TabPFN works best with <={self.max_samples} samples. "
                f"Randomly sampling {self.max_samples} samples for training.",
                UserWarning,
            )
            # Random sampling
            if self.random_state is not None:
                np.random.seed(self.random_state)

            indices = np.random.choice(
                len(X_data), size=self.max_samples, replace=False
            )
            X_data = X_data[indices]
            y_target = y_target[indices]

        # Initialize and fit TabPFN model
        try:
            self.model_ = TabPFNRegressor(
                device=self.device,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.kwargs,
            )
            self.model_.fit(X_data, y_target)
            self._is_fitted = True

        except Exception as e:
            # Fallback to CPU if GPU fails
            if self.device != "cpu":
                warnings.warn(
                    f"TabPFN failed on {self.device}, falling back to CPU: {str(e)}",
                    UserWarning,
                )
                self.model_ = TabPFNRegressor(
                    device="cpu",
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    **self.kwargs,
                )
                self.model_.fit(X_data, y_target)
                self._is_fitted = True
            else:
                raise e

        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        static_features: Optional[Union[pd.DataFrame, dict]] = None,
    ) -> np.ndarray:
        """
        Make predictions using the fitted TabPFN model.

        Parameters:
        -----------
        horizon : int
            Number of periods to forecast
        X : DataFrame or array, optional
            Future exogenous variables (ignored by TabPFN)
        static_features : DataFrame or dict, optional
            Static features (ignored by TabPFN)

        Returns:
        --------
        predictions : np.ndarray
            Predicted values for the forecast horizon
        """
        if not self._is_fitted:
            raise ValueError("This TabPFNWrapper instance is not fitted yet.")

        # For TabPFN time series prediction, we need to generate horizon predictions
        # This is a simplified approach - in practice, you might want more sophisticated logic

        # We need to store the feature dimension from training
        if not hasattr(self, "n_features_"):
            raise ValueError("Model must be fitted before making predictions")

        # Generate forecasts by creating features consistent with training
        predictions = []

        for i in range(horizon):
            # Create features matching the training dimension
            if self.n_features_ == 1:
                X_pred = np.array([[i]], dtype=np.float32)
            else:
                # For lag features, create a pattern
                X_pred = np.zeros((1, self.n_features_), dtype=np.float32)
                X_pred[0, -1] = i  # Set the last feature as step indicator

            # Apply scaling if enabled (following Nixtla pipeline, consistent with TOTO)
            if self.scaling and self.scaler_X_ is not None:
                X_pred = self.scaler_X_.transform(X_pred)

            pred_scaled = self.model_.predict(X_pred)[0]

            # Inverse transform prediction if scaling was applied
            if self.scaling and self.scaler_y_ is not None:
                pred = self.scaler_y_.inverse_transform([[pred_scaled]])[0, 0]
            else:
                pred = pred_scaled

            predictions.append(pred)

        return np.array(predictions)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            "device": self.device,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "scaling": self.scaling,
        }
        if deep:
            params.update(self.kwargs)
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key in [
                "device",
                "n_estimators",
                "max_samples",
                "random_state",
                "scaling",
            ]:
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
