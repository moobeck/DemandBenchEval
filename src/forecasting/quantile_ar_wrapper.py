"""
Autoregressive Quantile Regression Models for MLForecast.

This module implements quantile regression models compatible with MLForecast's
single-output requirement. Each quantile gets its own model instance.
"""

import numpy as np
from typing import List, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import QuantileRegressor


class SingleQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Single-output quantile regression model compatible with MLForecast.
    
    This model fits a single quantile regression model for one specific quantile.
    MLForecast will handle the autoregressive feature engineering automatically.
    """
    
    def __init__(
        self,
        quantile: float = 0.5,
        alpha: float = 0.1,
        solver: str = "highs",
        random_state: Optional[int] = None
    ):
        """
        Initialize the single quantile regressor.
        
        Parameters:
        -----------
        quantile : float
            Quantile to predict (between 0 and 1)
        alpha : float
            Regularization strength for quantile regression (default: 0.1)
        solver : str
            Solver to use for quantile regression (default: "highs")
        random_state : int, optional
            Random state for reproducibility
        """
        self.quantile = quantile
        self.alpha = alpha
        self.solver = solver
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the quantile regression model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features (MLForecast provides lagged features automatically)
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : SingleQuantileRegressor
            Returns self for method chaining
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Fit quantile regression model
        self.model_ = QuantileRegressor(
            quantile=self.quantile,
            alpha=self.alpha,
            solver=self.solver
        )
        self.model_.fit(X, y)
        
        return self
        
    def predict(self, X):
        """
        Predict quantile for given features.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features for prediction
            
        Returns:
        --------
        np.ndarray of shape (n_samples,)
            Single quantile predictions for each sample
        """
        X = np.asarray(X)
        return self.model_.predict(X)
        
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters:
        -----------
        deep : bool, default=True
            If True, will return parameters for sub-estimators
            
        Returns:
        --------
        dict
            Parameter names mapped to their values
        """
        return {
            'quantile': self.quantile,
            'alpha': self.alpha,
            'solver': self.solver,
            'random_state': self.random_state
        }
        
    def set_params(self, **params):
        """
        Set parameters for this estimator.
        
        Parameters:
        -----------
        **params : dict
            Estimator parameters
            
        Returns:
        --------
        self : SingleQuantileRegressor
            Returns self for method chaining
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self


class AutoQuantileAR:
    """
    AutoML-style wrapper for SingleQuantileRegressor to work with AutoMLForecast.
    
    This mimics the interface of AutoLightGBM, AutoCatboost, etc.
    """
    
    def __init__(
        self,
        quantile: float = 0.5,
        alpha: float = 0.1,
        solver: str = "highs",
        random_state: Optional[int] = None,
        **kwargs
    ):
        """Initialize the AutoQuantileAR wrapper."""
        self.quantile = quantile
        self.alpha = alpha
        self.solver = solver
        self.random_state = random_state
        
        # Create the underlying model (this is what AutoMLForecast expects)
        self.model = SingleQuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            solver=solver,
            random_state=random_state
        )
        
        # Add alias for identification
        self.model.alias = f"QuantileAR_q{quantile}"
        
        # Config function (AutoMLForecast expects this to be callable)
        def config_fn(trial=None):
            """Configuration function for hyperparameter optimization."""
            return {
                'quantile': quantile,
                'alpha': alpha,
                'solver': solver,
                'random_state': random_state
            }
        
        self.config = config_fn


def create_quantile_models(
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    alpha: float = 0.1,
    solver: str = "highs",
    random_state: Optional[int] = None
) -> List[SingleQuantileRegressor]:
    """
    Create multiple single-quantile regression models.
    
    Parameters:
    -----------
    quantiles : List[float]
        List of quantiles to create models for
    alpha : float
        Regularization strength for all models
    solver : str
        Solver to use for all models
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    List[SingleQuantileRegressor]
        List of single-quantile regression models
    """
    models = []
    for q in quantiles:
        model = SingleQuantileRegressor(
            quantile=q,
            alpha=alpha,
            solver=solver,
            random_state=random_state
        )
        # Set alias for identification
        model.alias = f"QuantileAR_q{q}"
        models.append(model)
    
    return models


# For backward compatibility - this is the main class that can create multiple models
class QuantileARRegressor(BaseEstimator, RegressorMixin):
    """
    Factory class that creates multiple single-quantile models for MLForecast compatibility.
    
    This class maintains the interface while providing single-output models that work
    with MLForecast's recursive forecasting approach.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        alpha: float = 0.1,
        solver: str = "highs",
        random_state: Optional[int] = None
    ):
        """
        Initialize the quantile AR model factory.
        
        Parameters:
        -----------
        quantiles : List[float]
            List of quantiles to predict
        alpha : float
            Regularization strength
        solver : str
            Solver to use
        random_state : int, optional
            Random state for reproducibility
        """
        self.quantiles = quantiles
        self.alpha = alpha
        self.solver = solver
        self.random_state = random_state
        
    def create_models(self) -> List[SingleQuantileRegressor]:
        """
        Create individual quantile models for MLForecast.
        
        Returns:
        --------
        List[SingleQuantileRegressor]
            List of single-quantile models
        """
        return create_quantile_models(
            quantiles=self.quantiles,
            alpha=self.alpha,
            solver=self.solver,
            random_state=self.random_state
        )
        
    def fit(self, X, y):
        """Placeholder fit method - use create_models() instead."""
        raise NotImplementedError("Use create_models() to get individual models for MLForecast")
        
    def predict(self, X):
        """Placeholder predict method - use create_models() instead."""
        raise NotImplementedError("Use create_models() to get individual models for MLForecast")
        
    def get_params(self, deep=True):
        """Get parameters for this factory."""
        return {
            'quantiles': self.quantiles,
            'alpha': self.alpha,
            'solver': self.solver,
            'random_state': self.random_state
        }