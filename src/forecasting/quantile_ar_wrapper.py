"""
Autoregressive Quantile Regression Models for MLForecast.

This module implements a generic quantile regression wrapper that can work with
any sklearn-compatible model. The wrapper handles quantile-specific configuration
and provides a consistent interface for MLForecast.
"""

import numpy as np
from typing import List, Optional, Union, Any, Dict, Callable
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import QuantileRegressor

# Optional imports for different model types
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

try:
    from sklearn_quantile import RandomForestQuantileRegressor
    RANDOM_FOREST_QUANTILE_AVAILABLE = True
except ImportError:
    RANDOM_FOREST_QUANTILE_AVAILABLE = False
    RandomForestQuantileRegressor = None


class QuantileARWrapper(BaseEstimator, RegressorMixin):
    """
    Generic quantile regression wrapper that can work with any sklearn-compatible model.
    
    This wrapper takes a base model and configures it for quantile regression,
    handling the quantile-specific parameters automatically. It provides a consistent
    interface regardless of the underlying model type.
    
    Examples:
    ---------
    # Linear quantile regression
    from sklearn.linear_model import QuantileRegressor
    model = QuantileARWrapper(
        base_model=QuantileRegressor(),
        quantile=0.5
    )
    
    # LightGBM quantile regression  
    import lightgbm as lgb
    model = QuantileARWrapper(
        base_model=lgb.LGBMRegressor(),
        quantile=0.25,
        quantile_config_fn=lambda model, q: model.set_params(objective='quantile', alpha=q)
    )
    """
    
    def __init__(
        self,
        base_model: BaseEstimator,
        quantile: float = 0.5,
        quantile_config_fn: Optional[Callable[[BaseEstimator, float], None]] = None,
        predict_config_fn: Optional[Callable[[BaseEstimator, np.ndarray, float], np.ndarray]] = None,
        alias: Optional[str] = None
    ):
        """
        Initialize the quantile regression wrapper.
        
        Parameters:
        -----------
        base_model : BaseEstimator
            The base sklearn-compatible model to wrap for quantile regression
        quantile : float
            Quantile to predict (between 0 and 1)
        quantile_config_fn : Callable, optional
            Function to configure the base model for quantile regression.
            Signature: fn(model, quantile) -> None
        predict_config_fn : Callable, optional  
            Function to handle quantile-specific prediction logic.
            Signature: fn(model, X, quantile) -> predictions
            If None, uses standard model.predict(X)
        alias : str, optional
            Custom alias for model identification. If None, auto-generated.
        """
        self.base_model = base_model
        self.quantile = quantile
        self.quantile_config_fn = quantile_config_fn
        self.predict_config_fn = predict_config_fn
        
        # Set alias for identification
        if alias is None:
            model_name = type(base_model).__name__.replace('Regressor', '')
            self.alias = f"Quantile{model_name}_q{quantile}"
        else:
            self.alias = alias
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
        self : QuantileARWrapper
            Returns self for method chaining
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Check that base_model exists
        if self.base_model is None:
            raise ValueError("base_model is None. Ensure AutoQuantileAR properly initializes the base model.")
        
        # Clone the base model to avoid modifying the original
        self.model_ = clone(self.base_model)
        
        # Configure the model for quantile regression if config function provided
        if self.quantile_config_fn is not None:
            self.quantile_config_fn(self.model_, self.quantile)
        
        # Fit the model
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
        
        # Use custom prediction function if provided
        if self.predict_config_fn is not None:
            return self.predict_config_fn(self.model_, X, self.quantile)
        else:
            # Standard prediction
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
        params = {
            'base_model': self.base_model,
            'quantile': self.quantile,
            'quantile_config_fn': self.quantile_config_fn,
            'predict_config_fn': self.predict_config_fn,
            'alias': self.alias
        }
        
        if deep and hasattr(self.base_model, 'get_params'):
            # Include base model parameters with prefix
            base_params = self.base_model.get_params(deep=deep)
            params.update({f'base_model__{k}': v for k, v in base_params.items()})
        
        return params
        
    def set_params(self, **params):
        """
        Set parameters for this estimator.
        
        Parameters:
        -----------
        **params : dict
            Estimator parameters
            
        Returns:
        --------
        self : QuantileARWrapper
            Returns self for method chaining
        """
        # Separate base model parameters from wrapper parameters
        base_model_params = {}
        wrapper_params = {}
        
        for param, value in params.items():
            if param.startswith('base_model__'):
                base_model_params[param[12:]] = value  # Remove 'base_model__' prefix
            else:
                wrapper_params[param] = value
        
        # Set wrapper parameters
        for param, value in wrapper_params.items():
            setattr(self, param, value)
        
        # Set base model parameters
        if base_model_params and hasattr(self.base_model, 'set_params'):
            self.base_model.set_params(**base_model_params)
        
        return self


# Helper functions to create specific quantile model configurations
def create_linear_quantile_model(quantile: float = 0.5, **kwargs) -> QuantileARWrapper:
    """Create a linear quantile regression model."""
    # Filter out parameters not supported by QuantileRegressor
    linear_kwargs = {k: v for k, v in kwargs.items() if k in ['alpha', 'solver', 'fit_intercept']}
    base_model = QuantileRegressor(**linear_kwargs)
    
    def config_fn(model, q):
        model.set_params(quantile=q)
    
    return QuantileARWrapper(
        base_model=base_model,
        quantile=quantile,
        quantile_config_fn=config_fn
    )


def create_lightgbm_quantile_model(quantile: float = 0.5, **kwargs) -> QuantileARWrapper:
    """Create a LightGBM quantile regression model."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available. Install with: pip install lightgbm")
    
    # Set default parameters for quantile regression
    lgb_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'verbosity': -1,
        **kwargs
    }
    base_model = lgb.LGBMRegressor(**lgb_params)
    
    def config_fn(model, q):
        model.set_params(alpha=q)  # LightGBM uses 'alpha' for quantile
    
    return QuantileARWrapper(
        base_model=base_model,
        quantile=quantile,
        quantile_config_fn=config_fn
    )


def create_catboost_quantile_model(quantile: float = 0.5, **kwargs) -> QuantileARWrapper:
    """Create a CatBoost quantile regression model."""
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost not available. Install with: pip install catboost")
    
    # Set default parameters for quantile regression
    # Filter out parameters that are not valid for CatBoostRegressor
    invalid_params = {'lags', 'quantile', 'alias'}
    cb_params = {
        'verbose': False,
        **{k: v for k, v in kwargs.items() if k not in invalid_params}
    }
    base_model = cb.CatBoostRegressor(**cb_params)
    
    def config_fn(model, q):
        model.set_params(loss_function=f'Quantile:alpha={q}')
    
    return QuantileARWrapper(
        base_model=base_model,
        quantile=quantile,
        quantile_config_fn=config_fn
    )


def create_random_forest_quantile_model(quantile: float = 0.5, **kwargs) -> QuantileARWrapper:
    """Create a Random Forest quantile regression model."""
    if not RANDOM_FOREST_QUANTILE_AVAILABLE:
        raise ImportError("sklearn-quantile not available. Install with: pip install sklearn-quantile")
    
    # Filter out parameters that are not valid for RandomForestQuantileRegressor
    invalid_params = {'lags', 'alias'}  # Keep 'quantile' out of invalid since RF needs it as 'q'
    rf_params = {
        'q': quantile,  # RandomForestQuantileRegressor uses 'q' parameter
        **{k: v for k, v in kwargs.items() if k not in invalid_params and k != 'quantile'}
    }
    base_model = RandomForestQuantileRegressor(**rf_params)
    
    def predict_fn(model, X, q):
        # RandomForestQuantileRegressor is configured with quantile at init
        # It only has a standard predict method
        return model.predict(X)
    
    return QuantileARWrapper(
        base_model=base_model,
        quantile=quantile,
        predict_config_fn=predict_fn
    )


class SimpleQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Simple quantile regressor that works directly with AutoMLForecast.
    
    This is a simplified version that avoids the abstraction complexity
    and works directly with AutoMLForecast's parameter-based approach.
    """
    
    def __init__(self, quantile: float = 0.5, model_type: str = 'linear', **kwargs):
        self.quantile = quantile
        self.model_type = model_type
        self.model_kwargs = kwargs
        self.alias = f"Quantile{model_type.title()}_q{quantile}"
    
    def _create_base_model(self):
        """Create the base model based on type."""
        if self.model_type == 'linear':
            # Filter params for QuantileRegressor
            valid_params = {k: v for k, v in self.model_kwargs.items() 
                          if k in ['alpha', 'solver', 'fit_intercept']}
            return QuantileRegressor(quantile=self.quantile, **valid_params)
        
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            return lgb.LGBMRegressor(
                objective='quantile',
                alpha=self.quantile,
                metric='quantile',
                verbosity=-1,
                **self.model_kwargs
            )
        
        elif self.model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            return cb.CatBoostRegressor(
                loss_function=f'Quantile:alpha={self.quantile}',
                verbose=False,
                **self.model_kwargs
            )
        
        elif self.model_type == 'random_forest':
            if not RANDOM_FOREST_QUANTILE_AVAILABLE:
                raise ImportError("sklearn-quantile not available")
            return RandomForestQuantileRegressor(
                q=self.quantile,
                **self.model_kwargs
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model_ = self._create_base_model()
        self.model_.fit(X, y)
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        return self.model_.predict(X)
    
    def get_params(self, deep=True):
        params = {
            'quantile': self.quantile,
            'model_type': self.model_type,
        }
        params.update(self.model_kwargs)
        return params
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['quantile', 'model_type']:
                setattr(self, key, value)
            else:
                self.model_kwargs[key] = value
        
        # Update alias when parameters change
        self.alias = f"Quantile{self.model_type.title()}_q{self.quantile}"
        return self


class AutoQuantileAR:
    """
    AutoML-style wrapper for simple quantile regression.
    """
    
    def __init__(self, quantile: float = 0.5, model_type: str = 'linear', **kwargs):
        self.quantile = quantile
        self.model_type = model_type
        self.kwargs = kwargs
        
        # Create the simple model
        self.model = SimpleQuantileRegressor(
            quantile=quantile,
            model_type=model_type,
            **kwargs
        )
        
        # Config function returns parameters
        def config_fn(trial=None):
            return {
                'quantile': quantile,
                'model_type': model_type,
                **kwargs
            }
        
        self.config = config_fn


def create_quantile_models(
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    base_model: Optional[BaseEstimator] = None,
    model_type: Optional[str] = None,
    **model_kwargs
) -> List[QuantileARWrapper]:
    """
    Create multiple quantile regression models using the abstraction.
    
    Parameters:
    -----------
    quantiles : List[float]
        List of quantiles to create models for
    base_model : BaseEstimator, optional
        Base model to wrap for each quantile. If provided, model_type is ignored.
    model_type : str, optional
        Type of model to create if base_model is None.
        Options: 'linear', 'lightgbm', 'catboost', 'random_forest'
    **model_kwargs : dict
        Additional model-specific parameters
        
    Returns:
    --------
    List[QuantileARWrapper]
        List of quantile regression models
    """
    models = []
    
    for q in quantiles:
        if base_model is not None:
            # Use provided base model (clone for each quantile)
            model = QuantileARWrapper(base_model=clone(base_model), quantile=q, **model_kwargs)
        elif model_type is not None:
            # Create model based on type
            auto_model = AutoQuantileAR(quantile=q, model_type=model_type, **model_kwargs)
            model = auto_model.model
        else:
            # Default to linear
            model = create_linear_quantile_model(quantile=q, **model_kwargs)
        
        models.append(model)
    
    return models

