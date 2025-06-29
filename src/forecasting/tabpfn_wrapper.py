from typing import Dict, Any
import logging
import torch
import numpy as np
import pandas as pd

from tabpfn import TabPFNRegressor
from src.forecasting.foundation_model_base import FoundationModelWrapper
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.enums import Frequency
import warnings


class TabPFNWrapper(FoundationModelWrapper):
    """
    Wrapper for TabPFN Regressor following TabPFN's design principles:
    - Simple tabular data approach
    - Minimal feature engineering (TabPFN does internal preprocessing)
    - Cross-sectional prediction style
    """

    def __init__(
        self,
        n_estimators: int = 8,
        max_samples: int = 10000,
        random_state: int = 42,
        n_lags: int = 12,
        alias="TabPFN",
        **kwargs,
    ):
        """
        Initialize TabPFN wrapper for time series forecasting.

        Args:
            n_estimators: Number of estimators in TabPFN ensemble
            max_samples: Maximum samples for TabPFN
            random_state: Random state for reproducibility
            n_lags: Number of lag features (keep simple)
            alias: Model alias for output column naming
            **kwargs: Additional TabPFN parameters
        """
        self.alias = alias
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_lags = n_lags
        self.kwargs = kwargs
        
        # Device handling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing TabPFN on device: {self.device}")
        
        # Single model for all series (TabPFN foundation model approach)
        self.model = None
        self._is_fitted = False
        self.model_type = f"TabPFN (n_estimators={n_estimators}, device={self.device})"
        
        # Store basic info for prediction
        self.feature_columns = []
        self.n_features = 0

    def predict(
        self,
        X: pd.DataFrame,
        forecast_columns: ForecastColumnConfig,
        horizon: int,
        freq: Frequency,
    ) -> pd.DataFrame:
        """
        Predict using TabPFN with simple tabular approach
        
        Args:
            X: Input DataFrame with time series data
            forecast_columns: Configuration object with column names
            horizon: Number of periods to forecast
            freq: Time frequency of the data
            
        Returns:
            DataFrame: Forecasts in Nixtla format
        """
        if not self._is_fitted:
            # Fit model if not already fitted
            self._fit_model(X, forecast_columns, freq)

        # Generate predictions for each series
        predictions = []
        unique_ids = X[forecast_columns.sku_index].unique().tolist()

        for series_id in unique_ids:
            # Get series data
            series_df = X[X[forecast_columns.sku_index] == series_id].sort_values(
                forecast_columns.date
            )
            
            # Generate forecast for this series
            series_forecast = self._forecast_series(
                series_df, horizon, forecast_columns, freq
            )
            predictions.append(series_forecast)

        # Stack all predictions: (n_series, horizon)
        predictions = np.array(predictions)

        # Convert to Nixtla format
        df = self._to_nixtla_df(
            predictions=predictions,
            unique_ids=unique_ids,
            start_date=X[forecast_columns.date].max(),
            forecast_columns=forecast_columns,
            frequency=freq,
        )

        return df

    def _fit_model(self, X: pd.DataFrame, forecast_columns: ForecastColumnConfig, freq: Frequency):
        """
        Fit single TabPFN model using simple tabular approach
        
        Args:
            X: Input DataFrame with time series data
            forecast_columns: Configuration object with column names
            freq: Time frequency of the data
        """
        logging.info("Creating training data for TabPFN...")
        
        # Create simple training dataset from all series
        X_train, y_train = self._create_training_data(X, forecast_columns, freq)
        
        if len(X_train) == 0:
            raise ValueError("No valid training data created")
            
        # Store feature info
        self.n_features = X_train.shape[1]
        self.feature_columns = [f"feature_{i}" for i in range(self.n_features)]
        
        logging.info(f"Training TabPFN with {len(X_train)} samples and {self.n_features} features")
        
        # Handle sample size limitation
        if len(X_train) > self.max_samples:
            logging.info(f"Sampling {self.max_samples} from {len(X_train)} samples for TabPFN")
            np.random.seed(self.random_state)
            indices = np.random.choice(len(X_train), size=self.max_samples, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        # Create and fit TabPFN model
        self.model = self._create_tabpfn_model()
        
        # Clean data for TabPFN
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        logging.info("TabPFN training completed successfully")

    def _create_tabpfn_model(self) -> TabPFNRegressor:
        """Create a TabPFN model with proper configuration"""
        # Handle CPU limitations
        kwargs = self.kwargs.copy()
        if self.device == "cpu":
            # TabPFN has strict limits on CPU, so we need to override them
            kwargs['ignore_pretraining_limits'] = True
            logging.info("Running TabPFN on CPU with ignore_pretraining_limits=True")
        
        try:
            return TabPFNRegressor(
                device=self.device,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **kwargs,
            )
        except Exception as e:
            if self.device != "cpu":
                logging.warning(f"TabPFN failed on {self.device}, falling back to CPU: {str(e)}")
                # Set CPU-specific parameters
                kwargs['ignore_pretraining_limits'] = True
                return TabPFNRegressor(
                    device="cpu",
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    **kwargs,
                )
            else:
                raise e

    def _create_training_data(self, X: pd.DataFrame, forecast_columns: ForecastColumnConfig, freq: Frequency) -> tuple:
        """
        Create simple training dataset for TabPFN
        
        Returns:
            tuple: (X_features, y_target) as numpy arrays
        """
        X_features = []
        y_target = []
        
        unique_ids = X[forecast_columns.sku_index].unique()
        
        for series_id in unique_ids:
            series_df = X[X[forecast_columns.sku_index] == series_id].sort_values(
                forecast_columns.date
            )
            
            if len(series_df) <= self.n_lags:
                continue  # Skip series with insufficient data
                
            target_values = series_df[forecast_columns.target].values
            
            # Create simple lag-based features for this series
            for i in range(self.n_lags, len(series_df)):
                # Simple feature vector: just lag values + basic info
                features = []
                
                # Lag features (core information)
                features.extend(target_values[i - self.n_lags : i])
                
                # Add simple static info if available
                if hasattr(forecast_columns, 'static') and forecast_columns.static:
                    for static_col in forecast_columns.static:
                        if static_col in series_df.columns:
                            val = series_df[static_col].iloc[0]  # Static value
                            if isinstance(val, (int, float)) and np.isfinite(val):
                                features.append(float(val))
                            else:
                                # Simple encoding for categorical
                                features.append(float(hash(str(val)) % 100))
                
                # Add basic time features (let TabPFN handle the complexity)
                current_date = pd.to_datetime(series_df.iloc[i][forecast_columns.date])
                features.extend([
                    float(current_date.month),
                    float(current_date.dayofweek),
                ])
                
                # Add basic exogenous if available  
                if hasattr(forecast_columns, 'base_exogenous') and forecast_columns.base_exogenous:
                    for exog_col in forecast_columns.base_exogenous[:2]:  # Limit to 2 most important
                        if exog_col in series_df.columns:
                            val = series_df.iloc[i][exog_col]
                            if isinstance(val, (int, float)) and np.isfinite(val):
                                features.append(float(val))
                            else:
                                features.append(0.0)
                
                X_features.append(features)
                y_target.append(target_values[i])
        
        if len(X_features) == 0:
            return np.array([]), np.array([])
            
        # Convert to numpy and ensure consistent feature dimensions
        X_features = np.array(X_features, dtype=np.float32)
        y_target = np.array(y_target, dtype=np.float32)
        
        # Ensure all samples have same number of features
        if X_features.ndim == 2 and X_features.shape[0] > 0:
            max_features = X_features.shape[1]
            # All rows should already have same length, but ensure it
            X_features = X_features[:, :max_features]
        
        return X_features, y_target

    def _forecast_series(self, series_df: pd.DataFrame, horizon: int, 
                        forecast_columns: ForecastColumnConfig, freq: Frequency) -> np.ndarray:
        """
        Generate forecast for a single time series using simple approach
        
        Args:
            series_df: DataFrame for the series
            horizon: Number of periods to forecast
            forecast_columns: Configuration object with column names
            freq: Time frequency of the data
            
        Returns:
            np.ndarray: Forecast values
        """
        if self.model is None:
            return np.zeros(horizon)

        target_values = series_df[forecast_columns.target].values
        
        # Use simple approach: create features similar to training
        if len(target_values) < self.n_lags:
            # Not enough history - return zeros or simple extrapolation
            return np.full(horizon, target_values[-1] if len(target_values) > 0 else 0.0)
        
        predictions = []
        
        # Start with last n_lags values
        recent_values = target_values[-self.n_lags:].copy()
        
        # Get static and other features (same approach as training)
        static_features = []
        if hasattr(forecast_columns, 'static') and forecast_columns.static:
            for static_col in forecast_columns.static:
                if static_col in series_df.columns:
                    val = series_df[static_col].iloc[0]
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        static_features.append(float(val))
                    else:
                        static_features.append(float(hash(str(val)) % 100))
        
        last_date = pd.to_datetime(series_df[forecast_columns.date].iloc[-1])
        
        # Generate predictions step by step
        for step in range(horizon):
            # Create feature vector
            features = list(recent_values)  # Lag features
            features.extend(static_features)  # Static features
            
            # Add time features for forecast period
            forecast_date = self._add_time_delta(last_date, step + 1, freq)
            features.extend([
                float(forecast_date.month),
                float(forecast_date.dayofweek),
            ])
            
            # Add basic exogenous (use last known values)
            if hasattr(forecast_columns, 'base_exogenous') and forecast_columns.base_exogenous:
                for exog_col in forecast_columns.base_exogenous[:2]:
                    if exog_col in series_df.columns:
                        val = series_df[exog_col].iloc[-1]  # Use last known value
                        if isinstance(val, (int, float)) and np.isfinite(val):
                            features.append(float(val))
                        else:
                            features.append(0.0)
            
            # Ensure correct feature dimension
            if len(features) != self.n_features:
                # Pad or truncate to match training
                if len(features) < self.n_features:
                    features.extend([0.0] * (self.n_features - len(features)))
                else:
                    features = features[:self.n_features]
            
            # Create prediction input
            X_pred = np.array(features, dtype=np.float32).reshape(1, -1)
            X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
            
            try:
                pred = self.model.predict(X_pred)[0]
                # Basic sanity check
                if not np.isfinite(pred):
                    pred = recent_values[-1]  # Use last known value
                predictions.append(pred)
                
                # Update recent values for next prediction
                recent_values = np.roll(recent_values, -1)
                recent_values[-1] = pred
                
            except Exception as e:
                logging.warning(f"Prediction failed at step {step}: {str(e)}")
                # Use last known value
                raise e

        return np.array(predictions)

    def _add_time_delta(self, date: pd.Timestamp, steps: int, freq: Frequency) -> pd.Timestamp:
        """Add time delta based on frequency"""
        if freq == Frequency.DAILY:
            return date + pd.Timedelta(days=steps)
        elif freq == Frequency.WEEKLY:
            return date + pd.Timedelta(weeks=steps)
        elif freq == Frequency.MONTHLY:
            return date + pd.DateOffset(months=steps)
        elif freq == Frequency.QUARTERLY:
            return date + pd.DateOffset(months=3 * steps)
        elif freq == Frequency.YEARLY:
            return date + pd.DateOffset(years=steps)
        else:
            # Default to daily
            return date + pd.Timedelta(days=steps)

    def _to_nixtla_df(
        self,
        predictions: np.ndarray,
        unique_ids: list[str],
        start_date: str,
        forecast_columns: ForecastColumnConfig,
        frequency: Frequency,
    ) -> pd.DataFrame:
        """Convert predictions to Nixtla format DataFrame"""
        
        # Create output DataFrame
        output_data = []
        
        for series_idx, series_id in enumerate(unique_ids):
            series_predictions = predictions[series_idx]
            
            # Generate forecast dates
            start_date_pd = pd.to_datetime(start_date)
            
            for step in range(len(series_predictions)):
                forecast_date = self._add_time_delta(start_date_pd, step + 1, frequency)
                
                output_data.append({
                    forecast_columns.sku_index: series_id,
                    forecast_columns.date: forecast_date,
                    self.alias: series_predictions[step]
                })
        
        return pd.DataFrame(output_data)


