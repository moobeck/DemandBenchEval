from typing import Dict, Any
import logging
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tabpfn import TabPFNRegressor
from src.forecasting.foundation_model_base import FoundationModelWrapper
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.enums import Frequency
import warnings


class TabPFNWrapper(FoundationModelWrapper):
    """
    Wrapper for TabPFN Regressor using proper time series forecasting implementation
    """

    def __init__(
        self,
        n_estimators: int = 8,
        max_samples: int = 10000,
        random_state: int = 42,
        n_lags: int = 10,
        scaling: bool = True,
        alias="TabPFN",
        **kwargs,
    ):
        """
        Initialize TabPFN wrapper for time series forecasting.

        Args:
            n_estimators: Number of estimators in TabPFN ensemble
            max_samples: Maximum samples for TabPFN (due to model limitations)
            random_state: Random state for reproducibility
            n_lags: Number of lag features to create for time series modeling
            scaling: Whether to apply min-max scaling
            alias: Model alias for output column naming
            **kwargs: Additional TabPFN parameters
        """
        self.alias = alias
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_lags = n_lags
        self.scaling = scaling
        self.kwargs = kwargs
        
        # Device handling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing TabPFN on device: {self.device}")
        
        # Model components
        self.models = {}  # One model per time series
        self.scalers_X = {}
        self.scalers_y = {}
        self._is_fitted = False
        self.model_type = f"TabPFN (n_estimators={n_estimators}, device={self.device})"

    def predict(
        self,
        X: pd.DataFrame,
        forecast_columns: ForecastColumnConfig,
        horizon: int,
        freq: Frequency,
    ) -> pd.DataFrame:
        """
        Predict using TabPFN with proper time series forecasting
        
        Args:
            X: Input DataFrame with time series data
            forecast_columns: Configuration object with column names
            horizon: Number of periods to forecast
            freq: Time frequency of the data
            
        Returns:
            DataFrame: Forecasts in Nixtla format
        """
        if not self._is_fitted:
            # Fit models if not already fitted
            self._fit_models(X, forecast_columns)

        # Generate predictions for each series
        predictions = []
        unique_ids = X[forecast_columns.sku_index].unique().tolist()

        for series_id in unique_ids:
            # Get series data and DataFrame
            series_df = X[X[forecast_columns.sku_index] == series_id].sort_values(
                forecast_columns.date
            )
            series_data = series_df[forecast_columns.target].values

            # Generate forecast for this series
            series_forecast = self._forecast_series(
                series_id, series_data, horizon, series_df, forecast_columns
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

    def _fit_models(self, X: pd.DataFrame, forecast_columns: ForecastColumnConfig):
        """
        Fit TabPFN models for each time series
        
        Args:
            X: Input DataFrame with time series data
            forecast_columns: Configuration object with column names
        """
        unique_ids = X[forecast_columns.sku_index].unique()
        
        for series_id in unique_ids:
            try:
                # Get series data sorted by date
                series_df = X[X[forecast_columns.sku_index] == series_id].sort_values(
                    forecast_columns.date
                )

                # Create comprehensive features including lag features, static features, and exogenous variables
                X_features, y_target = self._create_comprehensive_features(series_df, forecast_columns)
                
                if len(X_features) == 0:
                    logging.warning(f"Insufficient data for series {series_id}, skipping")
                    continue

                # Apply scaling if enabled
                if self.scaling:
                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()
                    
                    X_features = scaler_X.fit_transform(X_features)
                    y_target = scaler_y.fit_transform(y_target.reshape(-1, 1)).ravel()
                    
                    self.scalers_X[series_id] = scaler_X
                    self.scalers_y[series_id] = scaler_y

                # Handle sample size limitation
                if len(X_features) > self.max_samples:
                    logging.info(
                        f"Series {series_id} has {len(X_features)} samples, "
                        f"sampling {self.max_samples} for TabPFN"
                    )
                    np.random.seed(self.random_state)
                    indices = np.random.choice(
                        len(X_features), size=self.max_samples, replace=False
                    )
                    X_features = X_features[indices]
                    y_target = y_target[indices]

                # Fit TabPFN model
                model = self._create_tabpfn_model()
                model.fit(X_features, y_target)
                self.models[series_id] = model

            except Exception as e:
                logging.warning(f"Failed to fit model for series {series_id}: {str(e)}")
                continue

        self._is_fitted = True
        logging.info(f"Successfully fitted TabPFN models for {len(self.models)} series")

    def _create_tabpfn_model(self) -> TabPFNRegressor:
        """Create a TabPFN model with error handling"""
        try:
            return TabPFNRegressor(
                device=self.device,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                **self.kwargs,
            )
        except Exception as e:
            if self.device != "cpu":
                logging.warning(f"TabPFN failed on {self.device}, falling back to CPU: {str(e)}")
                return TabPFNRegressor(
                    device="cpu",
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    **self.kwargs,
                )
            else:
                raise e

    def _create_lag_features(self, series_data: np.ndarray) -> tuple:
        """
        Create lag features from time series data
        
        Args:
            series_data: Time series values
            
        Returns:
            tuple: (X_features, y_target) for supervised learning
        """
        if len(series_data) <= self.n_lags:
            # Not enough data for lag features
            return np.array([]), np.array([])

        X_features = []
        y_target = []

        for i in range(self.n_lags, len(series_data)):
            # Use previous n_lags values as features
            features = series_data[i - self.n_lags : i]
            target = series_data[i]
            
            X_features.append(features)
            y_target.append(target)

        return np.array(X_features, dtype=np.float32), np.array(y_target, dtype=np.float32)

    def _create_comprehensive_features(self, series_df: pd.DataFrame, forecast_columns: ForecastColumnConfig) -> tuple:
        """
        Create comprehensive features including lag features, static features, and exogenous variables
        
        Args:
            series_df: DataFrame with time series data for one series
            forecast_columns: Configuration object with column names
            
        Returns:
            tuple: (X_features, y_target) for supervised learning
        """
        if len(series_df) <= self.n_lags:
            return np.array([]), np.array([])

        # Get target values
        target_values = series_df[forecast_columns.target].values
        
        X_features = []
        y_target = []

        for i in range(self.n_lags, len(series_df)):
            # Lag features from target
            lag_features = target_values[i - self.n_lags : i]
            
            # Current row features (static + exogenous + date features)
            current_row = series_df.iloc[i]
            additional_features = []
            
            # Add static features (if available)
            if hasattr(forecast_columns, 'static') and forecast_columns.static:
                for static_col in forecast_columns.static:
                    if static_col in current_row:
                        additional_features.append(current_row[static_col])
            
            # Add exogenous features (if available)
            if hasattr(forecast_columns, 'base_exogenous') and forecast_columns.base_exogenous:
                for exog_col in forecast_columns.base_exogenous:
                    if exog_col in current_row:
                        additional_features.append(current_row[exog_col])
            
            # Add date features (month, day of week, etc.)
            if forecast_columns.date in current_row:
                date_val = pd.to_datetime(current_row[forecast_columns.date])
                additional_features.extend([
                    date_val.month,
                    date_val.day,
                    date_val.dayofweek,
                    date_val.dayofyear,
                ])
            
            # Combine all features - ensure all features are numeric and finite
            if additional_features:
                # Convert to numeric and handle any string/categorical values
                numeric_features = []
                for feat in additional_features:
                    if isinstance(feat, (int, float)) and np.isfinite(feat):
                        numeric_features.append(float(feat))
                    else:
                        # Handle categorical or missing values with hash
                        numeric_features.append(float(hash(str(feat)) % 1000))
                
                all_features = np.concatenate([lag_features, numeric_features])
            else:
                all_features = lag_features
                
            X_features.append(all_features)
            y_target.append(target_values[i])

        return np.array(X_features, dtype=np.float32), np.array(y_target, dtype=np.float32)

    def _forecast_series(self, series_id: str, series_data: np.ndarray, horizon: int, series_df: pd.DataFrame = None, forecast_columns: ForecastColumnConfig = None) -> np.ndarray:
        """
        Generate forecast for a single time series
        
        Args:
            series_id: Identifier for the time series
            series_data: Historical data for the series
            horizon: Number of periods to forecast
            series_df: Full DataFrame for the series (for additional features)
            forecast_columns: Configuration object with column names
            
        Returns:
            np.ndarray: Forecast values
        """
        if series_id not in self.models:
            # Return zeros if model not available
            logging.warning(f"No model found for series {series_id}, returning zeros")
            return np.zeros(horizon)

        model = self.models[series_id]
        predictions = []
        
        # Use last n_lags values as initial context
        if len(series_data) >= self.n_lags:
            target_context = series_data[-self.n_lags:].copy()
        else:
            # Pad with zeros if insufficient history
            target_context = np.zeros(self.n_lags)
            if len(series_data) > 0:
                target_context[-len(series_data):] = series_data

        # Get the last row for static features
        if series_df is not None and forecast_columns is not None:
            last_row = series_df.iloc[-1]
            
            # Extract static features
            static_features = []
            if hasattr(forecast_columns, 'static') and forecast_columns.static:
                for static_col in forecast_columns.static:
                    if static_col in last_row:
                        static_features.append(last_row[static_col])
            
            # Extract last known exogenous features (assuming they don't change much)
            exog_features = []
            if hasattr(forecast_columns, 'base_exogenous') and forecast_columns.base_exogenous:
                for exog_col in forecast_columns.base_exogenous:
                    if exog_col in last_row:
                        exog_features.append(last_row[exog_col])
        else:
            static_features = []
            exog_features = []

        # Generate multi-step forecasts
        for step in range(horizon):
            # Create features for this prediction step
            features = list(target_context)  # Lag features
            
            # Add static features (handle categorical/string values)
            for feat in static_features:
                if isinstance(feat, (int, float)) and np.isfinite(feat):
                    features.append(float(feat))
                else:
                    features.append(float(hash(str(feat)) % 1000))
            
            # Add exogenous features (handle categorical/string values)
            for feat in exog_features:
                if isinstance(feat, (int, float)) and np.isfinite(feat):
                    features.append(float(feat))
                else:
                    features.append(float(hash(str(feat)) % 1000))
            
            # Add date features for the forecast period
            if series_df is not None and forecast_columns is not None:
                last_date = pd.to_datetime(series_df[forecast_columns.date].iloc[-1])
                forecast_date = last_date + pd.Timedelta(days=step+1)  # Assuming daily frequency
                features.extend([
                    float(forecast_date.month),
                    float(forecast_date.day),
                    float(forecast_date.dayofweek),
                    float(forecast_date.dayofyear),
                ])
            
            X_pred = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # Apply scaling if enabled
            if self.scaling and series_id in self.scalers_X:
                X_pred = self.scalers_X[series_id].transform(X_pred)

            # Make prediction
            try:
                pred_scaled = model.predict(X_pred)[0]
                
                # Inverse transform if scaling was applied
                if self.scaling and series_id in self.scalers_y:
                    pred = self.scalers_y[series_id].inverse_transform([[pred_scaled]])[0, 0]
                else:
                    pred = pred_scaled

                predictions.append(pred)
                
                # Update target context for next prediction (sliding window)
                target_context = np.roll(target_context, -1)
                target_context[-1] = pred

            except Exception as e:
                logging.warning(f"Prediction failed for series {series_id} at step {step}: {str(e)}")
                # Use last known value or zero
                last_val = predictions[-1] if predictions else 0
                predictions.append(last_val)

        return np.array(predictions)

    def _to_nixtla_df(
        self,
        predictions: np.ndarray,
        unique_ids: list[str],
        start_date: str,
        forecast_columns: ForecastColumnConfig,
        frequency: Frequency,
    ) -> pd.DataFrame:
        """
        Convert predictions to Nixtla-compatible DataFrame format
        
        Args:
            predictions: Array of predictions (n_series, horizon)
            unique_ids: List of series identifiers
            start_date: Start date for forecasting
            forecast_columns: Column configuration
            frequency: Time frequency
            
        Returns:
            pd.DataFrame: Formatted forecast DataFrame
        """
        n_series, horizon = predictions.shape
        if len(unique_ids) != n_series:
            raise ValueError(
                f"unique_ids must be length {n_series}, got {len(unique_ids)}"
            )

        # Build date index for forecast horizon
        pd_frequency = Frequency.get_alias(frequency, "pandas")
        ds = pd.date_range(
            start=start_date, periods=horizon + 1, freq=pd_frequency, inclusive="right"
        )

        # Create long-format DataFrame
        uid_col = np.repeat(unique_ids, horizon)
        cutoff = np.repeat([start_date], n_series * horizon)
        ds_col = np.tile(ds, n_series)
        yhat = predictions.flatten()

        df = pd.DataFrame(
            {
                forecast_columns.sku_index: uid_col,
                forecast_columns.date: ds_col,
                forecast_columns.cutoff: cutoff,
                self.alias: yhat,
            }
        )

        return df


