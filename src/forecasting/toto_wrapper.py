from typing import Literal
import logging
from toto.toto.model.toto import Toto
from toto.toto.inference.forecaster import TotoForecaster
import torch
from src.forecasting.foundation_model_base import FoundationModelWrapper
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.enums import Frequency
from toto.toto.data.util.dataset import MaskedTimeseries
import pandas as pd
import numpy as np

MODEL_OPTION = "Datadog/Toto-Open-Base-1.0"
DAILY_IN_SECONDS = 86400.0  # Daily interval in seconds


class TOTOWrapper(FoundationModelWrapper):
    """
    Wrapper for TOTO using proper DataDog forecasting implementation
    """

    def __init__(self, alias="toto", min_history=100, num_samples=50, **kwargs):

        self.alias = alias
        self.min_history = min_history
        self.num_samples = num_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing TOTO on device: {self.device}")

        # Try to load the best available TOTO model
        self.toto_model: Toto = Toto.from_pretrained(MODEL_OPTION)

        # Move model to device with error handling
        self.toto_model.to(self.device)
        logging.info(f"TOTO model loaded successfully on {self.device}")

        # Initialize forecaster with the model's internal model
        self.forecaster = TotoForecaster(self.toto_model.model)
        self.model_type = f"TOTO (DataDog on {self.device})"

    def predict(
        self,
        X: pd.DataFrame,
        forecast_columns: ForecastColumnConfig,
        horizon: int,
        freq: Frequency
    ):
        """
        Predict using TOTO with proper multivariate time series forecasting
        """


        # Create and preprocess multivariate time series matrix
        multivariate_values = self._create_multivariate_matrix(X, forecast_columns)

        masked_ts = self._prepare_masked_timeseries(multivariate_values)

        # Generate multivariate forecasts
        forecast_result = self.forecaster.forecast(
            masked_ts,
            prediction_length=horizon,
            num_samples=self.num_samples,
        )

        predictions = forecast_result.mean.detach().cpu().numpy().squeeze(axis=0)
        unique_ids = X[forecast_columns.sku_index].unique().tolist()
        df = self._to_nixtla_df(
            predictions=predictions,
            unique_ids=unique_ids,
            start_date=X[forecast_columns.date].max(),
            forecast_columns=forecast_columns,
            frequency=freq,
        )

        return df

    def _create_multivariate_matrix(
        self, X: pd.DataFrame, forecast_columns: ForecastColumnConfig
    ) -> tuple:
        """
        Create a multivariate time series matrix from the input DataFrame.

        Args:
            X: Input DataFrame with time series data
            forecast_columns: Configuration object with column names

        Returns:
            tuple: (multivariate_values, series_ids) where multivariate_values is a numpy array
                with shape (time_steps, num_series) and series_ids is a list of series identifiers
        """
        # Pivot to get series as columns: (time_steps, num_series)
        multivariate_df = X.pivot(
            index=forecast_columns.date,
            columns=forecast_columns.sku_index,
            values=forecast_columns.target,
        )

        # Convert to numpy array and store column mapping for later reference
        multivariate_values = multivariate_df.values

        return multivariate_values
    
    def _pad_time_series(self, values: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad time series with trend-based extrapolation to reach target length.

        Args:
            values: Input array of shape (time_steps, num_series)
            target_length: Desired length after padding

        Returns:
            np.ndarray: Padded array with shape (target_length, num_series)
        """
        # Early return if no padding needed
        if len(values) >= target_length:
            return values

        padding_length = target_length - len(values)

        # Prepare container for padded values
        padded_values = []

        # Handle the case of insufficient data points for trend calculation
        if len(values) < 2:
            # Use first values or zeros if no data available
            first_vals = values[0] if len(values) > 0 else np.zeros(values.shape[1])
            padded_values = [first_vals] * padding_length
        else:
            # Calculate linear trends for each series
            for i in range(padding_length):
                trends = []
                for j in range(values.shape[1]):
                    series_vals = values[:, j]
                    # Calculate linear trend if enough variance in data
                    if len(series_vals) >= 2 and np.var(series_vals) > 1e-6:
                        trend = np.polyfit(range(len(series_vals)), series_vals, 1)[0]
                    else:
                        trend = 0
                    # Extrapolate backwards
                    base_val = series_vals[0] if len(series_vals) > 0 else 0
                    padded_val = max(0, base_val + trend * (i - padding_length))
                    trends.append(padded_val)
                padded_values.append(trends)

        # Stack padded values with original data
        return np.vstack([np.array(padded_values), values])

    def _prepare_masked_timeseries(
        self, multivariate_values: np.ndarray, time_interval: float = DAILY_IN_SECONDS
    ) -> MaskedTimeseries:
        """
        Prepare a MaskedTimeseries object from numpy array for TOTO model input.

        This method:
        1. Transposes data to TOTO's expected format
        2. Converts to PyTorch tensor
        3. Creates appropriate masks and timestamps
        4. Assembles the MaskedTimeseries object

        Args:
            multivariate_values: Numpy array with shape (time_steps, num_series)
            time_interval: Time interval between steps in seconds (default: daily)

        Returns:
            MaskedTimeseries: Properly formatted input for TOTO forecaster

        Raises:
            ValueError: If input array is empty or improperly shaped
            RuntimeError: If tensor creation fails
        """
        # Validate input
        if multivariate_values.size == 0:
            raise ValueError("Empty multivariate array provided")

        # Transpose to TOTO's expected format: (num_series, time_steps)
        multivariate_values = multivariate_values.T
        num_series, seq_len = multivariate_values.shape


        # Convert to tensor with batch dimension: (1, num_series, time_steps)
        series_tensor = torch.tensor(
            multivariate_values, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Create tensor components with consistent dimensions
        tensors = self._create_tensor_components(num_series, seq_len, time_interval)

        # Assemble MaskedTimeseries object
        return MaskedTimeseries(
            series=series_tensor,
            padding_mask=tensors["padding_mask"],
            id_mask=tensors["id_mask"],
            timestamp_seconds=tensors["timestamps"],
            time_interval_seconds=tensors["time_intervals"],
        )

    def _create_tensor_components(
        self, num_series: int, seq_len: int, time_interval: float
    ) -> dict:
        """
        Create tensor components needed for MaskedTimeseries.

        Args:
            num_series: Number of time series
            seq_len: Length of each time series
            time_interval: Time interval between steps in seconds

        Returns:
            dict: Dictionary containing all required tensor components
        """
        # Create attention mask (all values observed)
        padding_mask = torch.ones(
            (1, num_series, seq_len), dtype=torch.bool, device=self.device
        )

        # Create ID mask (all series in same group)
        id_mask = torch.zeros(
            (1, num_series, seq_len), dtype=torch.long, device=self.device
        )

        # Create timestamps tensor
        timestamps = (
            torch.arange(seq_len, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_series, -1)
            * time_interval
        )

        # Create time intervals tensor
        time_intervals = torch.full(
            (1, num_series), time_interval, dtype=torch.float32, device=self.device
        )

        return {
            "padding_mask": padding_mask,
            "id_mask": id_mask,
            "timestamps": timestamps,
            "time_intervals": time_intervals,
        }


    def _to_nixtla_df(
        self,
        predictions: np.ndarray,
        unique_ids: list[str],
        start_date: str,
        forecast_columns: ForecastColumnConfig,
        frequency: Frequency,
    ) -> pd.DataFrame:
        """
        Turn a (n_series × horizon) array into a Nixtla‑compatible DataFrame.
        """
        n_series, horizon = predictions.shape
        if len(unique_ids) != n_series:
            raise ValueError(
                f"unique_ids must be length {n_series}, got {len(unique_ids)}"
            )
        

        # 1. build the date index for one horizon
        pd_frequency = Frequency.get_alias(frequency, "pandas")
        ds = pd.date_range(
            start=start_date, periods=horizon+1, freq=pd_frequency, inclusive="right"
        )

        # 2. tile/flatten to long form
        uid_col = np.repeat(unique_ids, horizon)
        cutoff = np.repeat([start_date], n_series * horizon)
        ds_col = np.tile(ds, n_series)
        yhat = predictions.flatten()

        # 3. pack into DataFrame
        df = pd.DataFrame(
            {
                forecast_columns.sku_index: uid_col,
                forecast_columns.date: ds_col,
                forecast_columns.cutoff: cutoff,
                self.alias: yhat,
            }
        )

        return df
