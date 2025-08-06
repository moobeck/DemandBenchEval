from typing import Dict, Any
import logging
import torch
import pandas as pd
import numpy as np
from src.forecasting.foundation_model_base import FoundationModelWrapper
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.enums import Frequency, TimeInSeconds, ModelName

# Import from the cloned TOTO repository
import sys
import os

# Add the toto subdirectory to path to access the modules
toto_path = os.path.join(os.getcwd(), "toto", "toto")
if toto_path not in sys.path:
    sys.path.insert(0, toto_path)

from model.toto import Toto
from inference.forecaster import TotoForecaster
from data.util.dataset import MaskedTimeseries


class TOTOWrapper(FoundationModelWrapper):
    """
    Wrapper for TOTO using proper DataDog forecasting implementation
    """

    def __init__(
        self,
        num_samples: int = 50,
        samples_per_batch: int = 25,
        alias="Toto",
        model_option="Datadog/Toto-Open-Base-1.0",
        max_context_length: int = 2048,
        max_series_batch: int = 10,
        **kwargs,
    ):

        self.alias = alias
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.max_context_length = max_context_length
        self.max_series_batch = max_series_batch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"Initializing TOTO on device: {self.device}")
        logging.info(
            f"Memory-optimized settings: samples={num_samples}, samples_per_batch={samples_per_batch}, max_context={max_context_length}"
        )

        # Validate that num_samples is divisible by samples_per_batch
        if num_samples % samples_per_batch != 0:
            raise ValueError(
                f"num_samples ({num_samples}) must be divisible by samples_per_batch ({samples_per_batch})"
            )

        try:
            # Load the TOTO model
            self.toto_model: Toto = Toto.from_pretrained(model_option)

            # Move model to device
            self.toto_model.to(self.device)

            # Enable eval mode for inference
            self.toto_model.eval()

            # Don't compile for now to save memory
            # if hasattr(self.toto_model, 'compile'):
            #     try:
            #         self.toto_model.compile()
            #         logging.info("TOTO model compiled for faster inference")
            #     except Exception as e:
            #         logging.warning(f"Could not compile TOTO model: {e}")

            logging.info(f"TOTO model loaded successfully on {self.device}")

            # Initialize forecaster with the model's internal backbone
            self.forecaster = TotoForecaster(self.toto_model.model)
            self.model_type = f"TOTO (DataDog on {self.device})"

            # Log GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logging.info(
                    f"GPU memory after model load: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
                )

        except Exception as e:
            logging.error(f"Failed to initialize TOTO model: {e}")
            raise RuntimeError(f"TOTO initialization failed: {e}")

    def predict(
        self,
        X: pd.DataFrame,
        forecast_columns: ForecastColumnConfig,
        horizon: int,
        freq: Frequency,
    ):
        """
        Predict using TOTO with proper multivariate time series forecasting
        """
        try:
            # Clear GPU cache before prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create multivariate time series matrix
            multivariate_values, series_ids = self._create_multivariate_matrix(
                X, forecast_columns
            )

            # Check if we need to truncate context for memory
            if multivariate_values.shape[0] > self.max_context_length:
                logging.warning(
                    f"Truncating context from {multivariate_values.shape[0]} to {self.max_context_length} timesteps"
                )
                multivariate_values = multivariate_values[-self.max_context_length :]

            # Check if we have too many series - batch them if needed
            n_series = len(series_ids)
            if n_series > self.max_series_batch:
                logging.info(
                    f"Processing {n_series} series in batches of {self.max_series_batch}"
                )
                return self._predict_in_batches(
                    multivariate_values,
                    series_ids,
                    forecast_columns,
                    horizon,
                    freq,
                    start_date=X[forecast_columns.date].max(),
                )

            # Prepare the MaskedTimeseries object
            masked_ts = self._prepare_masked_timeseries(multivariate_values, freq)

            # Use samples_per_batch from config

            # Generate forecasts using the forecaster
            with (
                torch.cuda.amp.autocast()
                if torch.cuda.is_available()
                else torch.no_grad()
            ):
                forecast_result = self.forecaster.forecast(
                    masked_ts,
                    prediction_length=horizon,
                    num_samples=self.num_samples,
                    samples_per_batch=self.samples_per_batch,
                    use_kv_cache=True,
                )

            # Extract predictions (use median for better performance than mean)
            if forecast_result.samples is not None:
                predictions = forecast_result.median.detach().cpu().numpy()
            else:
                predictions = forecast_result.mean.detach().cpu().numpy()

            # Clean up GPU tensors immediately
            del forecast_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Remove batch dimension if present
            if predictions.ndim == 3 and predictions.shape[0] == 1:
                predictions = predictions.squeeze(0)

            # Convert to Nixtla format
            df = self._to_nixtla_df(
                predictions=predictions,
                unique_ids=series_ids,
                start_date=X[forecast_columns.date].max(),
                forecast_columns=forecast_columns,
                frequency=freq,
            )

            return df

        except Exception as e:
            logging.error(f"TOTO prediction failed: {e}")
            raise RuntimeError(f"TOTO prediction error: {e}")

    def _predict_in_batches(
        self,
        multivariate_values: np.ndarray,
        series_ids: list[str],
        forecast_columns: ForecastColumnConfig,
        horizon: int,
        freq: Frequency,
        start_date: str,
    ) -> pd.DataFrame:
        """
        Process large numbers of series in smaller batches to manage memory
        """
        all_predictions = []
        n_series = len(series_ids)

        for start_idx in range(0, n_series, self.max_series_batch):
            end_idx = min(start_idx + self.max_series_batch, n_series)
            batch_series_ids = series_ids[start_idx:end_idx]
            batch_values = multivariate_values[:, start_idx:end_idx]

            logging.info(
                f"Processing batch {start_idx//self.max_series_batch + 1}: series {start_idx+1}-{end_idx} of {n_series}"
            )

            try:
                # Prepare masked timeseries for this batch
                masked_ts = self._prepare_masked_timeseries(batch_values, freq)

                # Generate forecasts for this batch
                with (
                    torch.cuda.amp.autocast()
                    if torch.cuda.is_available()
                    else torch.no_grad()
                ):
                    forecast_result = self.forecaster.forecast(
                        masked_ts,
                        prediction_length=horizon,
                        num_samples=self.num_samples,
                        samples_per_batch=self.samples_per_batch,
                        use_kv_cache=True,
                    )

                # Extract predictions
                if forecast_result.samples is not None:
                    batch_predictions = forecast_result.median.detach().cpu().numpy()
                else:
                    batch_predictions = forecast_result.mean.detach().cpu().numpy()

                # Clean up GPU memory immediately
                del forecast_result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Remove batch dimension if present
                if batch_predictions.ndim == 3 and batch_predictions.shape[0] == 1:
                    batch_predictions = batch_predictions.squeeze(0)

                # Convert to Nixtla format for this batch
                batch_df = self._to_nixtla_df(
                    predictions=batch_predictions,
                    unique_ids=batch_series_ids,
                    start_date=start_date,
                    forecast_columns=forecast_columns,
                    frequency=freq,
                )

                all_predictions.append(batch_df)

            except Exception as e:
                logging.error(
                    f"Failed processing batch {start_idx//self.max_series_batch + 1}: {e}"
                )
                raise

        # Combine all batch predictions
        combined_df = pd.concat(all_predictions, ignore_index=True)
        return combined_df

    def _create_multivariate_matrix(
        self, X: pd.DataFrame, forecast_columns: ForecastColumnConfig
    ) -> tuple[np.ndarray, list[str]]:
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

        # Sort columns for consistent ordering
        multivariate_df = multivariate_df.sort_index(axis=1)

        # Extract series IDs and convert to list
        series_ids = multivariate_df.columns.tolist()

        # Convert to numpy array and handle NaNs
        multivariate_values = multivariate_df.values

        # Replace NaNs with zeros (TOTO can handle zeros with proper masking)
        multivariate_values = np.nan_to_num(multivariate_values, nan=0.0)

        return multivariate_values, series_ids

    def _prepare_masked_timeseries(
        self, multivariate_values: np.ndarray, frequency: Frequency
    ) -> MaskedTimeseries:
        """
        Prepare a MaskedTimeseries object from numpy array for TOTO model input.

        Args:
            multivariate_values: Numpy array with shape (time_steps, num_series)
            frequency: Time frequency of the data

        Returns:
            MaskedTimeseries: Properly formatted input for TOTO forecaster
        """
        # Validate input
        if multivariate_values.size == 0:
            raise ValueError("Empty multivariate array provided")

        # Transpose to TOTO's expected format: (num_series, time_steps)
        multivariate_values = multivariate_values.T
        num_series, seq_len = multivariate_values.shape

        # Convert to tensor: (num_series, time_steps) - no batch dimension yet
        # Use float16 if on GPU for memory efficiency, float32 otherwise
        # Note: Only the main series data uses float16, masks and timestamps use appropriate types
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        series_tensor = torch.tensor(
            multivariate_values, dtype=dtype, device=self.device
        )

        # Create padding mask (True for valid values)
        # All values are valid since we replaced NaNs with zeros
        padding_mask = torch.ones(
            (num_series, seq_len), dtype=torch.bool, device=self.device
        )

        # Create ID mask - all series belong to the same group (0)
        id_mask = torch.zeros(
            (num_series, seq_len), dtype=torch.long, device=self.device
        )

        # Create timestamps (dummy timestamps since TOTO doesn't use them in current version)
        # Each time step gets an incrementally increasing timestamp
        time_interval = int(TimeInSeconds[frequency.name].value)
        # Use int64 for timestamps as expected by TOTO
        timestamps = torch.arange(seq_len, dtype=torch.int64, device=self.device)
        timestamps = timestamps.unsqueeze(0).expand(num_series, -1) * time_interval

        # Create time intervals tensor (int64 as expected by TOTO)
        time_intervals = torch.full(
            (num_series,), time_interval, dtype=torch.int64, device=self.device
        )

        # Create MaskedTimeseries object (without batch dimension - TotoForecaster will add it)
        return MaskedTimeseries(
            series=series_tensor,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamps,
            time_interval_seconds=time_intervals,
        )

    def _to_nixtla_df(
        self,
        predictions: np.ndarray,
        unique_ids: list[str],
        start_date: str,
        forecast_columns: ForecastColumnConfig,
        frequency: Frequency,
    ) -> pd.DataFrame:
        """
        Convert TOTO predictions to Nixtla-compatible DataFrame format.

        Args:
            predictions: Numpy array with shape (n_series, horizon)
            unique_ids: List of series identifiers
            start_date: Starting date for predictions
            forecast_columns: Column configuration
            frequency: Data frequency

        Returns:
            pd.DataFrame: Nixtla-compatible forecast DataFrame
        """
        n_series, horizon = predictions.shape

        if len(unique_ids) != n_series:
            raise ValueError(
                f"unique_ids length ({len(unique_ids)}) must match n_series ({n_series})"
            )

        # Create date index for forecast horizon
        pd_frequency = Frequency.get_alias(frequency, "pandas")
        ds = pd.date_range(
            start=start_date, periods=horizon + 1, freq=pd_frequency, inclusive="right"
        )

        # Create long-form DataFrame
        uid_col = np.repeat(unique_ids, horizon)
        cutoff_col = np.repeat([start_date], n_series * horizon)
        ds_col = np.tile(ds, n_series)
        yhat_col = predictions.flatten()

        # Assemble DataFrame
        df = pd.DataFrame(
            {
                forecast_columns.sku_index: uid_col,
                forecast_columns.date: ds_col,
                forecast_columns.cutoff: cutoff_col,
                self.alias: yhat_col,
            }
        )

        return df

    def __del__(self):
        """Clean up GPU memory when wrapper is destroyed"""
        try:
            toto_model = getattr(self, "toto_model", None)
            if toto_model is not None:
                del self.toto_model
            forecaster = getattr(self, "forecaster", None)
            if forecaster is not None:
                del self.forecaster
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore cleanup errors
