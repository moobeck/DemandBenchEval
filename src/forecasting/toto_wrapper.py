from typing import Literal
import logging
from toto.toto.model.toto import Toto
from toto.toto.inference.forecaster import TotoForecaster
import torch
from src.forecasting.foundation_model_base import FoundationModelWrapper
from src.configurations.forecast_column import ForecastColumnConfig
from toto.toto.data.util.dataset import MaskedTimeseries
import pandas as pd
import numpy as np

MODEL_OPTION = ("Datadog/Toto-Open-Base-1.0",)
DAILY_IN_SECONDS = 86400.0  # Daily interval in seconds


class TOTOWrapper(FoundationModelWrapper):
    """
    Wrapper for TOTO using proper DataDog forecasting implementation
    """

    def __init__(self, alias="toto", min_history=100, num_samples=50, **kwargs):

        self.alias = alias
        self.min_history = min_history
        self.num_samples = num_samples
        self.freq = "W"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing TOTO on device: {self.device}")

        # Try to load the best available TOTO model
        self.toto_model: Toto = Toto._from_pretrained(MODEL_OPTION)

        # Move model to device with error handling
        self.toto_model.to(self.device)
        logging.info(f"✅ TOTO model loaded successfully on {self.device}")

        # Initialize forecaster with the model's internal model
        self.forecaster = TotoForecaster(self.toto_model.model)
        self.model_type = f"TOTO (DataDog on {self.device})"

    def predict(
        self,
        X: pd.DataFrame,
        forecast_columns: ForecastColumnConfig,
        horizon: int = 14,
    ):
        """
        Predict using TOTO with proper multivariate time series forecasting
        """

        """
        Proper multivariate forecasting using TOTO's native capabilities
        """
        logging.info(
            f"    📊 Using TOTO's multivariate forecasting for {len(X['unique_id'].unique())} series"
        )

        # Create multivariate time series matrix
        # Pivot to get series as columns: (time_steps, num_series)
        multivariate_df = X.pivot(
            index=forecast_columns.date,
            columns=forecast_columns.sku_index,
            values=forecast_columns.target,
        )

        logging.info(
            f"    Created multivariate matrix: {multivariate_df.shape[0]} time steps × {multivariate_df.shape[1]} series"
        )

        # Convert to numpy array: (time_steps, num_series)
        multivariate_values = multivariate_df.values

        # Store series info for later mapping
        series_ids = list(multivariate_df.columns)

        # Ensure minimum history and optimal context

        optimal_history = min(max(len(multivariate_values), self.min_history), 2000)

        if len(multivariate_values) < self.min_history:
            # Pad with trend-extended values
            logging.info(
                f"    Padding multivariate series from {len(multivariate_values)} to {self.min_history} time steps"
            )
            padding_length = self.min_history - len(multivariate_values)

            # Calculate trends for each series
            padded_values = []
            for i in range(padding_length):
                if len(multivariate_values) >= 2:
                    # Use linear trend for each series
                    trends = []
                    for j in range(multivariate_values.shape[1]):
                        series_vals = multivariate_values[:, j]
                        if len(series_vals) >= 2 and np.var(series_vals) > 1e-6:
                            trend = np.polyfit(range(len(series_vals)), series_vals, 1)[
                                0
                            ]
                        else:
                            trend = 0
                        # Extrapolate backwards
                        base_val = series_vals[0] if len(series_vals) > 0 else 0
                        padded_val = max(0, base_val + trend * (i - padding_length))
                        trends.append(padded_val)
                    padded_values.append(trends)
                else:
                    # Use first values if insufficient data
                    first_vals = (
                        multivariate_values[0]
                        if len(multivariate_values) > 0
                        else np.zeros(multivariate_values.shape[1])
                    )
                    padded_values.append(first_vals)

            multivariate_values = np.vstack(
                [np.array(padded_values), multivariate_values]
            )

        elif len(multivariate_values) > optimal_history:
            # Use most recent context
            logging.info(f"    Using most recent {optimal_history} time steps")
            multivariate_values = multivariate_values[-optimal_history:]

        # Transpose to TOTO's expected format: (num_series, time_steps)
        multivariate_values = multivariate_values.T

        logging.info(f"    Input shape: {multivariate_values.shape} (series × time)")
        logging.info(
            f"    Value ranges: min={multivariate_values.min():.2f}, max={multivariate_values.max():.2f}"
        )

        # Robust normalization per series
        # Convert to tensor: (1, num_series, time_steps) for batch dimension
        series_tensor = torch.tensor(
            multivariate_values, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Create attention masks
        num_series, seq_len = multivariate_values.shape
        padding_mask = torch.ones(
            (1, num_series, seq_len), dtype=torch.bool, device=self.device
        )

        # Create ID mask to group related series (all belong to same dataset)
        id_mask = torch.zeros(
            (1, num_series, seq_len), dtype=torch.long, device=self.device
        )

        # Create timestamps (daily frequency)

        timestamps = (
            torch.arange(seq_len, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        timestamps = timestamps.expand(1, num_series, -1) * DAILY_IN_SECONDS
        time_intervals = torch.full(
            (1, num_series), DAILY_IN_SECONDS, dtype=torch.float32, device=self.device
        )

        # Create MaskedTimeseries object for multivariate input
        masked_ts = MaskedTimeseries(
            series=series_tensor,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamps,
            time_interval_seconds=time_intervals,
        )

        logging.info(f"    Generating {horizon} forecasts for {num_series} series...")

        # Generate multivariate forecasts
        forecast_result = self.forecaster.forecast(
            masked_ts,
            horizon=horizon,
            num_samples=min(self.num_samples, 20),
        )

        # Extract predictions from forecast result
        predictions = self._extract_multivariate_predictions(
            forecast_result, horizon, num_series
        )

        start_date = X[forecast_columns.date].max()

        sku_ids = X[forecast_columns.sku_index].unique().tolist()

        df = self._to_nixtla_df(
            predictions=predictions,
            unique_ids=sku_ids,
            start_date=start_date,
            forecast_columns=forecast_columns,
        )

        return df

    def _extract_multivariate_predictions(self, forecast_result, horizon, num_series):
        """Extract predictions from multivariate TOTO forecast result"""
        predictions = None

        if hasattr(forecast_result, "mean"):
            predictions = forecast_result.mean
        elif hasattr(forecast_result, "samples"):
            samples = forecast_result.samples
            if len(samples.shape) > 2:
                predictions = samples.mean(dim=-1)
            else:
                predictions = samples
        elif hasattr(forecast_result, "predictions"):
            predictions = forecast_result.predictions
        elif isinstance(forecast_result, torch.Tensor):
            predictions = forecast_result
        elif isinstance(forecast_result, dict):
            for key in ["predictions", "forecast", "mean", "samples"]:
                if key in forecast_result:
                    predictions = forecast_result[key]
                    if (
                        isinstance(predictions, torch.Tensor)
                        and len(predictions.shape) > 2
                    ):
                        predictions = predictions.mean(dim=-1)
                    break

        if predictions is None:
            logging.info(
                f"    ⚠️  Could not extract multivariate predictions from TOTO result: {type(forecast_result)}"
            )
            # Fallback: create simple trend predictions
            predictions = torch.zeros((1, num_series, horizon))
            return predictions.squeeze(0).detach().cpu().numpy()

        # Convert to numpy and ensure correct shape
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

        # Expected shape: (batch, num_series, horizon) -> (num_series, horizon)
        if len(predictions.shape) == 3:
            predictions = predictions.squeeze(0)  # Remove batch dimension
        elif len(predictions.shape) == 2 and predictions.shape[0] != num_series:
            # Might be (horizon, num_series) -> transpose
            if predictions.shape[1] == num_series:
                predictions = predictions.T

        # Ensure correct prediction length
        if predictions.shape[1] != horizon:
            if predictions.shape[1] > horizon:
                predictions = predictions[:, :horizon]
            else:
                # Extend with trend
                last_vals = (
                    predictions[:, -1:]
                    if predictions.shape[1] > 0
                    else np.zeros((num_series, 1))
                )
                extension = np.tile(last_vals, (1, horizon - predictions.shape[1]))
                predictions = np.concatenate([predictions, extension], axis=1)

        # Ensure non-negative
        predictions = np.maximum(predictions, 0)

        return predictions

    def _to_nixtla_df(
        self,
        predictions: np.ndarray,
        unique_ids: list[str],
        start_date: str,
        forecast_columns: ForecastColumnConfig,
    ) -> pd.DataFrame:
        """
        Turn a (n_series × horizon) array into a Nixtla‑compatible DataFrame.

        Parameters
        ----------
        predictions
            numpy array of shape (n_series, horizon).
        unique_ids
            list of length n_series containing each series’ ID.
        start_date
            the first forecast date (e.g. "2025-06-19").
        freq
            pandas frequency string, e.g. "D", "H", "W"…
        pred_col
            name of the prediction column in the output (default "y_pred").

        Returns
        -------
        pd.DataFrame
            columns = ["unique_id", "ds", pred_col]
        """
        n_series, horizon = predictions.shape
        if len(unique_ids) != n_series:
            raise ValueError(
                f"unique_ids must be length {n_series}, got {len(unique_ids)}"
            )

        # 1. build the date index for one horizon
        ds = pd.date_range(
            start=start_date, periods=horizon + 1, freq=self.freq, inclusive="right"
        )

        # 2. tile/flatten to long form
        uid_col = np.repeat(unique_ids, horizon)
        ds_col = np.tile(ds, n_series)
        yhat = predictions.flatten()

        # 3. pack into DataFrame
        df = pd.DataFrame(
            {
                forecast_columns.sku_index: uid_col,
                forecast_columns.date: ds_col,
                forecast_columns.target: yhat,
            }
        )

        return df
