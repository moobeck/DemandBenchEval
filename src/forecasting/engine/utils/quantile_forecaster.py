import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.configurations.data.forecast_column import ForecastColumnConfig


class QuantileForecaster:
    """Handles quantile-based forecast interval calculations based on iid in-sample errors."""

    def __init__(self, models: List, forecast_columns: ForecastColumnConfig):
        """
        Initialize QuantileForecaster.

        Args:
            models: List of model names
            forecast_columns: Configuration for forecast columns
        """
        self.models = models
        self.forecast_columns = forecast_columns

    def add_quantiles(
        self, cv_df: pd.DataFrame, in_sample_fcst: pd.DataFrame, quantiles: List[float]
    ) -> pd.DataFrame:
        """
        Add quantile columns to the forecast results DataFrame.

        Args:
            cv_df: Cross-validation results DataFrame
            in_sample_fcst: In-sample forecast DataFrame
            quantiles: List of quantile values (e.g., [0.05, 0.95])

        Returns:
            DataFrame with added quantile columns
        """
        error_quantiles_df = self._calculate_absolute_errors(
            cv_df, in_sample_fcst, quantiles
        )
        cv_df = self._append_quantile_columns(cv_df, error_quantiles_df, quantiles)
        return cv_df

    def _calculate_absolute_errors(
        self, cv_df: pd.DataFrame, in_sample_fcst: pd.DataFrame, quantiles: List[float]
    ) -> pd.DataFrame:
        """
        Calculate quantiles of absolute in-sample errors for each time series and model.

        Args:
            cv_df: Cross-validation results DataFrame
            in_sample_fcst: In-sample forecast DataFrame
            quantiles: List of quantile values

        Returns:
            DataFrame with columns: [time_series_index, model, quantile, error_quantile]
        """
        # Calculate absolute errors for all models at once
        error_data = []

        for model in self.models:
            # Vectorized absolute error calculation
            errors = (
                in_sample_fcst[self.forecast_columns.target] - in_sample_fcst[model]
            ).abs()
            errors_df = pd.DataFrame(
                {
                    self.forecast_columns.time_series_index: in_sample_fcst[
                        self.forecast_columns.time_series_index
                    ],
                    "error": errors,
                }
            )

            # Vectorized quantile calculation using groupby
            for q in quantiles:
                quantile_vals = (
                    errors_df.groupby(self.forecast_columns.time_series_index)["error"]
                    .quantile(q)
                    .reset_index()
                )
                quantile_vals.columns = [
                    self.forecast_columns.time_series_index,
                    "error_quantile",
                ]
                quantile_vals["model"] = model
                quantile_vals["quantile"] = q
                error_data.append(quantile_vals)

        # Combine all results into a single DataFrame
        return pd.concat(error_data, ignore_index=True)
    

    @staticmethod
    def quantiles_to_outputs(quantiles):
        output_names = []
        for q in quantiles:
            if q < 0.50:
                output_names.append(f"-lo-{int(round(100-200*q, 0))}")
            elif q > 0.50:
                output_names.append(f"-hi-{int(round(100-200*(1-q), 0))}")
            else:
                output_names.append("-median")
        return quantiles, output_names


    def _append_quantile_columns(
        self,
        cv_df: pd.DataFrame,
        error_quantiles_df: pd.DataFrame,
        quantiles: List[float],
    ) -> pd.DataFrame:
        """
        Append quantile columns to the DataFrame.

        Args:
            cv_df: Cross-validation results DataFrame
            error_quantiles_df: DataFrame with error quantiles
            quantiles: List of quantile values

        Returns:
            DataFrame with appended quantile columns
        """
        for model in self.models:
            model_errors = error_quantiles_df[error_quantiles_df["model"] == model]

            for q, output_name in zip(*self.quantiles_to_outputs(quantiles)):
                quantile_col = f"{model}{output_name}"

                # Get error quantiles for this model and quantile
                q_errors = model_errors[model_errors["quantile"] == q][
                    [self.forecast_columns.time_series_index, "error_quantile"]
                ].copy()
                q_errors.columns = [
                    self.forecast_columns.time_series_index,
                    quantile_col + "_error",
                ]

                # Vectorized merge and addition
                cv_df = cv_df.merge(
                    q_errors, on=self.forecast_columns.time_series_index, how="left"
                )
                cv_df[quantile_col] = cv_df[model] + cv_df[quantile_col + "_error"]
                cv_df.drop(columns=[quantile_col + "_error"], inplace=True)

        print(f"CV df: {cv_df.head()}")
        print(f"Number of columns in CV df: {len(cv_df.columns)}")

        return cv_df
