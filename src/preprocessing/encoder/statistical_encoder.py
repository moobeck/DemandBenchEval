import pandas as pd
import scipy.stats
from src.configurations.evaluation.cross_validation import CrossValidationConfig
from src.configurations.utils.enums import Frequency
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.forecasting import ForecastConfig


class StatisticalFeaturesEncoder:

    def __init__(
        self,
        cv_cfg: CrossValidationConfig,
        freq: Frequency,
        forecast_columns: ForecastColumnConfig,
        forecast: ForecastConfig,
    ):
        """
        Initializes the MomentsEncoder with the specified configuration.
        """

        self.cv_cfg = cv_cfg
        self.freq = freq
        self.forecast_columns = forecast_columns
        self.forecast = forecast
        self.mean_col = f"{self.forecast_columns.target}_mean"
        self.std_col = f"{self.forecast_columns.target}_std"
        self.skewness_col = f"{self.forecast_columns.target}_skewness"
        self.kurtosis_col = f"{self.forecast_columns.target}_kurtosis"
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantile_cols = [
            f"{self.forecast_columns.target}_quantile_{q:.2f}" for q in self.quantiles
        ]
        self.out_columns = [self.skewness_col, self.kurtosis_col] + self.quantile_cols

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) compute cutoff exactly as before
        cutoff = self.cv_cfg.get_cutoff_date(
            max_date=df[self.forecast_columns.date].max(),
            freq=self.freq,
            split="test",
            horizon=self.forecast.horizon,
        )

        df_train = df[df[self.forecast_columns.date] <= cutoff]

        # 2) vectorized groupby + agg
        stats = df_train.groupby(self.forecast_columns.time_series_index)[
            self.forecast_columns.target
        ].agg(
            mean="mean",
            std="std",
            skewness=lambda x: scipy.stats.skew(x) if len(x) >= 4 else 0,
            kurtosis=lambda x: scipy.stats.kurtosis(x) if len(x) >= 4 else 0,
            **{
                f"quantile_{q:.2f}": lambda x, q=q: x.quantile(q) if len(x) >= 4 else 0
                for q in self.quantiles
            },
        )

        # 3) merge back to full df
        #    this automatically broadcasts per-SKU stats to all rows of that SKU
        df = df.merge(
            stats.reset_index(),
            how="left",
            left_on=self.forecast_columns.time_series_index,
            right_on=self.forecast_columns.time_series_index,
        )

        # 4) rename or assign to your desired column names
        df = df.rename(
            columns={
                "mean": self.mean_col,
                "std": self.std_col,
                "skewness": self.skewness_col,
                "kurtosis": self.kurtosis_col,
                **{
                    f"quantile_{q:.2f}": col
                    for q, col in zip(self.quantiles, self.quantile_cols)
                },
            }
        )

        return df
