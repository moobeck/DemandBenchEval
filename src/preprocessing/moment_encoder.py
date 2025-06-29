import scipy
import pandas as pd
import scipy.stats
from src.configurations.cross_validation import CrossValidationConfig
from src.configurations.enums import Frequency
from src.configurations.forecast_column import ForecastColumnConfig


class MomentsEncoder:

    def __init__(
        self,
        cv_cfg: CrossValidationConfig,
        freq: Frequency,
        forecast_columns: ForecastColumnConfig,
    ):
        """
        Initializes the MomentsEncoder with the specified configuration.
        """

        self.cv_cfg = cv_cfg
        self.freq = freq
        self.forecast_columns = forecast_columns
        self.skewness_col = "skewness"
        self.kurtosis_col = "kurtosis"
        self.out_columns = [self.skewness_col, self.kurtosis_col]



    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) compute cutoff exactly as before
        if self.freq == Frequency.DAILY:
            offset = pd.Timedelta(days=self.cv_cfg.cv_windows * self.cv_cfg.step_size)
        elif self.freq == Frequency.WEEKLY:
            offset = pd.Timedelta(weeks=self.cv_cfg.cv_windows * self.cv_cfg.step_size)
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        cutoff = df[self.forecast_columns.date].max() - offset
        df_train = df[df[self.forecast_columns.date] <= cutoff]

        # 2) vectorized groupby + agg
        stats = (
            df_train
            .groupby(self.forecast_columns.sku_index)[self.forecast_columns.target]
            .agg(
                skewness=lambda x: scipy.stats.skew(x) if len(x) >= 4 else 0,
                kurtosis=lambda x: scipy.stats.kurtosis(x) if len(x) >= 4 else 0,
            )        
        )

        # 3) merge back to full df
        #    this automatically broadcasts per-SKU stats to all rows of that SKU
        df = df.merge(
            stats.reset_index(),
            how="left",
            left_on=self.forecast_columns.sku_index,
            right_on=self.forecast_columns.sku_index,
        )

        # 4) rename or assign to your desired column names
        df = df.rename(columns={
            "skewness": self.skewness_col,
            "kurtosis": self.kurtosis_col
        })

        return df

