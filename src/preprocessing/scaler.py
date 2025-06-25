from mlforecast.target_transforms import BaseTargetTransform
import pandas as pd
from src.configurations.cross_validation import CrossValidationConfig
from src.configurations.enums import Frequency


class LocalStandardScaler(BaseTargetTransform):
    """Standardizes each series to have mean 0 and standard deviation 1,
    based on a training cutoff defined by the CV config."""

    def __init__(self,
                 cv_cfg: CrossValidationConfig,
                 freq: Frequency):
        self.cv_cfg = cv_cfg
        self.freq = freq
        self.stats_: pd.DataFrame

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) Compute offset
        if self.freq == Frequency.DAILY:
            offset = pd.Timedelta(days=self.cv_cfg.cv_windows
                                   * self.cv_cfg.step_size)
        elif self.freq == Frequency.WEEKLY:
            offset = pd.Timedelta(weeks=self.cv_cfg.cv_windows
                                   * self.cv_cfg.step_size)
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        cutoff = df[self.time_col].max() - offset

        df_train = df[df[self.time_col] <= cutoff]

        self.stats_ = (
            df_train
            .groupby(self.id_col)[self.target_col]
            .agg(mean_='mean', std_='std')
            .reset_index()
        )
        # Avoid division by zero
        self.stats_['std_'].replace(0, 1.0, inplace=True)

        df = df.merge(self.stats_, on=self.id_col, how='left')
        df[self.target_col] = (
            df[self.target_col] - df['mean_']
        ) / df['std_']
        return df.drop(columns=['mean_', 'std_'])

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Merge stats back on
        df = df.merge(self.stats_, on=self.id_col, how='left')
        # Only invert the target column
        df[self.target_col] = df[self.target_col] * df['std_'] + df['mean_']
        return df.drop(columns=['mean_', 'std_'])
