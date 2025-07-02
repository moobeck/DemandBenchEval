from mlforecast.target_transforms import BaseTargetTransform
import pandas as pd
from src.configurations.cross_validation import CrossValidationConfig
from src.configurations.enums import Frequency, TargetScalerType
from src.configurations.preprocessing import PreprocessingConfig



class TargetScaler(BaseTargetTransform):
    """Base class for target scaling transforms that use a cutoff based on cross-validation configuration."""

    def __init__(self, cv_cfg: CrossValidationConfig, freq: Frequency):
        self.cv_cfg = cv_cfg
        self.freq = freq
        self.stats_: pd.DataFrame = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")

    def _calculate_cutoff(self, df: pd.DataFrame) -> pd.Timestamp:
        """Calculate the cutoff timestamp based on frequency and CV configuration."""
        if self.freq == Frequency.DAILY:
            offset = pd.Timedelta(days=self.cv_cfg.cv_windows * self.cv_cfg.step_size)
        elif self.freq == Frequency.WEEKLY:
            offset = pd.Timedelta(weeks=self.cv_cfg.cv_windows * self.cv_cfg.step_size)
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        return df[self.time_col].max() - offset



class TargetScalerFactory:
    """Factory class to create target scalers based on configuration."""

    @staticmethod
    def create_scaler(scaler_type: PreprocessingConfig, cv_cfg: CrossValidationConfig, freq: Frequency):
        """Create a target scaler based on the specified type."""
        if scaler_type.target_transform == TargetScalerType.LOCAL_STANDARD:
            return LocalStandardScaler(cv_cfg, freq)
        elif scaler_type.target_transform == TargetScalerType.LOCAL_MAX:
            return LocalMaxScaler(cv_cfg, freq)
        elif scaler_type.target_transform == TargetScalerType.LOCAL_ROBUST:
            return LocalRobustScaler(cv_cfg, freq)
        else:
            raise ValueError(f"Unsupported target scaler type: {scaler_type.target_transform}")
      


class LocalStandardScaler(TargetScaler):
    """Standardizes each series to have mean 0 and standard deviation 1,
    based on a training cutoff defined by the CV config."""

    def __init__(self, cv_cfg: CrossValidationConfig, freq: Frequency):
        self.cv_cfg = cv_cfg
        self.freq = freq
        self.stats_: pd.DataFrame = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cutoff = self._calculate_cutoff(df)

        df_train = df[df[self.time_col] <= cutoff]

        self.stats_ = (
            df_train.groupby(self.id_col)[self.target_col]
            .agg(mean_="mean", std_="std")
            .reset_index()
        )
        # Avoid division by zero
        self.stats_["std_"].replace(0, 1.0, inplace=True)

        df = df.merge(self.stats_, on=self.id_col, how="left")
        df[self.target_col] = (df[self.target_col] - df["mean_"]) / df["std_"]
        return df.drop(columns=["mean_", "std_"])

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Merge stats back on
        df = df.merge(self.stats_, on=self.id_col, how="left")
        # Only invert the target column
        df[self.target_col] = df[self.target_col] * df["std_"] + df["mean_"]
        return df.drop(columns=["mean_", "std_"])


class LocalMaxScaler(TargetScaler):
    """Scales each series by dividing by the maximum value in the training set,
    based on a training cutoff defined by the CV config."""

    def __init__(self, cv_cfg: CrossValidationConfig, freq: Frequency):
        self.cv_cfg = cv_cfg
        self.freq = freq
        self.stats_: pd.DataFrame = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:


        cutoff = self._calculate_cutoff(df)

        df_train = df[df[self.time_col] <= cutoff]


        self.stats_ = (
            df_train.groupby(self.id_col)[self.target_col]
            .agg(max_="max")
            .reset_index()
        )
        # Avoid division by zero
        self.stats_["max_"].replace(0, 1.0, inplace=True)

        df = df.merge(self.stats_, on=self.id_col, how="left")
        df[self.target_col] = df[self.target_col] / df["max_"]
        return df.drop(columns=["max_"])

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Merge stats back on
        df = df.merge(self.stats_, on=self.id_col, how="left")
        # Only invert the target column
        df[self.target_col] = (
            df[self.target_col] * df["max_"]
        )
        return df.drop(columns=["max_"])
    

class LocalRobustScaler(BaseTargetTransform):
    """Scales each series by dividing by a certain quantile (default 90th percentile)
    based on a training cutoff defined by the CV config."""

    def __init__(self, cv_cfg: CrossValidationConfig, freq: Frequency, quantile: float = 0.9):
        self.cv_cfg = cv_cfg
        self.freq = freq
        self.quantile = quantile
        self.stats_: pd.DataFrame = None

    def _nonzero_quantile(self, x: pd.Series) -> float:
        """Calculate quantile for a series. If the quantile is zero, return the minimum non-zero value instead."""
        perc = x.quantile(self.quantile)
        nz = x[x > 0]
        min_nz = nz.min() if not nz.empty else 0
        return max(perc, min_nz)


    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # determine cutoff based on frequency and CV windows
        cutoff = self._calculate_cutoff(df)
        df_train: pd.DataFrame = df[df[self.time_col] <= cutoff]

        # compute quantile per series
        self.stats_ = df_train.groupby(self.id_col)[self.target_col].agg(
            quantile=self._nonzero_quantile
        ).reset_index()

        # merge and scale
        df = df.merge(self.stats_, on=self.id_col, how="left")
        df[self.target_col] = df[self.target_col] / df["quantile"]
        return df.drop(columns=["quantile"])

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # merge stats back on
        df = df.merge(self.stats_, on=self.id_col, how="left")
        # invert scaling
        df[self.target_col] = df[self.target_col] * df["quantile"]
        return df.drop(columns=["quantile"])
    
