from category_encoders import CatBoostEncoder
import pandas as pd

from src.configurations.cross_validation import CrossValidationConfig
from src.configurations.enums import Frequency
from src.configurations.forecast_column import ForecastColumnConfig


class CategoryEncoder:

    def __init__(
        self,
        cv_cfg: CrossValidationConfig,
        freq: Frequency,
        forecast_columns: ForecastColumnConfig,
    ):
        """
        Initializes the CategoryEncoder with the specified columns.

        Parameters:
        - columns (list[str]): List of column names to be encoded.
        """

        self._cv_cfg = cv_cfg
        self._freq = freq
        self._forecast_columns = forecast_columns
    
        self._encoder = CatBoostEncoder(
            cols=self._forecast_columns.categorical, return_df=False
        )
        self.out_columns = [
            f"{col}_encoded" for col in self._forecast_columns.categorical
        ]


    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes the categorical features in the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the features to encode.

        Returns:
        - pd.DataFrame: DataFrame with encoded features.
        """

        if self._freq == Frequency.DAILY:
            offset = pd.Timedelta(days=self._cv_cfg.cv_windows * self._cv_cfg.step_size)
        elif self._freq == Frequency.WEEKLY:
            offset = pd.Timedelta(
                weeks=self._cv_cfg.cv_windows * self._cv_cfg.step_size
            )
        else:
            raise ValueError(f"Unsupported frequency: {self._freq}")

        cutoff = df[self._forecast_columns.date].max() - offset
        df_train = df[df[self._forecast_columns.date] <= cutoff]

        # Fit the encoder on the training data
        self._encoder.fit(
            df_train[self._forecast_columns.categorical],
            df_train[self._forecast_columns.target],
        )

        # Transform the entire DataFrame
        df[self.out_columns] = self._encoder.transform(
            df[self._forecast_columns.categorical]
        )
        return df
