from category_encoders import CatBoostEncoder
import pandas as pd

from src.configurations.evaluation.cross_validation import CrossValidationConfig
from src.configurations.utils.enums import Frequency
from src.configurations.data.forecast_column import ForecastColumnConfig


class CategoryEncoder:

    def __init__(
        self,
        cv_cfg: CrossValidationConfig,
        freq: Frequency,
        forecast_columns: ForecastColumnConfig,
        horizon: int,
    ):
        """
        Initializes the CategoryEncoder with the specified columns.

        Parameters:
        - columns (list[str]): List of column names to be encoded.
        """

        self._cv_cfg = cv_cfg
        self._freq = freq
        self._forecast_columns = forecast_columns
        self._horizon = horizon

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

        cutoff = self._cv_cfg.get_cutoff_date(
            max_date=df[self._forecast_columns.date].max(),
            freq=self._freq,
            split="test",
            horizon=self._horizon,
        )

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
