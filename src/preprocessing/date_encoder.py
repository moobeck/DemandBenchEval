import pandas as pd
import numpy as np
from src.configurations.enums import Frequency
from typing import Callable


class DateEncoder:

    @staticmethod
    def weekday_sin(dates: pd.Series) -> pd.Series:
        """
        Returns the sine encoding of the weekday for the given dates.

        Parameters:
        - dates (pd.Series): Series of datetime objects.

        Returns:
        - pd.Series: Sine encoded weekday feature.
        """
        weekdays = dates.dt.dayofweek
        return np.sin(2 * np.pi * weekdays / 7)

    @staticmethod
    def weekday_cos(dates: pd.Series) -> pd.Series:
        """
        Returns the cosine encoding of the weekday for the given dates.

        Parameters:
        - dates (pd.Series): Series of datetime objects.

        Returns:
        - pd.Series: Cosine encoded weekday feature.
        """
        weekdays = dates.dt.dayofweek
        return np.cos(2 * np.pi * weekdays / 7)

    @staticmethod
    def month_sin(dates: pd.Series) -> pd.Series:
        """
        Returns the sine encoding of the month for the given dates.

        Parameters:
        - dates (pd.Series): Series of datetime objects.

        Returns:
        - pd.Series: Sine encoded month feature.
        """
        months = dates.dt.month
        return np.sin(2 * np.pi * months / 12)

    @staticmethod
    def month_cos(dates: pd.Series) -> pd.Series:
        """
        Returns the cosine encoding of the month for the given dates.

        Parameters:
        - dates (pd.Series): Series of datetime objects.

        Returns:
        - pd.Series: Cosine encoded month feature.
        """
        months = dates.dt.month
        return np.cos(2 * np.pi * months / 12)

    def __init__(self, freq: Frequency):
        """
        Initializes the DateEncoder with a specific frequency.

        Parameters:
        - freq (Frequency): The frequency of the time series data.
        """
        self.freq = freq
        self.out_columns = []

    def get_encoders(self) -> list[Callable[[pd.Series], pd.Series]]:
        """
        Returns a list of encoding functions based on the frequency.
        Always includes month encodings.
        If frequency is daily, also includes weekday encodings.
        """
        encoders = []
        self.out_columns = []
        if self.freq == Frequency.DAILY:
            encoders += [
                self.weekday_sin,
                self.weekday_cos,
            ]
            self.out_columns += ["weekday_sin", "weekday_cos"]
        encoders += [
            self.month_sin,
            self.month_cos,
        ]
        self.out_columns += ["month_sin", "month_cos"]
        return encoders
