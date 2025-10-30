from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class InputColumnConfig:
    """
    A dataclass to store the names of the columns used in the input DataFrame.
    """

    date: str = "date"
    time_series_index: str = "skuID"
    target: str = "target"
    frequency: str = "frequency"
