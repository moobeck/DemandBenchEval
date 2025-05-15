from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class InputColumnConfig:
    """
    A dataclass to store the names of the columns used in the input DataFrame.
    """

    date: str = "dateID"
    dp_index: str = "bdID"
    sku_index: str = "skuID"
    target: str = "target"
