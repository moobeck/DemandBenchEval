from dataclasses import dataclass, field
from typing import List


@dataclass
class ForecastColumnConfig:
    """
    A dataclass to store the names of the columns used in the output DataFrame.
    """

    date: str = "date"
    sku_index: str = "skuID"
    target: str = "demand"
    exogenous: List[str] = field(default_factory=list)
    static: List[str] = field(default_factory=list)
    cutoff: str = "cutoff"

    @property
    def ts_base_cols(self) -> List[str]:
        """
        Returns the three core columns for time-series: SKU, date, and target.
        """
        return [self.sku_index, self.date, self.target]
