from typing import List, Optional
from src.constants.quantile_values import DEFAULT_QUANTILE_VALUES
from src.configurations.forecasting.quantile import (
    QuantileConfig,
)


class QuantileUtils:

    @staticmethod
    def quantiles_to_level(quantiles: Optional[List[float]]) -> Optional[List[int]]:
        """
        Convert quantiles to percentage levels.

        Args:
            quantiles (Optional[List[float]]): List of quantiles to convert.
        """
        if quantiles is None or len(quantiles) < 2:
            return None

        levels = []
        for i in range(len(quantiles)):
            lb = quantiles[i]
            ub = quantiles[-(i + 1)]
            level = (ub - lb) * 100

            if level > 0:
                levels.append(int(round(level)))
            else:
                break

        return levels

    @staticmethod
    def create_quantiles(quantile_config: QuantileConfig) -> List[float]:
        """Create a list of quantiles based on the provided configuration."""

        return [float(q) for q in quantile_config.values]
