from typing import Any, Dict, List, Optional
from src.configurations.forecasting.quantile import QuantileConfig


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
        """
        Create a list of quantiles based on the quantile configuration.

        Args:
            quantile_config: Dictionary containing 'min', 'max', and 'step' keys
                            to define the quantile range and step size.

        Returns:
            List of quantile levels as floats between 0 and 1.
        """
        if not quantile_config:
            # Default quantiles from 1% to 99%
            return [round(x * 0.01, 2) for x in range(1, 100)]

        q_min = quantile_config.min
        q_max = quantile_config.max
        q_step = quantile_config.step

        return [round(x * 0.01, 2) for x in range(q_min, q_max + 1, q_step)]
