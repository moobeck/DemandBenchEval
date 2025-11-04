from typing import Optional

from neuralforecast.losses.pytorch import MQLoss

from src.configurations.forecasting.quantile import QuantileConfig
from src.utils.quantile import QuantileUtils


class QuantileLossFactory:
    """
    Factory class to create quantile-specific loss functions based on configuration.
    """

    @staticmethod
    def create_loss(quantile_config: Optional[QuantileConfig]) -> MQLoss:
        """
        Create a quantile-specific loss function based on the quantile_config.

        Args:
            quantile_config: Quantile configuration; when ``None`` the project
                default grid is used.

        Returns:
            An instance of MQLoss configured with the specified quantiles
        """
        quantiles = QuantileUtils.create_quantiles(quantile_config)
        levels = QuantileUtils.quantiles_to_level(quantiles)

        return MQLoss(level=levels)
