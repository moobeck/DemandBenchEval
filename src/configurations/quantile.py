from typing import Dict, Any, List
from neuralforecast.losses.pytorch import MQLoss
from .mixture import quantiles_to_level

def create_quantile_levels(quantile_config: Dict[str, Any]) -> List[float]:
    """
    Create a list of quantile levels based on the quantile configuration.
    
    Args:
        quantile_config: Dictionary containing 'min', 'max', and 'step' keys
                        to define the quantile range and step size.
    
    Returns:
        List of quantile levels as floats between 0 and 1.
    """
    if not quantile_config:
        # Default quantiles from 1% to 99%
        return [round(x * 0.01, 2) for x in range(1, 100)]
    
    q_min = quantile_config.get("min", 1)
    q_max = quantile_config.get("max", 99)
    q_step = quantile_config.get("step", 1)
    
    return [round(x * 0.01, 2) for x in range(q_min, q_max + 1, q_step)]



class QuantileLossFactory:
    """
    Factory class to create quantile-specific loss functions based on configuration.
    """
    
    @staticmethod
    def create_loss(quantile_config: Dict[str, Any]) -> MQLoss:
        """
        Create a quantile-specific loss function based on the quantile_config.
        
        Args:
            quantile_config: Dictionary containing quantile configuration
            
        Returns:
            An instance of MQLoss configured with the specified quantiles
        """
        quantiles = create_quantile_levels(quantile_config)
        levels = quantiles_to_level(quantiles)
        
        return MQLoss(level=levels)
