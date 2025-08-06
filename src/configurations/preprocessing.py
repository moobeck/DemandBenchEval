from dataclasses import dataclass, field
from src.configurations.enums import TargetScalerType


@dataclass
class PreprocessingConfig:
    """
    A dataclass to store the preprocessing configuration.
    """

    target_transform: TargetScalerType = field(default=TargetScalerType.LOCAL_STANDARD)
