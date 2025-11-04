from dataclasses import dataclass, field

from src.constants import DEFAULT_TARGET_TRANSFORM
from src.configurations.utils.enums import TargetScalerType


@dataclass
class PreprocessingConfig:
    """
    A dataclass to store the preprocessing configuration.
    """

    target_transform: TargetScalerType = field(default=DEFAULT_TARGET_TRANSFORM)


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
