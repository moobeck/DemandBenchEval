from dataclasses import dataclass


@dataclass(frozen=True)
class CrossValidationConfig:
    """
    A dataclass to store the time series cross-validation configuration.
    """

    cv_windows: int = 5
    step_size: int = 14
    refit: bool = False
