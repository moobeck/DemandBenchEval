from dataclasses import dataclass, field
from src.constants.quantile_values import DEFAULT_QUANTILE_VALUES
from typing import Tuple


@dataclass(frozen=True)
class QuantileConfig:
    """Immutable configuration describing the quantile grid used across the project."""

    values: Tuple[float, ...] = field(default=DEFAULT_QUANTILE_VALUES)

    def __post_init__(self) -> None:
        normalized = tuple(float(q) for q in self.values)

        if len(normalized) < 2:
            raise ValueError("At least two quantile values are required.")

        if any(q <= 0 or q >= 1 for q in normalized):
            raise ValueError("Quantile values must lie strictly between 0 and 1.")

        if list(normalized) != sorted(normalized):
            raise ValueError(
                "Quantile values must be provided in strictly increasing order."
            )

        if len(set(normalized)) != len(normalized):
            raise ValueError("Quantile values must be unique.")

        object.__setattr__(self, "values", normalized)


DEFAULT_QUANTILE_CONFIG = QuantileConfig()
