from dataclasses import dataclass


@dataclass(frozen=True)
class QuantileConfig:
    min: int = 1
    max: int = 99
    step: int = 1
