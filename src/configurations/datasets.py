from src.configurations.enums import DatasetName
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    """
    A dataclass to store which datasets are used for benchmarking
    """

    names: list[DatasetName] = field(default_factory=list)


