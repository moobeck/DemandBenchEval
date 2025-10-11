from dataclasses import dataclass

@dataclass
class SystemConfig:
    """
    A dataclass to store general configuration settings.
    """

    GPU: int = 0
    RANDOM_SEED: int = 42