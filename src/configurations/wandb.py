from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class WandbConfig:
    """
    A dataclass to store the configuration settings for Weights & Biases integration.
    """
    api_key: Optional[str] = None
    entity: Optional[str] = None
    project: str = "forecast-bench"
    tags: List[str] = field(default_factory=list)
