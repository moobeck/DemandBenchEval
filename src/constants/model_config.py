"""Default configuration values for forecasting frameworks."""

from typing import Dict, Any

from src.configurations.utils.enums import Framework

DEFAULT_MODEL_FRAMEWORK_CONFIG: Dict[Framework, Dict[str, Any]] = {
    Framework.NEURAL: {
        "gpus": 1,
        "cpus": 1,
        "num_samples": 50,
    },
    Framework.FM: {
        "num_samples": 100,
    },
}
