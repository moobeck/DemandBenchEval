from enum import Enum, auto


class Framework(Enum):
    STATS = auto()
    ML = auto()
    NEURAL = auto()


class ModelName(Enum):
    ARIMA = "arima"
    THETA = "theta"
    ETS = "ets"
    LGBM = "lgbm"
    RF = "rf"
    TSMIXER = "tsmixer"
    TIDE = "tide"
    TEST_TIDE = "test_tide"


class MetricName(Enum):
    MASE = "mase"
    MSSE = "msse"
