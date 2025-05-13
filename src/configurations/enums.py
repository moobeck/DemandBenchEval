from enum import Enum, auto


class Framework(Enum):
    STATS = auto()
    ML = auto()
    NEURAL = auto()  # future

class ModelName(Enum):
    ARIMA = "arima"
    THETA = "theta"
    ETS = "ets"
    LGBM = "lgbm"
    RF = "rf"

class MetricName(Enum):
    MASE = "mase"
    MSSE = "msse"



