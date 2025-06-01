from enum import Enum, auto


class Framework(Enum):
    STATS = auto()
    ML = auto()
    NEURAL = auto()


class ModelName(Enum):
    ARIMA = "arima"
    THETA = "theta"
    ETS = "ets"
    CES = "ces"
    LGBM = "lgbm"
    CATBOOST = "catboost"
    RF = "rf"
    Transformer = "transformer"
    MLP = "mlp"
    LSTM = "lstm"
    TIMESNET = "timesnet"
    TIMEXER = "timexer"



class MetricName(Enum):
    MASE = "mase"
    MSSE = "msse"
