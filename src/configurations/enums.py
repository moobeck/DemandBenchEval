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
    TRANSFORMER = "transformer"
    MLP = "mlp"
    LSTM = "lstm"
    TIMESNET = "timesnet"
    FEDFORMER = "fedformer"
    TIDE = "tide"
    NHITS = "nhits"


class MetricName(Enum):
    MASE = "mase"
    MSSE = "msse"


class DatasetName(Enum):
    M5 = "m5"
    FAVORITA = "favorita"
    BAKERY = "bakery"
    YAZ = "yaz"
    MAISHAMEDS = "maishameds"


class Frequency(Enum):
    DAILY = "Daily"
    WEEKLY = "Weekly"

    

    @staticmethod
    def get_alias(freq: "Frequency", context: str):

        CONTEXT_ALIASES = {
        'pandas': {
            Frequency.DAILY: 'D',
            Frequency.WEEKLY: 'W-MON',
        },
        'nixtla': {
            Frequency.DAILY: 'D',
            Frequency.WEEKLY: 'W-MON',
        },
        'demandbench': {
            Frequency.DAILY: 'daily',
            Frequency.WEEKLY: 'weekly',
        },
    }

        if context not in CONTEXT_ALIASES:
            raise ValueError(f"Unknown context: {context}")
        try:
            return CONTEXT_ALIASES[context][freq]
        except KeyError:
            raise ValueError(f"No alias defined for {freq} in context '{context}'")

