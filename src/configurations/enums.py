from enum import Enum, auto


class Framework(Enum):
    STATS = "STATS"
    ML = "ML"
    NEURAL = "NEURAL"
    FM = "FM"


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
    DEEPAR = "depar"
    BITCN = "bitcn"
    TABPFN = "tabpfn"
    TOTO = "toto"
    QUANTILE_AR = "quantile_ar"
    QUANTILE_LGBM = "quantile_lgbm"
    QUANTILE_CATBOOST = "quantile_catboost"
    QUANTILE_RF = "quantile_rf"


class MetricName(Enum):
    MASE = "mase"
    MSSE = "msse"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    SCALED_MQLOSS = "scaled_mqloss"


class DatasetName(Enum):
    M5 = "m5"
    FAVORITA = "favorita"
    ROHLIK = "rohlik"
    ROSSMANN = "rossmann"
    BAKERY = "bakery"
    YAZ = "yaz"
    PHARMACY = "pharmacy"
    HOTEL = "hoteldemand"
    ONLINERETAIL = "onlineretail"


class Frequency(Enum):
    DAILY = "Daily"
    WEEKLY = "Weekly"

    @staticmethod
    def get_alias(freq: "Frequency", context: str):

        CONTEXT_ALIASES = {
            "pandas": {
                Frequency.DAILY: "D",
                Frequency.WEEKLY: "W-MON",
            },
            "nixtla": {
                Frequency.DAILY: "D",
                Frequency.WEEKLY: "W-MON",
            },
            "demandbench": {
                Frequency.DAILY: "daily",
                Frequency.WEEKLY: "weekly",
            },
        }

        if context not in CONTEXT_ALIASES:
            raise ValueError(f"Unknown context: {context}")

        # Use value-based lookup to handle enum instance issues
        context_map = CONTEXT_ALIASES[context]
        for freq_key, alias in context_map.items():
            if freq.value == freq_key.value:  # Compare enum values instead of objects
                return alias

        raise ValueError(f"No alias defined for {freq} in context '{context}'")

    @staticmethod
    def get_season_length(freq: "Frequency") -> int:

        SEASON_LENGTHS = {
            Frequency.DAILY: 7,
            Frequency.WEEKLY: 52,
        }
        if freq not in SEASON_LENGTHS:
            raise ValueError(f"Unsupported frequency: {freq}")
        return SEASON_LENGTHS[freq]


class TimeInSeconds(Enum):
    DAILY = 86400
    WEEKLY = 604800


class TargetScalerType(Enum):
    LOCAL_STANDARD = "local_standard"
    LOCAL_MAX = "local_max"
    LOCAL_ROBUST = "local_robust"
