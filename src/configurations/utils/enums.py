from enum import Enum
from typing import Literal


class Framework(Enum):
    STATS = "STATS"
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
    NBEATS = "nbeats"
    BITCN = "bitcn"
    GRU = "gru"
    TCN = "tcn"
    TFT = "tft"
    PATCHTST = "patchtst"
    XLSTM = "xlstm"
    MOIRAI = "moirai"
    CHRONOS = "chronos"
    TABPFN = "tabpfn"


class MetricName(Enum):
    MASE = "mase"
    MSSE = "msse"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    SMQL = "smql"


PROBABILISTIC_METRICS = {
    MetricName.SMQL,
}


class DatasetName(Enum):
    M5 = "m5"
    FAVORITA = "favorita"
    ROHLIK = "rohlik"
    ROSSMANN = "rossmann"
    BAKERY = "bakery"
    YAZ = "yaz"
    PHARMACY = "pharmacy"
    PHARMACY2 = "pharmacy2"
    HOTEL = "hoteldemand"
    ONLINERETAIL = "onlineretail"
    ONLINERETAIL2 = "onlineretail2"
    FRESHRETAIL50K = "freshretail50k"
    HIERARCHICALSALES = "hierarchicalsales"
    AUSTRALIANRETAIL = "australianretail"
    CARPARTS = "carparts"
    KAGGLEDEMAND = "kaggledemand"
    PRODUCTDEMAND = "productdemand"
    VN1 = "vn1"
    KAGGLERETAIL = "kaggleretail"
    KAGGLEWALMART = "kagglewalmart"
    FOSSIL = "fossil"


class FrequencyType(Enum):
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"

    @staticmethod
    def get_alias(
        freq: "FrequencyType", context: Literal["pandas", "nixtla", "demandbench"]
    ) -> str:

        CONTEXT_ALIASES = {
            "pandas": {
                FrequencyType.DAILY: "D",
                FrequencyType.WEEKLY: "W-MON",
                FrequencyType.MONTHLY: "MS",
            },
            "nixtla": {
                FrequencyType.DAILY: "D",
                FrequencyType.WEEKLY: "W-MON",
                FrequencyType.MONTHLY: "MS",
            },
            "demandbench": {
                FrequencyType.DAILY: "daily",
                FrequencyType.WEEKLY: "weekly",
                FrequencyType.MONTHLY: "monthly",
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
    def get_season_length(freq: "FrequencyType") -> int:

        SEASON_LENGTHS = {
            FrequencyType.DAILY: 7,
            FrequencyType.WEEKLY: 52,
            FrequencyType.MONTHLY: 12,
        }
        if freq not in SEASON_LENGTHS:
            raise ValueError(f"Unsupported frequency: {freq}")
        return SEASON_LENGTHS[freq]


class HierarchyType(Enum):
    PRODUCT_STORE = "product/store"
    STORE = "store"
    PRODUCT = "product"

    @staticmethod
    def get_alias(
        hierarchy: "HierarchyType", context: Literal["demandbench"]
    ) -> str:

        CONTEXT_ALIASES = {
            "demandbench": {
                HierarchyType.PRODUCT_STORE: "product/store",
                HierarchyType.STORE: "store",
                HierarchyType.PRODUCT: "product",
            },
        }

        if context not in CONTEXT_ALIASES:
            raise ValueError(f"Unknown context: {context}")

        # Use value-based lookup to handle enum instance issues
        context_map = CONTEXT_ALIASES[context]
        for hierarchy_key, alias in context_map.items():
            if hierarchy.value == hierarchy_key.value:  # Compare enum values instead of objects
                return alias

        raise ValueError(f"No alias defined for {hierarchy} in context '{context}'")



class TargetScalerType(Enum):
    LOCAL_STANDARD = "local_standard"
    LOCAL_MAX = "local_max"
    LOCAL_ROBUST = "local_robust"


class FileFormat(Enum):
    PARQUET = "parquet"
    FEATHER = "feather"
    CSV = "csv"
