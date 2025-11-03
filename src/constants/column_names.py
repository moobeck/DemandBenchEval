from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastColumnNames:
    """Immutable definition of core column names used across the benchmark."""

    date: str = "date"
    time_series_index: str = "timeSeriesID"
    target: str = "target"
    store_index: str = "storeID"
    product_index: str = "productID"
    cutoff: str = "cutoff"


DEFAULT_FORECAST_COLUMN_NAMES = ForecastColumnNames()
