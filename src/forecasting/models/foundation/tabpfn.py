import sys
from contextlib import contextmanager

if sys.version_info >= (3, 13):
    raise ImportError("TabPFN requires Python < 3.13")

from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG

import pandas as pd
import torch
from tabpfn_client import set_access_token
from tabpfn_time_series import (
    TabPFNMode,
    TabPFNTimeSeriesPredictor,
    TimeSeriesDataFrame,
)

from src.forecasting.models.foundation.utils.forecaster import (
    Forecaster,
    maybe_convert_col_to_datetime,
    QuantileConverter,
)
from tqdm import tqdm
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)
from tabpfn_time_series.data_preparation import generate_test_X
from enum import Enum
from neuralforecast.losses.pytorch import quantiles_to_outputs


class TabPFNColumns(Enum):
    TARGET = "target"
    TIMESTAMP = "timestamp"
    ID = "item_id"


class TabPFN(Forecaster):
    """
    TabPFN is a zero-shot time series forecasting model that frames univariate
    forecasting as a tabular regression problem using TabPFNv2. It supports both
    point and probabilistic forecasts, and can incorporate exogenous variables via
    feature engineering. See the
    [official repo](https://github.com/PriorLabs/tabpfn-time-series) for more details.
    """

    def __init__(
        self,
        context_length: int = 4096,
        mode: TabPFNMode | None = None,
        api_key: str | None = None,
        alias: str = "TabPFN",
        futr_exog_list: list[str] | None = None,
        hist_exog_list: list[str] | None = None,
        stat_exog_list: list[str] | None = None,
    ):
        """
        Args:
            context_length (int, optional): Maximum context length (input window size)
                for the model. Defaults to 4096. Controls how much history is used for
                each forecast.
            mode (TabPFNMode, optional): Inference mode for TabPFN. If None, uses LOCAL
                (`"tabpfn-local"`) if a GPU is available, otherwise CLIENT (cloud
                inference via `"tabpfn-client"`). See
                [TabPFN-TS docs](https://github.com/PriorLabs/tabpfn-time-series/
                blob/3cd61ad556466de837edd1c6036744176145c024/tabpfn_time_series/
                predictor.py#L11) for available modes.
            api_key (str, optional): API key for tabpfn-client cloud inference. Required
                if using CLIENT mode and not already set in the environment.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "TabPFN".
            futr_exog_list (list[str], optional): List of future exogenous
                variables to include in the model. Defaults to None.
            hist_exog_list (list[str], optional): List of historical exogenous
                variables to include in the model. Defaults to None.
            stat_exog_list (list[str], optional): List of static exogenous
                variables to include in the model. Defaults to None.

        Notes:
            **Academic Reference:**

            - Paper: [From Tables to Time: How TabPFN-v2 Outperforms
            Specialized Time Series Forecasting Models](https://arxiv.org/abs/2501.02945)

            **Resources:**

            - GitHub: [PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series)

            **Technical Details:**

            - For LOCAL mode, a CUDA-capable GPU is recommended for best performance.
            - The model is only available for Python < 3.13.
        """

        self.context_length = context_length
        if mode is None:
            mode = TabPFNMode.LOCAL if torch.cuda.is_available() else TabPFNMode.CLIENT
        if mode == TabPFNMode.CLIENT and api_key is not None:
            set_access_token(api_key)
        self.mode = mode
        self.alias = alias
        self.futr_exog_list = futr_exog_list if futr_exog_list is not None else []
        self.hist_exog_list = hist_exog_list if hist_exog_list is not None else []
        self.stat_exog_list = stat_exog_list if stat_exog_list is not None else []

    @contextmanager
    def _get_model(self) -> TabPFNTimeSeriesPredictor:

        model = TabPFNTimeSeriesPredictor(tabpfn_mode=self.mode)
        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()

    def _prepare_df_for_forecast(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Prepare input DataFrame for forecasting by renaming columns and
        handling exogenous variables.
        """

        static_df = (
            df[self.stat_exog_list + [id_col]].drop_duplicates().set_index(id_col)
        )
        df = df.copy()
        df.drop(columns=self.stat_exog_list + self.hist_exog_list, inplace=True)
        df = df.rename(
            {
                target_col: TabPFNColumns.TARGET.value,
                time_col: TabPFNColumns.TIMESTAMP.value,
                id_col: TabPFNColumns.ID.value,
            },
            axis=1,
        )

        tsdf = TimeSeriesDataFrame(df, static_features=static_df)

        return tsdf

    def _post_process_forecast_df(
        self,
        fcst_df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Post-process forecast DataFrame by renaming columns back to original names."""

        fcst_df = fcst_df.reset_index()
        fcst_df = fcst_df.rename(
            {
                TabPFNColumns.TARGET.value: self.alias,
                TabPFNColumns.TIMESTAMP.value: time_col,
                TabPFNColumns.ID.value: id_col,
            },
            axis=1,
        )

        return pd.DataFrame(fcst_df)

    def _forecast(
        self,
        model: TabPFNTimeSeriesPredictor,
        df: pd.DataFrame,
        h: int,
        future_df: pd.DataFrame,
        quantiles: list[float] | None,
        freq: str | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """handles distinction between quantiles and no quantiles"""

        tsdf = self._prepare_df_for_forecast(
            df, id_col=id_col, time_col=time_col, target_col=target_col
        )

        if self.context_length > 0:
            tsdf = tsdf.slice_by_timestep(-self.context_length, None)
        future_tsdf = generate_test_X(tsdf, h)

        future_tsdf_exog = self._prepare_df_for_forecast(
            future_df, id_col=id_col, time_col=time_col, target_col=target_col
        )

        future_tsdf = future_tsdf.merge(
            future_tsdf_exog,
            on=[TabPFNColumns.ID.value, TabPFNColumns.TIMESTAMP.value],
            how="left",
        )

        fcst_df = model.predict(tsdf, future_tsdf)

        fcst_df = self._post_process_forecast_df(
            fcst_df, id_col=id_col, time_col=time_col, target_col=target_col
        )

        return fcst_df

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        future_df: pd.DataFrame | None = None,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the model.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

            h (int):
                Forecast horizon specifying how many future steps to predict.
            future_df (pd.DataFrame, optional):
                DataFrame containing future exogenous variables for the
                forecast period. It should include the same ID and time
                columns as `df`, along with any exogenous feature columns.
                If not provided, only endogenous features will be used.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.org/
                pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for
                valid values. If not provided, the frequency will be inferred
                from the data.
            level (list[int | float], optional):
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]). If provided, the returned
                DataFrame will include lower and upper interval columns for
                each specified level.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0
                and 1. Should not be used simultaneously with `level`. When
                provided, the output DataFrame will contain additional columns
                named in the format "model-q-{percentile}", where {percentile}
                = 100 × quantile value.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """

        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)

        if qc.quantiles and qc.quantiles != TABPFN_TS_DEFAULT_QUANTILE_CONFIG:
            raise ValueError(
                f"TabPFN only supports the default quantiles: "
                f"{TABPFN_TS_DEFAULT_QUANTILE_CONFIG}, but got {qc.quantiles}."
            )

        with self._get_model() as model:
            fcst_df = self._forecast(
                model,
                df,
                h,
                future_df=future_df,
                quantiles=qc.quantiles,
                freq=freq,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
        if qc.quantiles is not None:
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df

    def _rename_forecast_columns(
        self,
        df: pd.DataFrame,
        quantiles: list[float] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Rename forecast columns in the output DataFrame based on quantiles.

        Args:
            df (pd.DataFrame): DataFrame containing forecast results.
            quantiles (list[float] | None): List of quantiles used for forecasting.
        Returns:
            pd.DataFrame: DataFrame with renamed forecast columns.
        """

        features = self.stat_exog_list + self.hist_exog_list + self.futr_exog_list
        base_columns = ["cutoff", id_col, time_col, target_col]

        quantile_cols = [
            col
            for col in df.columns
            if col not in base_columns + features + [self.alias]
        ]

        quantile_cols, level_cols = quantiles_to_outputs(
            [float(q) for q in quantile_cols]
        )
        level_cols = [f"{self.alias}{level}" for level in level_cols]

        rename_mapping = dict(zip(quantile_cols, level_cols))

        df = df.rename(
            mapper=rename_mapping,
            axis=1,
        )

        return df

    def cross_validation(
        self,
        df: pd.DataFrame,
        n_windows: int = 1,
        horizon: int = 7,
        step_size: int = 1,
        quantiles: list[float] | None = None,
        freq: str | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Perform cross-validation on the time series data using rolling windows.

        This method evaluates the model's forecasting performance by generating
        predictions over multiple rolling windows. It supports both point and
        probabilistic forecasts. The input DataFrame can contain one or multiple
        time series in stacked (long) format.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series for cross-validation. It must include the following columns:
                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.
            n_windows (int, optional):
                Number of rolling windows to use for cross-validation. Defaults to 1.
            horizon (int, optional):
                Forecast horizon specifying how many future steps to predict in each window. Defaults to 7.
            step_size (int, optional):
                Step size between the start of each rolling window. Defaults to 1.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0 and 1.
           freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for monthly). See pandas frequency strings for more options.
            id_col (str, optional):
                Name of the column containing unique time series identifiers. Defaults to "unique_id".
            time_col (str, optional):
                Name of the column containing timestamps or periods. Defaults to "ds".
            target_col (str, optional):
                Name of the column containing the target variable. Defaults to "y".
        Returns:
            pd.DataFrame:
                DataFrame containing cross-validation forecast results. Includes:
                    - point forecasts for each timestamp and series.
                    - quantile forecasts if `quantiles` is specified.
                The output retains the same unique identifiers as the input DataFrame.
        """

        df = maybe_convert_col_to_datetime(df, time_col)

        results = []

        sort_idxs = maybe_compute_sort_indices(df, id_col, time_col)

        if sort_idxs is not None:
            df = take_rows(df, sort_idxs)

        splits = backtest_splits(
            df,
            n_windows=n_windows,
            h=horizon,
            step_size=step_size,
            id_col=id_col,
            time_col=time_col,
            freq=freq,
        )

        for _, (cutoffs, train, valid) in tqdm(enumerate(splits)):

            forecast_df = self.forecast(
                df=train,
                h=horizon,
                future_df=valid.drop(columns=[target_col]),
                freq=freq,
                quantiles=quantiles,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )

            forecast_df = join(
                forecast_df,
                cutoffs,
                on=id_col,
                how="left",
            )

            result = join(
                valid,
                forecast_df,
                on=[id_col, time_col],
            )

            if result.shape[0] < valid.shape[0]:
                raise ValueError(
                    "Cross validation result produced less results than expected. "
                    "Please verify that the frequency parameter (freq) "
                    "matches your series' "
                    "and that there aren't any missing periods."
                )
            results.append(result)

        out_df = vertical_concat(results)
        out_df = self._transform_forecast_results(
            quantiles, id_col, time_col, target_col, out_df
        )

        return out_df

    def _transform_forecast_results(
        self, quantiles, id_col, time_col, target_col, out_df
    ):
        out_df = drop_index_if_pandas(out_df)

        columns_to_drop = set(
            self.stat_exog_list + self.hist_exog_list + self.futr_exog_list
        ) & set(out_df.columns)
        if columns_to_drop:
            out_df = out_df.drop(columns=list(columns_to_drop))

        out_df = self._rename_forecast_columns(
            out_df,
            quantiles=quantiles,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

        return out_df
