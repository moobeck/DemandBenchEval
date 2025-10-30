from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import pandas as pd
import torch
import utilsforecast.processing as ufp
from gluonts.dataset.pandas import PandasDataset
from gluonts.model.forecast import Forecast
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.forecasting.models.foundation.utils.forecaster import (
    Forecaster,
    QuantileConverter,
)

from src.utils.cross_validation import get_offset


def fix_freq(freq: str) -> str:
    # see https://github.com/awslabs/gluonts/pull/2462/files
    replacer = {"MS": "M", "ME": "M"}
    return replacer.get(freq, freq)


def maybe_convert_col_to_float32(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if df[col_name].dtype != "float32":
        df = df.copy()
        df[col_name] = df[col_name].astype("float32")
    return df


class GluonTSForecaster(Forecaster):
    def __init__(
        self,
        repo_id: str,
        filename: str,
        alias: str,
        num_samples: int,
        futr_exog_list: list[str] | None = None,
        hist_exog_list: list[str] | None = None,
        stat_exog_list: list[str] | None = None,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.alias = alias
        self.num_samples = num_samples

        # Exogenous feature configuration
        self.futr_exog_list = futr_exog_list if futr_exog_list is not None else []
        self.hist_exog_list = hist_exog_list if hist_exog_list is not None else []
        self.stat_exog_list = stat_exog_list if stat_exog_list is not None else []

        # Dimensions of target variables
        self._target_dim = 1  # Currently only univariate target supported

        # Dimensions of exogenous features
        self._feat_dynamic_real_dim = len(self.futr_exog_list)
        self._past_feat_dynamic_real_dim = len(self.hist_exog_list)
        self._static_feat_dim = len(self.stat_exog_list)

    @property
    def checkpoint_path(self) -> str:
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
        )

    @property
    def map_location(self) -> str:
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        return map_location

    def load(self) -> Any:
        return torch.load(
            self.checkpoint_path,
            map_location=self.map_location,
        )

    @contextmanager
    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        raise NotImplementedError

    def gluonts_instance_fcst_to_df(
        self,
        fcst: Forecast,
        freq: str,
        model_name: str,
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        point_forecast = fcst.median
        h = len(point_forecast)
        dates = pd.date_range(
            fcst.start_date.to_timestamp(),
            freq=freq,
            periods=h,
        )
        fcst_df = pd.DataFrame(
            {
                "ds": dates,
                "unique_id": fcst.item_id,
                model_name: point_forecast,
            }
        )
        if quantiles is not None:
            for q in quantiles:
                fcst_df = ufp.assign_columns(
                    fcst_df,
                    f"{model_name}-q-{int(q * 100)}",
                    fcst.quantile(q),
                )
        return fcst_df

    def gluonts_fcsts_to_df(
        self,
        fcsts: Iterable[Forecast],
        freq: str,
        model_name: str,
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        df = []
        for fcst in tqdm(fcsts):
            fcst_df = self.gluonts_instance_fcst_to_df(
                fcst=fcst,
                freq=freq,
                model_name=model_name,
                quantiles=quantiles,
            )
            df.append(fcst_df)
        return pd.concat(df).reset_index(drop=True)

    def cross_validation(
        self,
        df: pd.DataFrame,
        n_windows: int = 1,
        horizon: int = 7,
        step_size: int = 1,
        quantiles: list[float] | None = None,
        level: list[int | float] | None = None,
        freq: str | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:

        static_df = df[[id_col] + self.stat_exog_list].drop_duplicates()
        df = df.drop(columns=self.stat_exog_list)
        df = maybe_convert_col_to_float32(df, id_col)

        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)

        gluonts_dataset = PandasDataset.from_long_dataframe(
            df.copy(deep=False),
            target=target_col,
            item_id=id_col,
            timestamp=time_col,
            freq=fix_freq(freq),
            feat_dynamic_real=self.futr_exog_list,
            past_feat_dynamic_real=self.hist_exog_list,
            static_features=(
                static_df.set_index(id_col) if static_df is not None else None
            ),
        )

        offset = get_offset(n_windows, step_size, horizon)
        _, test_template = split(gluonts_dataset, offset=-offset)

        test_data = test_template.generate_instances(
            prediction_length=horizon,
            windows=n_windows,
            distance=step_size,
        )

        with self.get_predictor(prediction_length=horizon) as predictor:
            fcsts = predictor.predict(
                test_data.input,
                num_samples=self.num_samples,
            )

        fcst_df = self.gluonts_fcsts_to_df(
            fcsts,
            freq=freq,
            model_name=self.alias,
            quantiles=qc.quantiles,
        )
        if qc.quantiles is not None:
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )

        return fcst_df

    def forecast(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame | None,
        horizon: int,
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
                DataFrame containing the time series to forecast.
            static_df (pd.DataFrame | None):
                DataFrame containing static features for each time series.
                Must include the `id_col`. If None, no static features are used.
            horizon (int):
                The forecast horizon (number of time steps to predict).
            freq (str | None):
                Frequency of the time series (e.g., 'D' for daily).
                If None, the frequency is inferred from the data.
            level (list[int | float] | None):
                List of confidence levels for prediction intervals (e.g., [80, 95]).
            quantiles (list[float] | None):
                List of quantiles to forecast (e.g., [0.1, 0.5, 0.9]).
            id_col (str):
                Column name for the unique time series identifier.
            time_col (str):
                Column name for the time index.
            target_col (str):
                Column name for the target variable to forecast.
        Returns:
            pd.DataFrame:
                DataFrame containing the forecasts in long format
        """

        return self.cross_validation(
            df=df,
            static_df=static_df,
            n_windows=1,
            horizon=horizon,
            step_size=1,
            quantiles=quantiles,
            level=level,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
