from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
import utilsforecast.processing as ufp
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.model.forecast import Forecast
from gluonts.torch.model.predictor import PyTorchPredictor
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
        context_length: int | None = None,
        futr_exog_list: list[str] | None = None,
        hist_exog_list: list[str] | None = None,
        stat_exog_list: list[str] | None = None,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.alias = alias
        self.num_samples = num_samples
        self.context_length = context_length

        self.futr_exog_list = futr_exog_list if futr_exog_list is not None else []
        self.hist_exog_list = hist_exog_list if hist_exog_list is not None else []
        self.stat_exog_list = stat_exog_list if stat_exog_list is not None else []

        self._target_dim = 1
        self._feat_dynamic_real_dim = len(self.futr_exog_list)
        self._past_feat_dynamic_real_dim = len(self.hist_exog_list)
        self._static_feat_dim = len(self.stat_exog_list)

    @property
    def checkpoint_path(self) -> str:
        return hf_hub_download(repo_id=self.repo_id, filename=self.filename)

    @property
    def map_location(self) -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def load(self) -> Any:
        return torch.load(self.checkpoint_path, map_location=self.map_location)

    @contextmanager
    def get_predictor(self, prediction_length: int, freq: str) -> PyTorchPredictor:
        raise NotImplementedError

    @staticmethod
    def _freq_offset(freq: str) -> pd.tseries.offsets.BaseOffset:
        return pd.tseries.frequencies.to_offset(fix_freq(freq))

    @staticmethod
    def _cutoff_from_fcst_start(fcst: Forecast, freq: str) -> pd.Timestamp:
        # forecast start == first predicted timestamp
        start = fcst.start_date.to_timestamp()
        return start - GluonTSForecaster._freq_offset(freq)

    @staticmethod
    def _extract_future_target(label_entry: Any) -> np.ndarray:
        # GluonTS instance labels are typically dict-like DataEntry
        if isinstance(label_entry, dict):
            for k in ("future_target", "target", "y"):
                if k in label_entry:
                    return np.asarray(label_entry[k], dtype="float32")
        raise KeyError(
            f"Could not find future target in label entry. "
            f"Keys={list(label_entry.keys()) if isinstance(label_entry, dict) else type(label_entry)}"
        )

    @staticmethod
    def _instance_forecast_to_pred_df(
        fcst: Forecast,
        freq: str,
        model_name: str,
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        point_forecast = fcst.median
        h = len(point_forecast)

        start = fcst.start_date.to_timestamp()
        dates = pd.date_range(start, freq=freq, periods=h)
        cutoff = GluonTSForecaster._cutoff_from_fcst_start(fcst, freq)

        fcst_df = pd.DataFrame(
            {
                "ds": dates,
                "unique_id": fcst.item_id,
                "cutoff": cutoff,
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

    @staticmethod
    def _instance_label_to_truth_df(
        fcst: Forecast,
        label_entry: Any,
        freq: str,
        target_col: str,
    ) -> pd.DataFrame:
        y = GluonTSForecaster._extract_future_target(label_entry)
        h = len(y)

        start = fcst.start_date.to_timestamp()
        dates = pd.date_range(start, freq=freq, periods=h)
        cutoff = GluonTSForecaster._cutoff_from_fcst_start(fcst, freq)

        return pd.DataFrame(
            {
                "ds": dates,
                "unique_id": fcst.item_id,
                "cutoff": cutoff,
                target_col: y,
            }
        )


    @staticmethod
    def _instance_forecast_to_pred_df_no_cutoff(
        fcst: Forecast,
        freq: str,
        model_name: str,
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        point_forecast = fcst.median
        h = len(point_forecast)

        start = fcst.start_date.to_timestamp()
        dates = pd.date_range(start, freq=freq, periods=h)

        out = pd.DataFrame(
            {
                "ds": dates,
                "unique_id": fcst.item_id,
                model_name: point_forecast,
            }
        )

        if quantiles is not None:
            for q in quantiles:
                out = ufp.assign_columns(
                    out,
                    f"{model_name}-q-{int(q * 100)}",
                    fcst.quantile(q),
                )

        return out



    @staticmethod
    def _rename_forecast_base_columns(
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        return df.rename(
            mapper={"ds": time_col, "unique_id": id_col},
            axis=1,
        )

    def cross_validation(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame | None = None,
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
        # Ensure timestamp dtype
        if not np.issubdtype(df[time_col].dtype, np.datetime64):
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col])

        # Build / use static features
        if static_df is None and self.stat_exog_list:
            static_df = df[[id_col] + self.stat_exog_list].drop_duplicates()
        if self.stat_exog_list:
            df = df.drop(columns=self.stat_exog_list)

        # IMPORTANT: cast target, NOT id
        df = maybe_convert_col_to_float32(df, target_col)

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
            static_features=(static_df.set_index(id_col) if static_df is not None else None),
        )

        offset = get_offset(n_windows, step_size, horizon)
        _, test_template = split(gluonts_dataset, offset=-offset)

        test_data = test_template.generate_instances(
            prediction_length=horizon,
            windows=n_windows,
            distance=step_size,
        )

        with self.get_predictor(prediction_length=horizon, freq=freq) as predictor:
            fcst_iter = predictor.predict(test_data.input, num_samples=self.num_samples)

            pred_parts: list[pd.DataFrame] = []
            truth_parts: list[pd.DataFrame] = []

            for fcst, label in tqdm(
                zip(fcst_iter, test_data.label),
                total=None,
                desc=f"{self.alias} CV",
            ):
                pred_parts.append(
                    self._instance_forecast_to_pred_df(
                        fcst=fcst,
                        freq=freq,
                        model_name=self.alias,
                        quantiles=qc.quantiles,
                    )
                )
                truth_parts.append(
                    self._instance_label_to_truth_df(
                        fcst=fcst,
                        label_entry=label,
                        freq=freq,
                        target_col=target_col,
                    )
                )

        pred_df = pd.concat(pred_parts, ignore_index=True)
        truth_df = pd.concat(truth_parts, ignore_index=True)
        
        if qc.quantiles is not None:
            pred_df = qc.convert_quantiles_to_level(pred_df, models=[self.alias])


        # Chronos-like: include cutoff + realized target for the horizon
        out_df = truth_df.merge(
            pred_df,
            on=["unique_id", "ds", "cutoff"],
            how="left",
            validate="one_to_one",
        )

        out_df = self._rename_forecast_base_columns(out_df, id_col=id_col, time_col=time_col)
        return out_df

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
        # Ensure timestamp dtype
        if not np.issubdtype(df[time_col].dtype, np.datetime64):
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col])

        # Static features
        if static_df is None and self.stat_exog_list:
            static_df = df[[id_col] + self.stat_exog_list].drop_duplicates()
        if self.stat_exog_list:
            df = df.drop(columns=self.stat_exog_list)

        # IMPORTANT: cast target, NOT id
        df = maybe_convert_col_to_float32(df, target_col)

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
            static_features=(static_df.set_index(id_col) if static_df is not None else None),
        )

        with self.get_predictor(prediction_length=horizon) as predictor:
            fcst_iter = predictor.predict(gluonts_dataset, num_samples=self.num_samples)

            pred_parts: list[pd.DataFrame] = []
            for fcst in tqdm(fcst_iter, desc=f"{self.alias} forecast"):
                pred_parts.append(
                    self._instance_forecast_to_pred_df_no_cutoff(
                        fcst=fcst,
                        freq=freq,
                        model_name=self.alias,
                        quantiles=qc.quantiles,
                    )
                )

        out_df = pd.concat(pred_parts, ignore_index=True)

        if qc.quantiles is not None:
            out_df = qc.convert_quantiles_to_level(out_df, models=[self.alias])

        out_df = self._rename_forecast_base_columns(out_df, id_col=id_col, time_col=time_col)

        # guarantee no cutoff column
        out_df = out_df.drop(columns=["cutoff"], errors="ignore")

        return out_df

