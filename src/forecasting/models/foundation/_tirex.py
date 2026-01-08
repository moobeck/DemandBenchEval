import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tirex import load_model
from tirex.base import PretrainedModel


from src.forecasting.models.foundation.utils.forecaster import (
    Forecaster,
    QuantileConverter,
    maybe_convert_col_to_datetime,
    maybe_compute_sort_indices,
)
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    take_rows,
    vertical_concat,
)

from .utils.dataset import TimeSeriesDataset  


class TiRex(Forecaster):
    """
    TiRex (NX-AI/TiRex) zero-shot forecasting.
    Notes:
      TiRex currently has no covariate/static feature support in this integration.
      Feature args are accepted for interface compatibility but ignored (warn once).
    """

    def __init__(
        self,
        repo_id: str = "NX-AI/TiRex",
        batch_size: int = 16,
        alias: str = "TiRex",
        futr_exog_list: list[str] | None = None,
        hist_exog_list: list[str] | None = None,
        stat_exog_list: list[str] | None = None,
    ):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias

        self.futr_exog_list = futr_exog_list if futr_exog_list is not None else []
        self.hist_exog_list = hist_exog_list if hist_exog_list is not None else []
        self.stat_exog_list = stat_exog_list if stat_exog_list is not None else []

        self._did_warn_features = False

    def _warn_features_unused(self, static_df: pd.DataFrame | None) -> None:
        if self._did_warn_features:
            return
        if self.futr_exog_list or self.hist_exog_list or self.stat_exog_list or static_df is not None:
            warnings.warn(
                "TiRex does not support covariates/static features here. "
                "Feature columns are accepted for interface compatibility but will be ignored.",
                stacklevel=2,
            )
        self._did_warn_features = True

    @staticmethod
    def _ensure_float32(series: pd.Series, name: str) -> pd.Series:
        if series.dtype == object:
            series = pd.to_numeric(series, errors="raise")
        return series.astype("float32", copy=False)

    def _prepare_core_df(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        df = maybe_convert_col_to_datetime(df, time_col)

        sort_idxs = maybe_compute_sort_indices(df, id_col, time_col)
        if sort_idxs is not None:
            df = take_rows(df, sort_idxs)

        df = df.copy(deep=False)
        df[target_col] = self._ensure_float32(df[target_col], target_col)

        # TiRex dataset util expects these canonical names
        return df[[id_col, time_col, target_col]].rename(
            columns={id_col: "unique_id", time_col: "ds", target_col: "y"}
        )

    @contextmanager
    def _get_model(self) -> PretrainedModel:
        if sys.version_info < (3, 11):
            raise RuntimeError("TiRex requires Python >= 3.11")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        prev_no_cuda = os.environ.get("TIREX_NO_CUDA")
        if device == "cpu":
            # disable custom CUDA kernels on CPU
            os.environ["TIREX_NO_CUDA"] = "1"


        model = load_model(self.repo_id, device=device)
        try:
            yield model
        finally:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            # restore env only if we changed it
            if device == "cpu":
                if prev_no_cuda is None:
                    os.environ.pop("TIREX_NO_CUDA", None)
                else:
                    os.environ["TIREX_NO_CUDA"] = prev_no_cuda

    @staticmethod
    def _run_batches(
        model: PretrainedModel,
        dataset: TimeSeriesDataset,
        horizon: int,
        quantiles: list[float] | None,
        desc: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        means: list[np.ndarray] = []
        qs: list[np.ndarray] = []

        with torch.inference_mode():
            if quantiles is None:
                for batch in tqdm(dataset, desc=desc):
                    q_arr, mean_arr = model.forecast(
                        batch,
                        prediction_length=horizon,
                        output_type="numpy",
                    )
                    means.append(np.asarray(mean_arr))
                return np.concatenate(means, axis=0), None

            for batch in tqdm(dataset, desc=desc):
                q_arr, mean_arr = model.forecast(
                    batch,
                    prediction_length=horizon,
                    output_type="numpy",
                )
                qs.append(np.asarray(q_arr))
                means.append(np.asarray(mean_arr))

        return np.concatenate(means, axis=0), np.concatenate(qs, axis=0)

    def _forecast_with_model(
        self,
        model: PretrainedModel,
        core_df: pd.DataFrame,  # columns: unique_id, ds, y
        horizon: int,
        freq: str,
        qc: QuantileConverter,
        id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        dataset = TimeSeriesDataset.from_df(core_df, batch_size=self.batch_size)
        out = dataset.make_future_dataframe(h=horizon, freq=freq)

        mean_np, q_np = self._run_batches(
            model=model,
            dataset=dataset,
            horizon=horizon,
            quantiles=qc.quantiles,
            desc=f"{self.alias} forecast",
        )

        out[self.alias] = mean_np.reshape(-1)

        if qc.quantiles is not None and q_np is not None:
            for i, q in enumerate(qc.quantiles):
                out[f"{self.alias}-q-{int(q * 100)}"] = q_np[..., i].reshape(-1)
            out = qc.convert_quantiles_to_level(out, models=[self.alias])

        # return in caller naming
        out = out.rename(columns={"unique_id": id_col, "ds": time_col})
        out = out.drop(columns=["cutoff"], errors="ignore")
        return out

    def forecast(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame | None = None,
        horizon: int = 7,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        self._warn_features_unused(static_df)

        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)

        core_df = self._prepare_core_df(df, id_col=id_col, time_col=time_col, target_col=target_col)

        with self._get_model() as model:
            return self._forecast_with_model(
                model=model,
                core_df=core_df,
                horizon=horizon,
                freq=freq,
                qc=qc,
                id_col=id_col,
                time_col=time_col,
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
        self._warn_features_unused(static_df)

        df = maybe_convert_col_to_datetime(df, time_col)
        sort_idxs = maybe_compute_sort_indices(df, id_col, time_col)
        if sort_idxs is not None:
            df = take_rows(df, sort_idxs)

        df = df.copy(deep=False)
        df[target_col] = self._ensure_float32(df[target_col], target_col)

        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)

        splits = backtest_splits(
            df,
            n_windows=n_windows,
            h=horizon,
            step_size=step_size,
            id_col=id_col,
            time_col=time_col,
            freq=freq,
        )

        results: list[pd.DataFrame] = []
        with self._get_model() as model:
            for _, (cutoffs, train, valid) in tqdm(enumerate(splits), desc=f"{self.alias} CV"):
                core_train = self._prepare_core_df(train, id_col=id_col, time_col=time_col, target_col=target_col)

                pred_df = self._forecast_with_model(
                    model=model,
                    core_df=core_train,
                    horizon=horizon,
                    freq=freq,
                    qc=qc,
                    id_col=id_col,
                    time_col=time_col,
                )

                pred_df = join(pred_df, cutoffs, on=id_col, how="left")
                result = join(valid, pred_df, on=[id_col, time_col])

                if result.shape[0] < valid.shape[0]:
                    raise ValueError(
                        "Cross validation produced fewer rows than expected. "
                        "Check freq alignment and missing periods."
                    )
                results.append(result)

        out_df = vertical_concat(results)
        out_df = drop_index_if_pandas(out_df)

        # drop provided features from output (ignored by TiRex anyway)
        feature_cols = set(self.futr_exog_list + self.hist_exog_list + self.stat_exog_list)
        drop_cols = list(feature_cols & set(out_df.columns))
        if drop_cols:
            out_df.drop(columns=drop_cols, inplace=True)
        
        return out_df
