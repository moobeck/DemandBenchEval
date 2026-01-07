import os
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import repo_exists
from tqdm import tqdm

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
import timesfm
from timesfm import TimesFM_2p5_200M_torch

DEFAULT_QUANTILES_TFM: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_Q_PCTS = {int(q * 100) for q in DEFAULT_QUANTILES_TFM}
Q_PCT_TO_V2P5_INDEX = {int(q * 100): i + 1 for i, q in enumerate(DEFAULT_QUANTILES_TFM)}  # +1 skips "mean" at idx 0



class TimesFM(Forecaster):
    """
    TimesFM integration. 

    Covariates/static features:
    This wrapper accepts feature lists and static_df for interface compatibility with Chronos,
    but TimesFM is run without using them. A single warning is emitted.
    """

    def __init__(
        self,
        repo_id: str = "google/timesfm-2.5-200m-pytorch",
        context_length: int = 2048,
        batch_size: int = 64,
        alias: str = "TimesFM",
        futr_exog_list: list[str] | None = None,
        hist_exog_list: list[str] | None = None,
        stat_exog_list: list[str] | None = None,
        **kwargs: dict,
    ):
        if "pytorch" not in repo_id:
            raise ValueError("TimesFM wrapper supports PyTorch checkpoints only (repo_id must include 'pytorch').")

        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.kwargs = kwargs

        self.futr_exog_list = futr_exog_list if futr_exog_list is not None else []
        self.hist_exog_list = hist_exog_list if hist_exog_list is not None else []
        self.stat_exog_list = stat_exog_list if stat_exog_list is not None else []

        self._did_warn_features = False

        if "2.5" in repo_id:
            self._variant = "2.5"
        elif "2.0" in repo_id or "1.0" in repo_id:
            self._variant = "v1"
        else:
            raise ValueError("TimesFM wrapper supports repo_id containing '1.0', '2.0', or '2.5'.")

    def _warn_features_unused(self, static_df: pd.DataFrame | None) -> None:
        if self._did_warn_features:
            return
        if self.futr_exog_list or self.hist_exog_list or self.stat_exog_list or static_df is not None:
            warnings.warn(
                "TimesFM covariates/static features are not used in this integration. "
                "Feature args are accepted for interface compatibility only.",
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

        return df[[id_col, time_col, target_col]].rename(
            columns={id_col: "unique_id", time_col: "ds", target_col: "y"}
        )

    @staticmethod
    def _requested_quantiles_from_qc(qc: QuantileConverter) -> list[float] | None:
        if qc.quantiles is None:
            return None
        qs = [float(q) for q in qc.quantiles]
        missing = {int(round(q * 100)) for q in qs} - DEFAULT_Q_PCTS
        if missing:
            raise ValueError(
                "TimesFM supports only these quantiles: "
                f"{DEFAULT_QUANTILES_TFM}. "
                f"Requested quantiles include unsupported percentiles: {sorted(missing)}."
            )
        return qs

    @contextmanager
    def _get_predictor(self, prediction_length: int) -> Any:
        if self._variant == "v1":
            import timesfm_v1

            backend = "gpu" if torch.cuda.is_available() else "cpu"
            v2_version = "2.0" in self.repo_id

            context_len = min(self.context_length, 512) if not v2_version else self.context_length
            num_layers = 50 if v2_version else 20
            use_positional_embedding = not v2_version

            hparams = timesfm_v1.TimesFmHparams(
                backend=backend,
                horizon_len=prediction_length,
                quantiles=DEFAULT_QUANTILES_TFM,  # always load defaults; we can subset after inference
                context_len=context_len,
                num_layers=num_layers,
                use_positional_embedding=use_positional_embedding,
                per_core_batch_size=self.batch_size,
            )

            if os.path.exists(self.repo_id):
                ckpt_path = os.path.join(self.repo_id, "torch_model.ckpt")
                checkpoint = timesfm_v1.TimesFmCheckpoint(path=ckpt_path)
            elif repo_exists(self.repo_id):
                checkpoint = timesfm_v1.TimesFmCheckpoint(huggingface_repo_id=self.repo_id)
            else:
                raise OSError(
                    f"Failed to load model. Searched for '{self.repo_id}' as a local path and as a Hugging Face repo_id."
                )

            model = timesfm_v1.TimesFm(hparams=hparams, checkpoint=checkpoint)
            try:
                yield model
            finally:
                del model
                torch.cuda.empty_cache()
            return



        if os.path.exists(self.repo_id):
            safetensors_path = os.path.join(self.repo_id, "model.safetensors")
            model = TimesFM_2p5_200M_torch()
            model.load_checkpoint(safetensors_path)
        elif repo_exists(self.repo_id):
            model = TimesFM_2p5_200M_torch.from_pretrained(self.repo_id)
        else:
            raise OSError(
                f"Failed to load model. Searched for '{self.repo_id}' as a local path and as a Hugging Face repo_id."
            )

        default_cfg = {
            "max_context": self.context_length,
            "max_horizon": prediction_length,
            "normalize_inputs": True,
            "use_continuous_quantile_head": True,
            "fix_quantile_crossing": True,
            "force_flip_invariance": True,
            "infer_is_positive": True,
        }
        cfg = timesfm.ForecastConfig(**{**default_cfg, **(self.kwargs or {})})
        model.compile(cfg)

        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()

    def _forecast_v1(
        self,
        predictor: Any,
        core_df: pd.DataFrame,  # unique_id, ds, y
        horizon: int,
        freq: str,
        requested_q: list[float] | None,
        qc: QuantileConverter,
        id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        fcst_df = predictor.forecast_on_df(
            inputs=core_df,
            freq=freq,
            value_name="y",
            model_name=self.alias,
            num_jobs=1,
        )

        if requested_q is None:
            out = fcst_df[["unique_id", "ds", self.alias]].rename(columns={"unique_id": id_col, "ds": time_col})
            return out

        renames: dict[str, str] = {}
        keep_cols = {"unique_id", "ds", self.alias}

        for c in fcst_df.columns:
            prefix = f"{self.alias}-q-"
            if c.startswith(prefix):
                q_str = c[len(prefix) :]
                q = float(q_str)
                pct = int(round(q * 100))
                new_name = f"{self.alias}-q-{pct}"
                renames[c] = new_name

        fcst_df = fcst_df.rename(columns=renames)

        requested_pcts = {int(round(q * 100)) for q in requested_q}
        for pct in requested_pcts:
            keep_cols.add(f"{self.alias}-q-{pct}")

        out = fcst_df[list(keep_cols)].rename(columns={"unique_id": id_col, "ds": time_col})
        out = qc.convert_quantiles_to_level(out, models=[self.alias])
        return out

    def _forecast_v2p5(
        self,
        predictor: Any,
        core_df: pd.DataFrame,  # unique_id, ds, y
        horizon: int,
        freq: str,
        requested_q: list[float] | None,
        qc: QuantileConverter,
        id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        dataset = TimeSeriesDataset.from_df(core_df, batch_size=self.batch_size, dtype=torch.float32)
        out = dataset.make_future_dataframe(h=horizon, freq=freq)

        means: list[np.ndarray] = []
        quants: list[np.ndarray] = []

        with torch.inference_mode():
            for batch in tqdm(dataset, desc=f"{self.alias} forecast"):

                point_fcst, quant_fcst = predictor.forecast(inputs=batch, horizon=horizon)
                means.append(np.asarray(point_fcst))
                quants.append(np.asarray(quant_fcst))

        mean_np = np.concatenate(means, axis=0)
        quant_np = np.concatenate(quants, axis=0)  # (N, h, 10): mean then 10..90

        out[self.alias] = mean_np.reshape(-1)

        if requested_q is not None:
            for q in requested_q:
                pct = int(round(q * 100))
                idx = Q_PCT_TO_V2P5_INDEX[pct]
                out[f"{self.alias}-q-{pct}"] = quant_np[..., idx].reshape(-1)

            out = qc.convert_quantiles_to_level(out, models=[self.alias])

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
        requested_q = self._requested_quantiles_from_qc(qc)

        core_df = self._prepare_core_df(df, id_col=id_col, time_col=time_col, target_col=target_col)

        with self._get_predictor(prediction_length=horizon) as predictor:
            if self._variant == "v1":
                return self._forecast_v1(
                    predictor=predictor,
                    core_df=core_df,
                    horizon=horizon,
                    freq=freq,
                    requested_q=requested_q,
                    qc=qc,
                    id_col=id_col,
                    time_col=time_col,
                )
            return self._forecast_v2p5(
                predictor=predictor,
                core_df=core_df,
                horizon=horizon,
                freq=freq,
                requested_q=requested_q,
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
        requested_q = self._requested_quantiles_from_qc(qc)

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
        with self._get_predictor(prediction_length=horizon) as predictor:
            for _, (cutoffs, train, valid) in tqdm(enumerate(splits), desc=f"{self.alias} CV"):
                core_train = self._prepare_core_df(
                    train, id_col=id_col, time_col=time_col, target_col=target_col
                )

                if self._variant == "v1":
                    pred_df = self._forecast_v1(
                        predictor=predictor,
                        core_df=core_train,
                        horizon=horizon,
                        freq=freq,
                        requested_q=requested_q,
                        qc=qc,
                        id_col=id_col,
                        time_col=time_col,
                    )
                else:
                    pred_df = self._forecast_v2p5(
                        predictor=predictor,
                        core_df=core_train,
                        horizon=horizon,
                        freq=freq,
                        requested_q=requested_q,
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


        feature_cols = set(self.futr_exog_list + self.hist_exog_list + self.stat_exog_list)
        drop_cols = list(feature_cols & set(out_df.columns))
        if drop_cols:
            out_df.drop(columns=drop_cols, inplace=True)

        return out_df
