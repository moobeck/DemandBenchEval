from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm

from chronos.chronos2 import Chronos2Pipeline
from src.forecasting.models.foundation.utils.forecaster import (
    Forecaster,
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


def interpolate_quantiles(
    target_quantiles: list[float],
    source_quantiles: list[float],
    source_values: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized linear interpolation of quantile forecasts.

    Args:
        target_quantiles: desired quantiles.
        source_quantiles: quantiles available in source_values (last dim).
        source_values: tensor of shape (..., n_source_quantiles)

    Returns:
        tensor of shape (..., n_target_quantiles)
    """
    if len(target_quantiles) == 0:
        raise ValueError("target_quantiles must be non-empty.")
    if len(source_quantiles) == 0:
        raise ValueError("source_quantiles must be non-empty.")
    if source_values.shape[-1] != len(source_quantiles):
        raise ValueError(
            f"source_values last dim ({source_values.shape[-1]}) must match len(source_quantiles) ({len(source_quantiles)})."
        )

    device = source_values.device
    dtype = source_values.dtype

    source_q = torch.tensor(source_quantiles, device=device, dtype=torch.float32)
    target_q = torch.tensor(target_quantiles, device=device, dtype=torch.float32)

    # sort source quantiles and reorder values accordingly
    sort_idx = torch.argsort(source_q)
    source_q = source_q[sort_idx]
    source_values = torch.index_select(source_values, dim=-1, index=sort_idx)

    n_source = source_q.numel()
    prefix = source_values.shape[:-1]

    # insertion indices in [0..n_source]
    ins = torch.searchsorted(source_q, target_q)
    upper = ins.clamp(0, n_source - 1)
    lower = (ins - 1).clamp(0, n_source - 1)

    # expand indices to (..., n_target)
    view_shape = (1,) * len(prefix) + (-1,)
    lower_idx = lower.view(view_shape).expand(*prefix, -1)
    upper_idx = upper.view(view_shape).expand(*prefix, -1)

    lower_val = torch.take_along_dim(source_values, lower_idx, dim=-1)
    upper_val = torch.take_along_dim(source_values, upper_idx, dim=-1)

    lower_q = source_q[lower].view(view_shape).expand(*prefix, -1)
    upper_q = source_q[upper].view(view_shape).expand(*prefix, -1)

    denom = upper_q - lower_q
    tq = target_q.view(view_shape).expand(*prefix, -1)

    # for out-of-range (or exact match), denom can be 0 -> weight 0 -> pick lower_val (==upper_val)
    weight = torch.where(denom > 0, (tq - lower_q) / denom, torch.zeros_like(denom, dtype=torch.float32))
    weight = weight.to(dtype=dtype)

    return lower_val + weight * (upper_val - lower_val)


def quantiles_to_outputs(quantiles):
    output_names = []
    for q in quantiles:
        if q < 0.50:
            output_names.append(f"-lo-{int(np.round(100-200*q,0))}")
        elif q > 0.50:
            output_names.append(f"-hi-{int(np.round(100-200*(1-q),0))}")
        else:
            output_names.append("-median")
    return quantiles, output_names


class Chronos(Forecaster):
    """
    Chronos models are large pre-trained models for time series forecasting.
    Chronos2 supports probabilistic forecasting and provides zero-shot capabilities for univariate, multivariate, and covariate-informed tasks.
    See the official repository for more details.
    """

    def __init__(
        self,
        repo_id: str = "s3://autogluon/chronos-2",
        batch_size: int = 256,
        alias: str = "Chronos",
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

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    @contextmanager
    def _get_model(self) -> Chronos2Pipeline:
        model = Chronos2Pipeline.from_pretrained(
            self.repo_id,
            device_map=self.device,
        )
        try:
            yield model
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _rename_forecast_columns(
        self,
        df: pd.DataFrame,
        quantiles: list[float] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        pred_col = "predictions"
        rename_mapping = {pred_col: f"{self.alias}"}

        features = self.stat_exog_list + self.hist_exog_list + self.futr_exog_list
        base_columns = ["cutoff", id_col, time_col, target_col]

        quantile_cols = [col for col in df.columns if col not in base_columns + features + [pred_col]]

        quantile_cols, level_cols = quantiles_to_outputs([float(q) for q in quantile_cols])
        level_cols = [f"{self.alias}{level}" for level in level_cols]
        rename_mapping.update(dict(zip([str(q) for q in quantile_cols], level_cols)))

        return df.rename(mapper=rename_mapping, axis=1)

    @staticmethod
    def _as_float32_numpy(col: pd.Series, name: str) -> np.ndarray:
        arr = col.to_numpy()
        if arr.dtype == object:
            try:
                arr = pd.to_numeric(col, errors="raise").to_numpy()
            except Exception as e:
                raise TypeError(f"Column '{name}' must be numeric. Got dtype=object and conversion failed: {e}")
        return arr.astype(np.float32, copy=False)

    def _build_gpu_store(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        feature_cols: list[str],
        device: torch.device,
    ) -> dict[Any, dict[str, Any]]:
        store: dict[Any, dict[str, Any]] = {}

        # df must be sorted by (id, time) already
        for uid, g in df.groupby(id_col, sort=False):
            ts = g[time_col].to_numpy()  # keep on CPU
            y_np = self._as_float32_numpy(g[target_col], target_col)
            y = torch.as_tensor(y_np, device=device, dtype=torch.float32)

            covs: dict[str, torch.Tensor] = {}
            for c in feature_cols:
                cov_np = self._as_float32_numpy(g[c], c)
                covs[c] = torch.as_tensor(cov_np, device=device, dtype=torch.float32)

            store[uid] = {"ts": ts, "y": y, "covs": covs}

        return store

    def _predict_window_gpu(
        self,
        model: Chronos2Pipeline,
        store: dict[Any, dict[str, Any]],
        cutoffs: pd.DataFrame,
        horizon: int,
        quantiles: list[float],
        id_col: str,
        time_col: str,
        target_col: str,
        device: torch.device,
    ) -> pd.DataFrame:
        training_quantiles = list(model.quantiles)
        if 0.5 not in training_quantiles:
            raise RuntimeError("Model quantiles do not include 0.5; cannot derive point forecast as median.")

        median_idx = training_quantiles.index(0.5)

        # fast path: direct indexing if requested quantiles are a subset
        subset = set(quantiles).issubset(training_quantiles)
        if subset:
            q_idx = torch.tensor([training_quantiles.index(q) for q in quantiles], dtype=torch.long)
        else:
            q_idx = None

        # long-horizon unrolling quantiles must be subset of training quantiles
        default_unrolled = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        unrolled = [q for q in default_unrolled if q in training_quantiles] or [0.5]
        unrolled_q = torch.tensor(unrolled, device=device, dtype=torch.float32)

        cov_order = self.hist_exog_list + self.futr_exog_list + self.stat_exog_list
        known_future = set(self.futr_exog_list + self.stat_exog_list)

        n_variates_per_task = 1 + len(cov_order)
        context_length = model.model_context_length

        # batch_size here counts "time series streams" (targets + covariates)
        max_tasks_per_batch = max(1, int(self.batch_size // max(1, n_variates_per_task)))

        pred_parts: list[pd.DataFrame] = []

        ids = cutoffs[id_col].to_list()
        cut_ts = cutoffs["cutoff"].to_list()

        nan_ctx = torch.nan  # readability

        for start in range(0, len(ids), max_tasks_per_batch):
            end = min(len(ids), start + max_tasks_per_batch)
            chunk_ids = ids[start:end]
            chunk_cut = cut_ts[start:end]

            # allocate for worst-case (no skips)
            max_tasks = len(chunk_ids)
            Bmax = max_tasks * n_variates_per_task
            ctx = torch.full((Bmax, context_length), nan_ctx, device=device, dtype=torch.float32)
            fut = torch.full((Bmax, horizon), nan_ctx, device=device, dtype=torch.float32)
            group_ids = torch.empty((Bmax,), device=device, dtype=torch.long)

            kept_uids: list[Any] = []
            future_ts_list: list[np.ndarray] = []

            kept = 0
            for uid, cutoff_ts in zip(chunk_ids, chunk_cut):
                item = store.get(uid)
                if item is None:
                    continue

                ts = item["ts"]
                y = item["y"]
                covs: dict[str, torch.Tensor] = item["covs"]

                cutoff64 = np.datetime64(cutoff_ts)
                idx = int(np.searchsorted(ts, cutoff64))
                if idx >= len(ts) or ts[idx] != cutoff64:
                    raise ValueError(f"Cutoff timestamp {cutoff_ts} not found for series '{uid}'.")
                if idx + horizon >= len(ts):
                    raise ValueError(f"Series '{uid}' is too short for horizon={horizon} at cutoff={cutoff_ts}.")

                hist_start = max(0, idx + 1 - context_length)

                base = kept * n_variates_per_task
                group_ids[base : base + n_variates_per_task] = kept

                # target history
                y_hist = y[hist_start : idx + 1]
                y_use = y_hist[-context_length:]
                ctx[base, -y_use.numel() :] = y_use

                # covariates (history always; future only for known-future covs)
                for j, c in enumerate(cov_order, start=1):
                    row = base + j
                    c_hist = covs[c][hist_start : idx + 1]
                    c_use = c_hist[-context_length:]
                    ctx[row, -c_use.numel() :] = c_use
                    if c in known_future:
                        c_fut = covs[c][idx + 1 : idx + 1 + horizon]
                        if c_fut.numel() != horizon:
                            raise ValueError(f"Future covariate '{c}' for series '{uid}' has length != horizon.")
                        fut[row, :] = c_fut

                future_ts = ts[idx + 1 : idx + 1 + horizon]
                if future_ts.shape[0] != horizon:
                    raise ValueError(f"Future timestamps length mismatch for series '{uid}' at cutoff={cutoff_ts}.")
                future_ts_list.append(future_ts)
                kept_uids.append(uid)
                kept += 1

            if kept == 0:
                continue

            B = kept * n_variates_per_task
            ctx_t = ctx[:B]
            fut_t = fut[:B]
            group_ids_t = group_ids[:B]

            target_idx_ranges = [(i * n_variates_per_task, i * n_variates_per_task + 1) for i in range(kept)]

            with torch.inference_mode():
                preds_list = model._predict_batch(
                    context=ctx_t,
                    group_ids=group_ids_t,
                    future_covariates=fut_t,
                    unrolled_quantiles_tensor=unrolled_q,
                    prediction_length=horizon,
                    max_output_patches=model.max_output_patches,
                    target_idx_ranges=target_idx_ranges,
                )

            # preds_list: list of CPU tensors, each (1, n_train_q, horizon)
            preds = torch.stack(preds_list, dim=0)  # (kept, 1, n_train_q, h) on CPU
            preds_bvhq = rearrange(preds, "b v q h -> b v h q")  # (kept, 1, h, n_train_q)

            point = preds_bvhq[..., median_idx]  # (kept, 1, h)

            if q_idx is not None:
                q_idx_cpu = q_idx.to(device=preds_bvhq.device)
                q_pred = torch.index_select(preds_bvhq, dim=-1, index=q_idx_cpu)  # (kept, 1, h, n_out)
            else:
                q_pred = interpolate_quantiles(quantiles, training_quantiles, preds_bvhq)  # (kept, 1, h, n_out)

            ds_col = np.concatenate(future_ts_list, axis=0)
            id_col_arr = np.repeat(np.array(kept_uids, dtype=object), horizon)
            pred_arr = point[:, 0, :].reshape(-1).numpy()

            out: dict[str, Any] = {
                id_col: id_col_arr,
                time_col: ds_col,
                "target_name": np.full(ds_col.shape[0], target_col, dtype=object),
                "predictions": pred_arr,
            }

            q_pred_np = q_pred[:, 0, :, :].reshape(kept * horizon, len(quantiles)).numpy()
            for j, q in enumerate(quantiles):
                out[str(q)] = q_pred_np[:, j]

            pred_parts.append(pd.DataFrame(out))

        if not pred_parts:
            return pd.DataFrame(columns=[id_col, time_col, "target_name", "predictions"] + [str(q) for q in quantiles])

        return pd.concat(pred_parts, ignore_index=True)

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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for Chronos cross_validation, but torch.cuda.is_available() is False.")
        if self.device == "cpu":
            raise RuntimeError("Chronos was initialized without CUDA; GPU is required for cross_validation.")

        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        df = maybe_convert_col_to_datetime(df, time_col)

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

        device = torch.device(self.device)
        feature_cols = self.stat_exog_list + self.hist_exog_list + self.futr_exog_list

        # move full panel once
        gpu_store = self._build_gpu_store(
            df=df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            feature_cols=feature_cols,
            device=device,
        )

        results = []
        with self._get_model() as model:
            for _, (cutoffs, _train, valid) in tqdm(enumerate(splits)):
                pred_df = self._predict_window_gpu(
                    model=model,
                    store=gpu_store,
                    cutoffs=cutoffs,
                    horizon=horizon,
                    quantiles=quantiles,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    device=device,
                )

                pred_df = join(pred_df, cutoffs, on=id_col, how="left")
                result = join(valid, pred_df, on=[id_col, time_col])

                if result.shape[0] < valid.shape[0]:
                    raise ValueError(
                        "Cross validation result produced less results than expected. "
                        "Please verify that the frequency parameter (freq) matches your series "
                        "and that there aren't any missing periods."
                    )

                results.append(result)

            out_df = vertical_concat(results)
            out_df = self._transform_forecast_results(quantiles, id_col, time_col, target_col, out_df)

        return out_df

    def _transform_forecast_results(self, quantiles, id_col, time_col, target_col, out_df):
        out_df = drop_index_if_pandas(out_df)

        columns_to_drop = set(self.stat_exog_list + self.hist_exog_list + self.futr_exog_list) & set(out_df.columns)
        columns_to_drop.add("target_name")
        out_df.drop(columns=list(columns_to_drop), inplace=True)

        out_df = self._rename_forecast_columns(
            out_df,
            quantiles=quantiles,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        return out_df
