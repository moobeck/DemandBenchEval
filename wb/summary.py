from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("eval_summarizer")

ID_COL_CANONICAL = "timeSeriesID"
ID_NAME_VARIANTS = {"timeseriesid", "productid", "storeid"}
CLASSIFICATION_DIR = Path("wb/artifacts/classification")
CLASSIFICATION_SUFFIX = "_ids"


def _normalize_id_value(value: object) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    if not raw:
        return raw
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return raw
    if numeric.is_integer():
        return str(int(numeric))
    return raw


def _prepare_summary_dataframe(csv_path: Path) -> tuple[pd.DataFrame, str, str, str | None]:
    df = pd.read_csv(csv_path)
    df = _drop_meaningless_index_cols(df)

    metric_col = _find_metric_col(df, csv_path)
    df, id_col = _find_and_normalize_id_col(df)
    if id_col:
        df[id_col] = df[id_col].apply(_normalize_id_value)

    return df, infer_dataset_name(csv_path), metric_col, id_col


def _summarize_filtered_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    model_from_dir: str,
    metric_col: str,
    id_col: str | None,
    allowed_ids: set[str] | None,
) -> pd.DataFrame:
    if allowed_ids is not None:
        if not id_col or not allowed_ids:
            return pd.DataFrame()
        filtered = df.loc[df[id_col].isin(allowed_ids)]
        if filtered.empty:
            return pd.DataFrame()
    else:
        filtered = df

    id_vars = [metric_col]
    if id_col:
        id_vars.insert(0, id_col)

    value_cols = [c for c in filtered.columns if c not in set(id_vars)]
    if not value_cols:
        raise ValueError(f"No value columns found in {dataset_name} ({model_from_dir})")

    long_df = _to_long(filtered, id_vars, value_cols, model_from_dir)

    g = long_df.groupby(["model", metric_col], dropna=False)["value"]
    wide = (
        pd.DataFrame(
            {
                "mean": g.mean(),
                "median": g.median(),
                "std": g.std(ddof=1),
            }
        )
        .reset_index()
        .rename(columns={metric_col: "metric"})
    )

    out = wide.melt(
        id_vars=["model", "metric"],
        value_vars=["mean", "median", "std"],
        var_name="agg",
        value_name="value",
    )
    out.insert(0, "dataset", dataset_name)
    return out[["dataset", "model", "metric", "agg", "value"]]


def infer_dataset_name(csv_path: Path) -> str:
    return csv_path.stem


def _drop_meaningless_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.startswith("Unnamed") or c.lower() == "index"]
    return df.drop(columns=drop_cols) if drop_cols else df


def _find_metric_col(df: pd.DataFrame, csv_path: Path) -> str:
    metric_col = next((c for c in df.columns if c.lower() == "metric"), None)
    if metric_col is None:
        raise ValueError(f"Missing required column 'metric' in {csv_path}")
    return metric_col


def _find_and_normalize_id_col(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    id_col = next((c for c in df.columns if c.lower() in ID_NAME_VARIANTS), None)
    if id_col is None:
        return df, None
    if id_col != ID_COL_CANONICAL:
        df = df.rename(columns={id_col: ID_COL_CANONICAL})
    return df, ID_COL_CANONICAL


def _to_long(df: pd.DataFrame, id_vars: list[str], value_cols: list[str], model_from_dir: str) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="model_col",
        value_name="value",
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["value"] = long_df["value"].replace([np.inf, -np.inf], np.nan)

    if len(value_cols) == 1:
        long_df["model"] = model_from_dir
    else:
        long_df["model"] = long_df["model_col"].astype(str).str.strip().str.lower()

    return long_df


def summarize_one_csv(csv_path: Path, model_from_dir: str) -> pd.DataFrame:
    df, dataset_name, metric_col, id_col = _prepare_summary_dataframe(csv_path)
    return _summarize_filtered_dataframe(df, dataset_name, model_from_dir, metric_col, id_col, None)


def _win_rate_row(values: np.ndarray) -> np.ndarray:
    # lower is always better
    valid = ~np.isnan(values)
    k = int(valid.sum())
    out = np.full_like(values, np.nan, dtype=float)
    if k <= 1:
        return out

    less = (values[:, None] < values[None, :]) & (valid[:, None] & valid[None, :])
    eq = (values[:, None] == values[None, :]) & (valid[:, None] & valid[None, :])
    np.fill_diagonal(eq, False)

    wins = less.sum(axis=1) + 0.5 * eq.sum(axis=1)
    out[valid] = wins[valid] / (k - 1)
    return out


def average_win_rate_ranking(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = summary[(summary["agg"] == "mean") & (summary["metric"].astype(str).str.lower() == metric.lower())]
    if df.empty:
        return pd.DataFrame()

    mat = df.pivot_table(index="dataset", columns="model", values="value", aggfunc="mean")
    arr = mat.to_numpy(dtype=float)

    win_rates = np.vstack([_win_rate_row(arr[i]) for i in range(arr.shape[0])])
    avg_win_rate = np.nanmean(win_rates, axis=0)

    out = pd.DataFrame(
        {
            "model": mat.columns,
            "avg_win_rate": avg_win_rate,
            "tasks_count": np.sum(~np.isnan(arr), axis=0),
        }
    )
    out = out.sort_values("avg_win_rate", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.insert(0, "metric", metric.upper())
    return out


def per_dataset_ranking(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = summary[(summary["agg"] == "mean") & (summary["metric"].astype(str).str.lower() == metric.lower())]
    if df.empty:
        return pd.DataFrame()

    mat = df.pivot_table(index="dataset", columns="model", values="value", aggfunc="mean")
    rows = []

    for ds, row in mat.iterrows():
        win = _win_rate_row(row.to_numpy(dtype=float))
        tmp = pd.DataFrame(
            {
                "dataset": ds,
                "model": mat.columns,
                "value": row.values,
                "win_rate": win,
            }
        ).dropna(subset=["value"])
        tmp = tmp.sort_values("win_rate", ascending=False).reset_index(drop=True)
        tmp.insert(0, "rank", range(1, len(tmp) + 1))
        tmp.insert(0, "metric", metric.upper())
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_classification_groups() -> dict[str, dict[str, set[str]]]:
    if not CLASSIFICATION_DIR.exists():
        return {}

    groups: dict[str, dict[str, set[str]]] = {}
    for path in sorted(CLASSIFICATION_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception as exc:
            log.warning("Skipping classification file %s (%s)", path, exc)
            continue

        dataset_map: dict[str, set[str]] = {}
        for dataset, ids in raw.items():
            if not ids:
                continue
            normalized = {_normalize_id_value(value) for value in ids}
            normalized.discard("")
            if normalized:
                dataset_map[str(dataset).lower()] = normalized

        if not dataset_map:
            continue

        group_name = path.stem
        if group_name.endswith(CLASSIFICATION_SUFFIX):
            group_name = group_name[: -len(CLASSIFICATION_SUFFIX)]
        groups[group_name] = dataset_map

    return groups


def _write_rankings(out_dir: Path, prefix: str, summary: pd.DataFrame, metrics: list[str]) -> None:
    for m in metrics:
        r = average_win_rate_ranking(summary, m)
        if not r.empty:
            r.to_csv(out_dir / f"{prefix}_{m.lower()}_ranking.csv", index=False)
        else:
            log.info("No rows found for ranking metric=%s (prefix=%s)", m, prefix)

        pr = per_dataset_ranking(summary, m)
        if not pr.empty:
            pr.to_csv(out_dir / f"{prefix}_per_dataset_{m.lower()}_ranking.csv", index=False)
        else:
            log.info("No rows found for per-dataset ranking metric=%s (prefix=%s)", m, prefix)


def summarize_all(
    eval_root: Path,
    out_dir: Path,
    overall_metrics: list[str],
    intermittent_metrics: list[str],
    enable_intermittent: bool = True,
) -> None:
    frames: list[pd.DataFrame] = []
    classification_groups = _load_classification_groups() if enable_intermittent else {}
    if classification_groups:
        log.info("Loaded classification groups: %s", ", ".join(sorted(classification_groups.keys())))

    group_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in classification_groups}

    model_dirs = [p for p in eval_root.iterdir() if p.is_dir()]
    log.info("Found %d model folders under %s", len(model_dirs), eval_root)

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name.strip().lower()
        csv_files = sorted(model_dir.rglob("*.csv"))
        log.info("Model '%s': %d csv files", model_name, len(csv_files))

        for csv in csv_files:
            try:
                df, dataset_name, metric_col, id_col = _prepare_summary_dataframe(csv)
                frame = _summarize_filtered_dataframe(df, dataset_name, model_name, metric_col, id_col, None)
                frames.append(frame)

                if classification_groups:
                    dataset_key = dataset_name.lower()
                    for group_name, dataset_map in classification_groups.items():
                        dataset_ids = dataset_map.get(dataset_key)
                        if not dataset_ids:
                            continue
                        group_frame = _summarize_filtered_dataframe(
                            df,
                            dataset_name,
                            model_name,
                            metric_col,
                            id_col,
                            dataset_ids,
                        )
                        if not group_frame.empty:
                            group_frames[group_name].append(group_frame)
            except Exception as e:
                log.warning("Skipping %s (%s)", csv, e)

    if not frames:
        raise RuntimeError("No evaluation CSVs summarized (nothing found or all files failed).")

    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values(["dataset", "model", "metric", "agg"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "eval_summary.csv", index=False)
    log.info("Wrote eval_summary.csv (%d rows)", len(summary))

    _write_rankings(out_dir, "overall", summary, overall_metrics)

    if not enable_intermittent:
        log.info("Intermittent disabled.")
        return

    if not classification_groups:
        log.info("No classification ids JSON found.")
        return

    for group_name, frames_list in group_frames.items():
        if not frames_list:
            log.info("No rows for classification group %s", group_name)
            continue

        group_summary = pd.concat(frames_list, ignore_index=True)
        group_summary = group_summary.sort_values(["dataset", "model", "metric", "agg"]).reset_index(drop=True)
        group_summary.to_csv(out_dir / f"{group_name}_summary.csv", index=False)
        log.info("Wrote %s_summary.csv (%d rows)", group_name, len(group_summary))

        metrics_for_group = (
            intermittent_metrics if "intermittent" in group_name.lower() else overall_metrics
        )
        _write_rankings(out_dir, group_name, group_summary, metrics_for_group)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, default=Path("wb/artifacts/evaluation-results"))
    p.add_argument("--out-dir", type=Path, default=Path("wb/artifacts/evaluation-summary"))

    # CHANGED: multiple overall metrics
    p.add_argument("--overall-metrics", nargs="*", default=["scaled_spec", "sapis", "MASE", "scaled_mqloss"])

    p.add_argument("--no-intermittent", action="store_true")
    p.add_argument("--intermittent-metrics", nargs="*", default=["scaled_spec", "sapis", "MASE", "scaled_mqloss"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summarize_all(
        eval_root=args.eval_root,
        out_dir=args.out_dir,
        overall_metrics=list(args.overall_metrics),
        intermittent_metrics=list(args.intermittent_metrics),
        enable_intermittent=not bool(args.no_intermittent),
    )
