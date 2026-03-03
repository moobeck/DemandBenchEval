import pandas as pd
from utilsforecast.evaluation import evaluate
import logging
from src.configurations.evaluation.metrics import MetricConfig
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.utils.quantile import QuantileUtils
import seaborn as sns
from typing import Optional, List, Dict, Any
from matplotlib import pyplot as plt
import re


import gc
import logging
import re
import pandas as pd
from typing import Optional, List, Dict, Any

from utilsforecast.evaluation import evaluate
from src.configurations.evaluation.metrics import MetricConfig
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.utils.quantile import QuantileUtils


import gc
import logging
import re
import pandas as pd
from typing import Optional, List, Dict, Any, Iterable

from utilsforecast.evaluation import evaluate
from src.configurations.evaluation.metrics import MetricConfig
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.utils.quantile import QuantileUtils


class Evaluator:
    _POINT_SUFFIXES = ("median", "mean")
    _PROB_PAT = re.compile(r"^(?P<model>.+)-(lo|hi)-(?P<level>\d+)$")

    def __init__(
        self,
        metric_config: MetricConfig,
        forecast_columns: ForecastColumnConfig,
        models: Optional[List[str]] = None,
        chunk_size: int = 1000,  # number of series per chunk
    ):
        self._metric_config = metric_config
        self._forecast_columns = forecast_columns
        self._explicit_models = models
        self.chunk_size = int(chunk_size) if chunk_size and int(chunk_size) > 0 else 0

    def _get_level(self) -> Optional[List[int]]:
        qcfg = self._metric_config.quantiles
        if qcfg and self._metric_config.contains_probabilistic:
            quantiles = QuantileUtils.create_quantiles(qcfg)
            return QuantileUtils.quantiles_to_level(quantiles)
        return None

    def _infer_model_names(self, df: pd.DataFrame) -> List[str]:
        if self._explicit_models:
            return list(self._explicit_models)

        names = set()
        for c in df.columns:
            parts = c.split("-")
            if len(parts) >= 2 and parts[-1] in self._POINT_SUFFIXES:
                names.add("-".join(parts[:-1]))
                continue
            m = self._PROB_PAT.match(c)
            if m:
                names.add(m.group("model"))

        model_names = sorted(names)
        if not model_names:
            raise ValueError(
                "No model columns detected. Expected '{model}-median' and/or '{model}-lo-90'/'{model}-hi-90'."
            )
        return model_names

    def _ensure_point_forecasts(self, df: pd.DataFrame, model_names: List[str]) -> pd.DataFrame:
        rename_map = {}
        missing = []
        for m in model_names:
            if m in df.columns:
                continue
            if f"{m}-median" in df.columns:
                rename_map[f"{m}-median"] = m
                continue
            if f"{m}-mean" in df.columns:
                rename_map[f"{m}-mean"] = m
                continue
            missing.append(m)

        if missing:
            raise ValueError(
                f"Missing point forecast columns for models: {missing}. Need '{m}' or '{m}-median'/'{m}-mean'."
            )

        return df.rename(columns=rename_map) if rename_map else df

    def _trim_for_evaluation(self, df: pd.DataFrame, model_names: List[str]) -> pd.DataFrame:
        id_col = self._forecast_columns.time_series_index
        time_col = self._forecast_columns.date
        target_col = self._forecast_columns.target

        keep = {id_col, time_col, target_col}

        cutoff_col = getattr(self._forecast_columns, "cutoff", None)
        if cutoff_col and cutoff_col in df.columns:
            keep.add(cutoff_col)

        keep |= set(model_names)

        level = self._get_level()
        if level:
            levels = set(map(str, level))
            prefixes = tuple(m + "-" for m in model_names)
            for c in df.columns:
                if c.startswith(prefixes):
                    m = self._PROB_PAT.match(c)
                    if m and m.group("level") in levels:
                        keep.add(c)

        return df.loc[:, [c for c in df.columns if c in keep]]

    def _shrink_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self._forecast_columns.time_series_index

        # category ids save a lot
        if df[id_col].dtype == "object":
            df[id_col] = df[id_col].astype("category")

        # float64 -> float32
        float64_cols = df.select_dtypes(include=["float64"]).columns
        if len(float64_cols):
            df.loc[:, float64_cols] = df.loc[:, float64_cols].astype("float32")

        return df

    def _iter_id_chunks(self, unique_ids: pd.Index, chunk_size: int) -> Iterable[pd.Index]:
        n = len(unique_ids)
        for i in range(0, n, chunk_size):
            yield unique_ids[i : i + chunk_size]

    def _slice_by_id_block(self, df_sorted: pd.DataFrame, ids_sorted: pd.Series, id_block: pd.Index) -> pd.DataFrame:
        """
        df_sorted is sorted by id. ids_sorted is df_sorted[id_col] as a Series (same order).
        id_block is a sorted list/index of ids.
        We locate row start/end boundaries via searchsorted to avoid building big boolean masks.
        """
        arr = ids_sorted.to_numpy()
        block = id_block.to_numpy()

        start = arr.searchsorted(block[0], side="left")
        end = arr.searchsorted(block[-1], side="right")

        # This relies on the block being contiguous in sorted id order.
        # If ids can be missing in between, that's fine; slice may include extra ids.
        # So we do a small final filter INSIDE the slice (cheap, slice is much smaller than full df).
        out = df_sorted.iloc[start:end]
        return out[out[ids_sorted.name].isin(id_block)]

    def evaluate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logging.info("Starting evaluation...")

        model_names = self._infer_model_names(df)
        logging.info("Detected %d model(s): %s", len(model_names), ", ".join(model_names[:20]))

        df = self._ensure_point_forecasts(df, model_names)
        df = self._trim_for_evaluation(df, model_names)
        df = self._shrink_dtypes(df)

        id_col = self._forecast_columns.time_series_index
        time_col = self._forecast_columns.date

        mem_gb = df.memory_usage(deep=True).sum() / 1e9
        n_ids = df[id_col].nunique()
        logging.info(
            "Eval frame: rows=%d cols=%d series=%d approx_mem=%.2f GB",
            len(df), df.shape[1], n_ids, mem_gb
        )

        eval_base_kwargs = dict(
            models=model_names,
            target_col=self._forecast_columns.target,
            time_col=time_col,
            id_col=id_col,
            level=self._get_level(),
        )
        eval_base_kwargs.update(kwargs)

        metric_fns = list(self._metric_config.metrics.values())

        # Sort once so each chunk is a contiguous block
        df_sorted = df.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)
        ids_sorted = df_sorted[id_col]

        # Unique ids in sorted order
        unique_ids = pd.Index(ids_sorted.unique())
        chunk_size = self.chunk_size if self.chunk_size else len(unique_ids)

        logging.info("Chunked evaluation: chunk_size=%d series, total_series=%d", chunk_size, len(unique_ids))

        all_results = []

        for chunk_idx, id_block in enumerate(self._iter_id_chunks(unique_ids, chunk_size), start=1):
            # id_block is already sorted in the same order as unique_ids
            batch_df = self._slice_by_id_block(df_sorted, ids_sorted, id_block)

            logging.info(
                "Chunk %d: series=%d rows=%d",
                chunk_idx, len(id_block), len(batch_df)
            )

            # Evaluate one metric at a time inside the chunk to keep peak memory lower
            chunk_metric_results = []
            for metric_fn in metric_fns:
                logging.info("Chunk %d metric: %s", chunk_idx, getattr(metric_fn, "__name__", str(metric_fn)))
                res = evaluate(df=batch_df, metrics=[metric_fn], **eval_base_kwargs)
                chunk_metric_results.append(res)

                del res
                gc.collect()

            all_results.append(pd.concat(chunk_metric_results, ignore_index=True))

            del batch_df, chunk_metric_results
            gc.collect()

        return pd.concat(all_results, ignore_index=True)

    def summarize_metrics(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        model_cols = [c for c in metrics_df.columns if c not in {self._forecast_columns.time_series_index, "metric"}]
        metrics = metrics_df["metric"].unique()

        summary: Dict[str, Any] = {metric: {} for metric in metrics}
        for metric in metrics:
            for model in model_cols:
                vals = metrics_df.loc[metrics_df["metric"] == metric, model].values
                summary[metric][model] = {
                    "mean": float(vals.mean()) if len(vals) else None,
                    "std": float(vals.std()) if len(vals) else None,
                    "min": float(vals.min()) if len(vals) else None,
                    "max": float(vals.max()) if len(vals) else None,
                }
        return summary



class EvaluationPlotter:
    """
    A class to plot evaluation metrics distributions for forecast models.
    """

    def __init__(
        self,
        evaluations: pd.DataFrame,
        forecast_columns: ForecastColumnConfig,
        metric_config: MetricConfig,
        ylim: Optional[tuple] = None,
    ):
        """
        Initializes the plotter.

        :param evaluations: DataFrame containing columns for sku index, metric labels, and model error values.
        :param forecast_columns: ForecastColumnConfig instance with column names.
        :param metrics: List of metric names to filter and plot (case-insensitive).
        :param ylim: Tuple of (ymin, ymax) for the plots.
        """
        self.evaluations = evaluations.copy()
        self.forecast_columns = forecast_columns
        self._metric_config = metric_config
        self.metrics = [m.name.upper() for m in self._metric_config.metrics]
        self.ylim = ylim or (-0.5, 4)

    def _prepare_long_df(self) -> pd.DataFrame:
        """
        Transforms the wide evaluations DataFrame into a long format.
        """
        # Identify model columns
        id_cols = [self.forecast_columns.time_series_index, "metric"]
        model_cols = [c for c in self.evaluations.columns if c not in id_cols]

        # Melt into long form
        long_df = self.evaluations.melt(
            id_vars=id_cols, value_vars=model_cols, var_name="model", value_name="error"
        )

        # Upper-case the metric labels
        long_df["metric"] = long_df["metric"].str.upper()

        # Filter desired metrics
        long_df = long_df[long_df["metric"].isin(self.metrics)]
        return long_df, model_cols

    def plot_error_distributions(self):
        """
        Plots box plots for the specified metrics across models, with mean lines.
        """

        logging.info("Plotting error distributions...")

        long_df, models = self._prepare_long_df()

        sns.set(style="whitegrid")
        g = sns.catplot(
            data=long_df,
            x="model",
            y="error",
            col="metric",
            kind="box",
            sharey=True,
            height=5,
            aspect=1,
            width=0.6,
        )

        g.set(ylim=self.ylim)

        # Overlay mean lines
        for ax, metric in zip(g.axes.flatten(), self.metrics):
            means = (
                long_df[long_df["metric"] == metric].groupby("model")["error"].mean()
            )
            for idx, model in enumerate(models):
                m_val = means.get(model)
                if m_val is not None:
                    ax.hlines(
                        y=m_val,
                        xmin=idx - 0.3,
                        xmax=idx + 0.3,
                        linewidth=2,
                        colors="orange",
                        linestyles="dashed",
                    )

        # Labels & titles
        g.set_axis_labels("", "Error Value")
        g.set_titles("{col_name} Distribution")
        plt.tight_layout()

        return g
