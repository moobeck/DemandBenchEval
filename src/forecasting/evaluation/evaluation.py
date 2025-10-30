import pandas as pd
from utilsforecast.evaluation import evaluate
import logging
from src.configurations.evaluation.metrics import MetricConfig
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.forecasting.utils.quantile import QuantileUtils
import seaborn as sns
from typing import Optional, List, Dict, Any
from matplotlib import pyplot as plt


class Evaluator:
    """
    A class to evaluate the performance of a forecasting model using various metrics.
    """

    def __init__(
        self,
        metric_config: MetricConfig,
        forecast_columns: ForecastColumnConfig,
    ):

        self._metric_config = metric_config
        self._forecast_columns = forecast_columns

        self.metrics = self._metric_config.metrics

    def _get_model_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Get the model columns from the DataFrame.
        """
        # Get all columns that are not metadata columns
        metadata_cols = {
            self._forecast_columns.time_series_index,
            self._forecast_columns.date,
            self._forecast_columns.target,
            self._forecast_columns.cutoff,
            "metric",
        }

        potential_model_cols = [col for col in df.columns if col not in metadata_cols]

        model_names = set()
        for col in potential_model_cols:
            # Split by '-' and take the first part as the base model name
            base_name = col.split("-")[0]
            model_names.add(base_name)

        model_cols = [name for name in model_names]

        return model_cols

    def _fill_model_columns(self, df: pd.DataFrame, model_names) -> pd.DataFrame:
        """
        Ensure that for each model, there is a corresponding column in the DataFrame.
        If a model column is missing fill it using the '{model}-median' column.
        """

        # Models that already have direct columns
        present_models = {m for m in model_names if m in df.columns}

        # Models that can be created from '{model}-median'
        median_available = [
            m
            for m in model_names
            if m not in present_models and f"{m}-median" in df.columns
        ]

        # Models missing both direct and median columns
        missing_models = [
            m
            for m in model_names
            if m not in present_models and m not in median_available
        ]

        if missing_models:
            raise ValueError(
                f"Missing columns for models {missing_models}. "
                "Each model must have either a column named '{model}' or '{model}-median'."
            )

        # Create missing model columns from their '-median' counterparts in a single assignment
        if median_available:
            median_cols = [f"{m}-median" for m in median_available]
            # Assign values; ensure correct column ordering by using .values to avoid align-by-label
            df[median_available] = df[median_cols].values

        return df

    def _get_level(self):
        """
        Calculate the quantile levels based on the provided quantiles.
        """

        if self._metric_config.contains_probabilistic:

            quantiles_cfg = self._metric_config.quantiles

            if quantiles_cfg is None:
                return None

            quantiles = QuantileUtils.create_quantiles(quantiles_cfg)
            levels = QuantileUtils.quantiles_to_level(quantiles)

            return levels

        else:
            return None

    def evaluate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Evaluate the model using the specified metrics.
        """

        logging.info("Starting evaluation...")
        model_names = self._get_model_cols(df)
        df = self._fill_model_columns(df, model_names)

        return evaluate(
            df=df,
            models=model_names,
            target_col=self._forecast_columns.target,
            time_col=self._forecast_columns.date,
            id_col=self._forecast_columns.time_series_index,
            metrics=list(self.metrics.values()),
            level=self._get_level(),
            **kwargs,
        )

    def summarize_metrics(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Summarizes the metrics DataFrame into a dictionary.
        """

        model_cols = self._get_model_cols(metrics_df)
        metrics = metrics_df["metric"].unique()

        summary = {metric: {} for metric in metrics}

        for metric in metrics:
            for model in model_cols:

                metric_values = metrics_df.loc[
                    metrics_df["metric"] == metric, model
                ].values
                if len(metric_values) > 0:
                    summary[metric][model] = {
                        "mean": metric_values.mean(),
                        "std": metric_values.std(),
                        "min": metric_values.min(),
                        "max": metric_values.max(),
                    }
                else:
                    summary[metric][model] = {
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
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
