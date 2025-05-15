import pandas as pd
from utilsforecast.evaluation import evaluate
import logging
from src.configurations.metrics import MetricConfig
from src.configurations.forecast_column import ForecastColumnConfig
import seaborn as sns
from typing import Optional, List
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

    def evaluate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Evaluate the model using the specified metrics.
        """

        logging.info("Starting evaluation...")

        model_cols = [
            col
            for col in df.columns
            if col
            not in [
                self._forecast_columns.sku_index,
                self._forecast_columns.date,
                self._forecast_columns.target,
                self._forecast_columns.cutoff,
            ]
        ]

        return evaluate(
            df=df,
            models=model_cols,
            target_col=self._forecast_columns.target,
            time_col=self._forecast_columns.date,
            id_col=self._forecast_columns.sku_index,
            metrics=list(self.metrics.values()),
            **kwargs
        )


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
        id_cols = [self.forecast_columns.sku_index, "metric"]
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
        Plots violin plots for the specified metrics across models, with mean lines.
        """

        logging.info("Plotting error distributions...")

        long_df, models = self._prepare_long_df()

        sns.set(style="whitegrid")
        g = sns.catplot(
            data=long_df,
            x="model",
            y="error",
            col="metric",
            kind="violin",
            inner=None,
            sharey=True,
            height=5,
            aspect=1,
        )

        g.set(ylim=self.ylim)

        # Overlay mean lines
        for ax, metric in zip(g.axes.flatten(), [m for m in self.metrics]):
            means = (
                long_df[long_df["metric"] == metric].groupby("model")["error"].mean()
            )
            for idx, model in enumerate(models):
                m_val = means.get(model, None)
                if m_val is not None:
                    ax.hlines(
                        y=m_val,
                        xmin=idx - 0.2,
                        xmax=idx + 0.2,
                        linewidth=3,
                        colors="orange",
                        linestyles="dotted",
                    )

        # Labels & titles
        g.set_axis_labels("", "Error Value")
        g.set_titles("{col_name} Distribution")
        plt.tight_layout()

        return g
