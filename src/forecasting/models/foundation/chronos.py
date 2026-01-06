import pandas as pd
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
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)
from tqdm import tqdm
import numpy as np


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
    See the [official repository](https://github.com/amazon-science/chronos-forecasting) for more details.
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
        """
        Args:
            repo_id (str, optional): The
                path to load the Chronos model from.
            batch_size (int, optional): Batch size to use for inference.
                Larger models may require smaller batch sizes due to GPU
                memory constraints.
            alias (str, optional): Name to use for the model in output
                DataFrames and logs. Defaults to "Chronos".
            futr_exog_list (list[str], optional): List of future exogenous
                variables to include in the model. Defaults to None.
            hist_exog_list (list[str], optional): List of historical exogenous
                variables to include in the model. Defaults to None.
            stat_exog_list (list[str], optional): List of static exogenous
                variables to include in the model. Defaults to None.

        Notes:
            - GitHub: [amazon-science/chronos-forecasting]

        """

        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias
        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.repo_id,
            device_map="cuda"
        )
        self.futr_exog_list = futr_exog_list if futr_exog_list is not None else []
        self.hist_exog_list = hist_exog_list if hist_exog_list is not None else []
        self.stat_exog_list = stat_exog_list if stat_exog_list is not None else []

    def _rename_forecast_columns(
        self,
        df: pd.DataFrame,
        quantiles: list[float] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """
        Rename forecast columns in the output DataFrame based on quantiles.

        Args:
            df (pd.DataFrame): DataFrame containing forecast results.
            quantiles (list[float] | None): List of quantiles used for forecasting.
        Returns:
            pd.DataFrame: DataFrame with renamed forecast columns.
        """

        pred_col = "predictions"

        rename_mapping = {
            pred_col: f"{self.alias}",
        }

        features = self.stat_exog_list + self.hist_exog_list + self.futr_exog_list
        base_columns = ["cutoff", id_col, time_col, target_col]

        quantile_cols = [
            col for col in df.columns if col not in base_columns + features + [pred_col]
        ]

        quantile_cols, level_cols = quantiles_to_outputs(
            [float(q) for q in quantile_cols]
        )
        level_cols = [f"{self.alias}{level}" for level in level_cols]

        rename_mapping.update(dict(zip([str(q) for q in quantile_cols], level_cols)))

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

        base_cols = [id_col, time_col]

        print(f" Columns in training data: {df.columns.tolist()} ")


        for _, (cutoffs, train, valid) in tqdm(enumerate(splits)):      

            future_df = valid[base_cols + self.futr_exog_list + self.stat_exog_list]
            train = self._limit_context_length(
                train,
                time_col,
                horizon,
                freq,
            )
            pred_df = self.pipeline.predict_df(
                df=train,
                future_df=future_df,
                prediction_length=horizon,
                quantile_levels=quantiles,
                id_column=id_col,
                timestamp_column=time_col,
                target=target_col,
                batch_size=self.batch_size,
            )
            pred_df = join(
                pred_df,
                cutoffs,
                on=id_col,
                how="left",
            )
            result = join(
                valid,
                pred_df,
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
