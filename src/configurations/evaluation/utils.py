from utilsforecast.losses import _base_docstring, mae, rmse, bias
from typing import Callable, List
import numpy as np
import pandas as pd


def _get_group_cols(df: pd.DataFrame, id_col: str, cutoff_col: str) -> list[str]:
    """Determines if we group by just ID or ID + Cutoff (for Cross Validation)."""
    if cutoff_col in df.columns:
        group_cols = [cutoff_col, id_col]
    else:
        group_cols = [id_col]
    return group_cols

def _base_docstring(*args, **kwargs) -> Callable:
    """Decorator to append standard arguments to docstrings."""
    base_docstring = """

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actual values and predictions.
        models (list of str): Columns that identify the models predictions.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.
    """

    def docstring_decorator(f: Callable):
        if f.__doc__ is not None:
            f.__doc__ += base_docstring
        return f

    return docstring_decorator(*args, **kwargs)

def _aggregate_column(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    cutoff_col: str,
    agg: str = "mean",
    alias: str = "scale",
) -> pd.DataFrame:
    """Group by ID/cutoff and aggregate the target column."""
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    aggregated = (
        df.groupby(group_cols, as_index=False)[target_col]
        .agg(agg)
        .rename(columns={target_col: alias})
    )
    return aggregated.sort_values(group_cols)


def _create_train_with_cutoffs(
    train_df: pd.DataFrame,
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    cutoff_col: str
):
    """Filters training data to ensure no data leakage for Cross Validation."""
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    train_df = train_df.copy()

    if cutoff_col in group_cols:
        cutoffs_df = df[group_cols].drop_duplicates()
        train_df = train_df.merge(cutoffs_df, on=id_col, how="inner")
        train_df = train_df[train_df[time_col] <= train_df[cutoff_col]]

    return train_df




def _spec_helper(y_true: np.ndarray, y_pred: np.ndarray, a1: float, a2: float) -> float:
    """
    Vectorized implementation of SPEC logic to avoid slow Python loops.
    Complexity reduced via matrix broadcasting.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    # 1. Precompute Cumulative Sums
    cum_y = np.cumsum(y_true)
    cum_f = np.cumsum(y_pred)

    # 2. Broadcast to create matrices for all (i, t) pairs
    cy_i = cum_y[:, np.newaxis] 
    cf_i = cum_f[:, np.newaxis]
    y_i = y_true[:, np.newaxis]
    f_i = y_pred[:, np.newaxis]
    
    cy_t = cum_y[np.newaxis, :]
    cf_t = cum_f[np.newaxis, :]
    
    # 3. Calculate Deltas
    delta1 = cy_i - cf_t 
    delta2 = cf_i - cy_t

    # 4. Calculate Costs
    term1 = a1 * np.minimum(y_i, delta1)
    term2 = a2 * np.minimum(f_i, delta2)
    
    cost_matrix = np.maximum(0.0, np.maximum(term1, term2))

    # 5. Apply Time Weighting
    i_idx = np.arange(n)[:, np.newaxis]
    t_idx = np.arange(n)[np.newaxis, :]
    weights = (t_idx - i_idx + 1)
    
    # 6. Filter Valid Steps (i <= t)
    mask = i_idx <= t_idx
    
    total_spec = np.sum(cost_matrix * weights * mask)
    
    return total_spec / n


@_base_docstring
def spec(
    df: pd.DataFrame,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "demand",
    cutoff_col: str = "cutoff",
    a1: float = 1.0,
    a2: float = 1.0,
) -> pd.DataFrame:
    """Stock-keeping-oriented Prediction Error Costs (SPEC)"""
    
    #  Initial Calculation (Result per ID per Cutoff)
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    
    def _group_apply(sub_df):
        y_true = sub_df[target_col].values
        res = {c: sub_df[c].iloc[0] for c in group_cols}
        for model in models:
            y_pred = sub_df[model].values
            res[model] = _spec_helper(y_true, y_pred, a1, a2)
        return pd.Series(res)

    # Execute GroupBy -> Apply
    results_df = df.groupby(group_cols).apply(_group_apply).reset_index(drop=True)
    
    # 2. Aggregate over cutoffs (Result per ID)
    if cutoff_col in results_df.columns:
        results_df = results_df.groupby(id_col)[models].mean().reset_index()

    return results_df.sort_values(id_col)


def scaled_spec(
    df: pd.DataFrame,
    models: List[str],
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "date",
) -> pd.DataFrame:
    """
    Compute the scaled SPEC (Stock-keeping-oriented Prediction Error Costs)
      metric for one or more models. This involves calculating the SPEC for each model
    and scaling it by the average scale of the training data,
    yielding a scale-independent measure that can be aggregated across series.

    Args:
        df (pandas or polars DataFrame): Input dataframe with id, actuals and predictions.
        models (list of str): Columns that identify the models predictions.
        train_df (pandas or polars DataFrame): Training dataframe with id and actual values. Must be sorted by time.
        id_col (str, optional): Column that identifies each serie. Defaults to 'unique_id'.
        target_col (str, optional): Column that contains the target. Defaults to 'y'.
        cutoff_col (str, optional): Column that identifies the cutoff point for each forecast cross-validation fold. Defaults to 'cutoff'.

    Returns:
        pandas or polars DataFrame: dataframe with one row per id and one column per model.    
    """
    train_df = _create_train_with_cutoffs(train_df=train_df, df=df, id_col=id_col, time_col=time_col, cutoff_col=cutoff_col)
    scales = _aggregate_column(
        df=train_df,
        target_col=target_col,
        id_col=id_col,
        cutoff_col=cutoff_col,
        agg="mean",
    )
    raw = spec(df=df, models=models, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col, a1=1.0, a2=1.0)

    # Calculate the in sample mean for each series
    scale_means = (
        scales.groupby(id_col)["scale"]
        .mean()
        .reset_index()
        .rename(columns={"scale": "in_sample_mean"})
    )

    # Divide the raw SPEC by the in-sample means of each series
    result = raw.merge(scale_means, on=id_col, how="left")
    for model in models:
        result[model] = result[model] / result["in_sample_mean"]
    result = result.drop(columns=["in_sample_mean"])

    return result.sort_values(id_col)



def _apis_helper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the absolute sum of cumulative forecast errors.
    """
    if len(y_true) == 0:
        return 0.0
    
    cumulative_errors = np.cumsum(y_true - y_pred)
    return np.abs(np.sum(cumulative_errors))


@_base_docstring
def apis(
    df: pd.DataFrame,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> pd.DataFrame:
    """Absolute Periodate in Stock (Unscaled)"""
    
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    
    def _group_apply(sub_df):
        y_true = sub_df[target_col].values
        res = {c: sub_df[c].iloc[0] for c in group_cols}
        for model in models:
            y_pred = sub_df[model].values
            res[model] = _apis_helper(y_true, y_pred)
        return pd.Series(res)

    results_df = df.groupby(group_cols).apply(_group_apply).reset_index(drop=True)
    
    if cutoff_col in results_df.columns:
        results_df = results_df.groupby(id_col)[models].mean().reset_index()

    return results_df.sort_values(id_col)


def sapis(
    df: pd.DataFrame,
    models: List[str],
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "date",
) -> pd.DataFrame:
    """
    Scaled Absolute Periodate in Stock (sAPIS).
    Calculated as APIS / mean(training_actuals).
    """
    train_df = _create_train_with_cutoffs(train_df=train_df, df=df, id_col=id_col, time_col=time_col, cutoff_col=cutoff_col)
    
    scales = _aggregate_column(
        df=train_df,
        target_col=target_col,
        id_col=id_col,
        cutoff_col=cutoff_col,
        agg="mean"
    )
    
    raw = apis(df=df, models=models, id_col=id_col, target_col=target_col, cutoff_col=cutoff_col)

    scale_means = (
        scales.groupby(id_col)["scale"]
        .mean()
        .reset_index()
        .rename(columns={"scale": "in_sample_mean"})
    )

    result = raw.merge(scale_means, on=id_col, how="left")
    for model in models:
        result[model] = result[model] / result["in_sample_mean"]
    
    result = result.drop(columns=["in_sample_mean"])

    return result.sort_values(id_col)




from typing import List
import numpy as np
import pandas as pd

def _in_sample_mean_scale(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    cutoff_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: [id_col, in_sample_mean]
    computed as mean over cutoffs of mean(train_y in each cutoff window).
    """
    train_df = _create_train_with_cutoffs(
        train_df=train_df,
        df=df,
        id_col=id_col,
        time_col=time_col,
        cutoff_col=cutoff_col,
    )

    scales = _aggregate_column(
        df=train_df,
        target_col=target_col,
        id_col=id_col,
        cutoff_col=cutoff_col,
        agg="mean",
    )  # expected to return columns [id_col, cutoff_col, "scale"]

    scale_means = (
        scales.groupby(id_col)["scale"]
        .mean()
        .reset_index()
        .rename(columns={"scale": "in_sample_mean"})
    )
    return scale_means


def scaled_mae(
    df: pd.DataFrame,
    models: List[str],
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "date",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Scaled MAE = MAE / mean(train_y)  (mean computed like in scaled_spec).
    """
    raw = mae(df=df, models=models, id_col=id_col, target_col=target_col)

    scale_means = _in_sample_mean_scale(
        df=df,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        time_col=time_col,
    )

    result = raw.merge(scale_means, on=id_col, how="left")
    denom = result["in_sample_mean"].abs().clip(lower=eps)

    for model in models:
        result[model] = result[model] / denom

    return result.drop(columns=["in_sample_mean"]).sort_values(id_col)


def scaled_rmse(
    df: pd.DataFrame,
    models: List[str],
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "date",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Scaled RMSE = RMSE / mean(train_y)  (mean computed like in scaled_spec).
    """
    raw = rmse(df=df, models=models, id_col=id_col, target_col=target_col)

    scale_means = _in_sample_mean_scale(
        df=df,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        time_col=time_col,
    )

    result = raw.merge(scale_means, on=id_col, how="left")
    denom = result["in_sample_mean"].abs().clip(lower=eps)

    for model in models:
        result[model] = result[model] / denom

    return result.drop(columns=["in_sample_mean"]).sort_values(id_col)


def scaled_bias(
    df: pd.DataFrame,
    models: List[str],
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
    time_col: str = "date",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Scaled Bias = Bias / mean(train_y)  (mean computed like in scaled_spec).
    Bias is (pred - actual), so sign is preserved after scaling.
    """
    raw = bias(df=df, models=models, id_col=id_col, target_col=target_col)

    scale_means = _in_sample_mean_scale(
        df=df,
        train_df=train_df,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        time_col=time_col,
    )

    result = raw.merge(scale_means, on=id_col, how="left")
    denom = result["in_sample_mean"].abs().clip(lower=eps)

    for model in models:
        result[model] = result[model] / denom

    return result.drop(columns=["in_sample_mean"]).sort_values(id_col)
