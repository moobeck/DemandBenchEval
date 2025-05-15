import argparse
import logging
import yaml
import random
import numpy as np
from src.configurations.file_path import FilePathConfig
from src.configurations.input_column import InputColumnConfig
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.cross_validation import CrossValidationConfig
from src.forecasting.training import ForecastTrainer
from src.forecasting.evaluation import Evaluator, EvaluationPlotter
from src.configurations.forecasting import ForecastConfig
from src.configurations.metrics import MetricConfig
from src.configurations.enums import ModelName, MetricName
from src.preprocessing.nixtla_preprocessor import NixtlaPreprocessor


def parse_args():
    p = argparse.ArgumentParser(
        description="Full pipeline: preprocess → train → cross‐validate → evaluate"
    )
    p.add_argument(
        "--config",
        "-c",
        help="Path to a YAML file with all settings.",
        default="config/config.yaml",
    )

    return p.parse_args()


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    # map hyphens to underscores for all keys
    return {k.replace("-", "_"): v for k, v in raw.items()}


def set_seed(seed: int):
    """
    Set seeds for all relevant random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)


def build_configs(
    args_dict,
) -> tuple[
    FilePathConfig,
    InputColumnConfig,
    ForecastColumnConfig,
    CrossValidationConfig,
    ForecastConfig,
    MetricConfig,
    int,
]:

    filepaths: dict[str, str] = args_dict["filepaths"]
    input_columns: dict[str, str] = args_dict["input_columns"]
    forecast_columns: dict[str, str] = args_dict["forecast_columns"]
    forecast: dict[str, str] = args_dict["forecast"]
    cross_validation: dict[str, str] = args_dict["cross_validation"]
    metrics: dict[str, str] = args_dict["metrics"]

    # file paths
    file_cfg = FilePathConfig(
        train_data_features=filepaths["train_feat"],
        val_data_features=filepaths["val_feat"],
        train_data_target=filepaths["train_tgt"],
        val_data_target=filepaths["val_tgt"],
        preprocessed_data=filepaths["preprocessed"],
        eval_results=filepaths["eval_results"],
        eval_plots=filepaths["eval_plots"],
    )

    # input cols
    incol_cfg = InputColumnConfig(
        date=input_columns["date"],
        dp_index=input_columns["dp_index"],
        sku_index=input_columns["sku_index"],
        target=input_columns["target"],
    )

    # forecast cols (static = subset of exogenous)
    focol_cfg = ForecastColumnConfig(
        date=forecast_columns["date"],
        sku_index=forecast_columns["sku_index"],
        target=forecast_columns["target"],
        exogenous=forecast_columns["exog_vars"],
        static=forecast_columns["static"],
    )

    # forecasting model settings
    fcast_cfg = ForecastConfig(
        names=[ModelName[name] for name in forecast["models"]],
        freq=forecast["freq"],
        season_length=forecast["season_length"],
        horizon=forecast["horizon"],
        lags=forecast["lags"],
        date_features=forecast["date_features"],
    )

    # cross-validation settings
    cv_cfg = CrossValidationConfig(
        cv_windows=cross_validation["cv_windows"],
        step_size=cross_validation["step_size"],
    )

    # metrics configuration
    met_cfg = MetricConfig(
        names=[MetricName[name] for name in metrics["metrics"]],
        seasonality=metrics["seasonality"],
    )

    seed: int = args_dict["seed"]

    return (file_cfg, incol_cfg, focol_cfg, cv_cfg, fcast_cfg, met_cfg, seed)


def set_seed(seed: int):
    """
    Set seeds for all relevant random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    # flatten cfg keys to match arg names
    (file_cfg, incol_cfg, focol_cfg, cv_cfg, fcast_cfg, met_cfg, seed) = build_configs(
        cfg
    )

    set_seed(seed)


    # set up logging
    logging.basicConfig(level=logging.INFO)

    # 1) Preprocessing
    prep = NixtlaPreprocessor(file_cfg, incol_cfg, focol_cfg)
    prep.load_data()
    prep.merge()
    # optionally parameterize these SKUs too
    prep.remove_skus([2254, 2255, 2256])
    df = prep.prepare_nixtla()
    df.to_feather(file_cfg.preprocessed_data)

    # 2) Cross‐validation
    trainer = ForecastTrainer(fcast_cfg, focol_cfg)
    cv_df = trainer.cross_validate(
        df=df,
        n_windows=cv_cfg.cv_windows,
        step_size=cv_cfg.step_size,
    )

    # 3) Evaluation
    evaluator = Evaluator(met_cfg, focol_cfg)
    eval_df = evaluator.evaluate(cv_df, train_df=df)

    # 4) Save results
    logging.info(f"Saving evaluation results to {file_cfg.eval_results}")
    eval_df.to_feather(file_cfg.eval_results)

    # 5) Plotting
    plotter = EvaluationPlotter(
        eval_df,
        forecast_columns=focol_cfg,
        metric_config=met_cfg,
        ylim=(-0.5, 4),
    )
    # Save plots
    plotter.plot_error_distributions().savefig(
        file_cfg.eval_plots, dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
