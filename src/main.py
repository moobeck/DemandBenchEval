#!/usr/bin/env python3
import argparse
import logging
import yaml
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
        "--config", "-c",
        help="Path to a YAML file with all settings. If given, CLI args are ignored.",
        default=None
    )
    # File paths
    p.add_argument("--train-feat", required=False, help="train data features (.feather)")
    p.add_argument("--val-feat",   required=False, help="val data features (.feather)")
    p.add_argument("--train-tgt",  required=False, help="train data target (.feather)")
    p.add_argument("--val-tgt",    required=False, help="val data target (.feather)")
    p.add_argument(
        "--eval-results", required=False,
        help="Path to save evaluation results (.feather)"
    )
    # Columns
    p.add_argument("--date-col",     required=False, help="name of the date column")
    p.add_argument("--dp-index",     required=False, help="name of the dp index column")
    p.add_argument("--sku-index",    required=False, help="name of the sku index column")
    p.add_argument("--target-col",   required=False, help="name of the target column")
    p.add_argument(
        "--exog-vars", nargs="+", required=False,
        help="list of exogenous variable names"
    )
    # Forecast settings
    p.add_argument(
        "--models", nargs="+", default=["ETS", "LGBM"],
        choices=[m.name for m in ModelName],
        help="which models to train"
    )
    p.add_argument("--freq",          default="D")
    p.add_argument("--season-length", type=int, default=7)
    p.add_argument("--horizon",       type=int, default=14)
    p.add_argument("--lags", nargs="+", type=int, default=[1, 7])
    p.add_argument("--date-features", nargs="+", default=["dayofweek", "month"])
    # CV settings
    p.add_argument("--cv-windows", type=int, default=1)
    p.add_argument("--step-size",  type=int, default=1)
    # Metrics
    p.add_argument(
        "--metrics", nargs="+", default=["MASE", "MSSE"],
        choices=[m.name for m in MetricName]
    )
    p.add_argument("--seasonality", type=int, default=7)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    # map hyphens to underscores for all keys
    return {k.replace("-", "_"): v for k, v in raw.items()}



def build_configs(args_dict) -> tuple[FilePathConfig, InputColumnConfig, ForecastColumnConfig, CrossValidationConfig, ForecastConfig, MetricConfig]:

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
    return file_cfg, incol_cfg, focol_cfg, cv_cfg, fcast_cfg, met_cfg


def main():
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        # flatten cfg keys to match arg names
        file_cfg, incol_cfg, focol_cfg, cv_cfg, fcast_cfg, met_cfg = build_configs(cfg)
    else:
        # convert Namespace to dict
        dd = vars(args)
        file_cfg, incol_cfg, focol_cfg, fcast_cfg, met_cfg = build_configs(dd)

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
