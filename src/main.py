#!/usr/bin/env python3
import argparse
import logging
import yaml
from src.configurations.file_path import FilePathConfig
from src.configurations.input_column import InputColumnConfig
from src.configurations.forecast_column import ForecastColumnConfig
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



def build_configs(args_dict):
    # file paths
    file_cfg = FilePathConfig(
        train_data_features=args_dict["train_feat"],
        val_data_features=args_dict["val_feat"],
        train_data_target=args_dict["train_tgt"],
        val_data_target=args_dict["val_tgt"],
        eval_results=args_dict["eval_results"],
        eval_plots=args_dict["eval_plots"],
    )
    # input cols
    input_cfg = InputColumnConfig(
        date=args_dict["date_col"],
        dp_index=args_dict["dp_index"],
        sku_index=args_dict["sku_index"],
        target=args_dict["target_col"],
        exogenous=args_dict["exog_vars"],
    )
    # forecast cols (static = subset of exogenous)
    forecast_cfg = ForecastColumnConfig(
        date=args_dict["date_col"],
        sku_index=args_dict["sku_index"],
        target=args_dict["target_col"],  # or rename?
        exogenous=args_dict["exog_vars"],
        static=args_dict["exog_vars"][:3],  # example
    )
    # forecasting model settings
    fc = ForecastConfig(
        names=[ModelName[name] for name in args_dict["models"]],
        freq=args_dict["freq"],
        season_length=args_dict["season_length"],
        horizon=args_dict["horizon"],
        lags=args_dict["lags"],
        date_features=args_dict["date_features"],
    )
    # metrics
    mc = MetricConfig(
        names=[MetricName[name] for name in args_dict["metrics"]],
        seasonality=args_dict["seasonality"],
    )
    return file_cfg, input_cfg, forecast_cfg, fc, mc


def main():
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        # flatten cfg keys to match arg names
        file_cfg, in_cfg, fo_cfg, fcast_cfg, met_cfg = build_configs(cfg)
    else:
        # convert Namespace to dict
        dd = vars(args)
        file_cfg, in_cfg, fo_cfg, fcast_cfg, met_cfg = build_configs(dd)

    # set up logging
    logging.basicConfig(level=logging.INFO)

    # 1) Preprocessing
    prep = NixtlaPreprocessor(file_cfg, in_cfg, fo_cfg)
    prep.load_data()
    prep.merge()
    # optionally parameterize these SKUs too
    prep.remove_skus([2254, 2255, 2256])
    df = prep.prepare_nixtla()

    # 2) Cross‐validation
    trainer = ForecastTrainer(fcast_cfg, fo_cfg)
    cv_df = trainer.cross_validate(
        df=df,
        n_windows=args.cv_windows,
        step_size=args.step_size,
    )

    # 3) Evaluation
    evaluator = Evaluator(met_cfg, fo_cfg)
    eval_df = evaluator.evaluate(cv_df, train_df=df)

    # 4) Save results
    logging.info(f"Saving evaluation results to {file_cfg.eval_results}")
    eval_df.to_feather(file_cfg.eval_results)

    # 5) Plotting
    plotter = EvaluationPlotter(
        eval_df,
        forecast_columns=fo_cfg,
        metric_config=met_cfg,
        ylim=(-0.5, 4),
    )
    # Save plots
    plotter.plot_error_distributions().savefig(
        file_cfg.eval_plots, dpi=300, bbox_inches="tight"
        )



    
    


if __name__ == "__main__":
    main()
