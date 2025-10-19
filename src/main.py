import argparse
import logging
import yaml
from typing import Any
from src.configurations.core.file_path import FilePathConfig
from src.configurations.data.datasets import DatasetConfig
from src.configurations.core.system import SystemConfig
from src.configurations.data.input_column import InputColumnConfig
from src.configurations.data.forecast_column import ForecastColumnConfig
from src.configurations.evaluation.cross_validation import (
    CrossValidationConfig,
    CrossValWindowConfig,
    CrossValDatasetConfig,
)
from src.configurations.forecasting.forecasting import ForecastConfig
from src.configurations.evaluation.metrics import MetricConfig
from src.configurations.forecasting.quantile import QuantileConfig
from src.configurations.utils.enums import (
    ModelName,
    MetricName,
    DatasetName,
    Framework,
    TargetScalerType,
    FileFormat,
)
from src.configurations.utils.wandb import WandbConfig
from src.configurations.core.global_cfg import GlobalConfig
from src.configurations.data.preprocessing import PreprocessingConfig
from src.utils.wandb_orchestrator import WandbOrchestrator
from src.utils.dataframe import DataFrameHandler
from src.utils.system_settings import SystemSettings
from src.dataset.dataset_factory import DatasetFactory
from src.preprocessing.preprocessor import Preprocessor
from src.utils.statistics import SKUStatistics
from src.forecasting.cross_validation.cross_validation import CrossValidator
from src.forecasting.evaluation.evaluation import Evaluator, EvaluationPlotter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline: preprocess → train → cross‐validate → evaluate"
    )
    parser.add_argument(
        "-c",
        "--public-config",
        dest="public_config",
        help="Path to public YAML config file (no secrets).",
        default="config/public/config.yaml",
    )
    parser.add_argument(
        "-s",
        "--private-config",
        dest="private_config",
        help="Path to private YAML config file (with secrets).",
        default="config/private/config.yaml",
    )
    return parser.parse_args()


def load_config_dict(path) -> dict[str, Any]:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning(f"Config file not found: {path}. Returning empty config.")
        return {}


def build_config(public_config: dict, private_config: dict) -> GlobalConfig:

    system = public_config.get("system", {})
    if not system:
        logging.warning("No system settings provided in the public config.")

    filepaths = public_config.get("filepaths", {})
    if not filepaths:
        logging.warning("No file paths provided in the public config.")
    dataset_names = public_config.get("datasets", [])
    input_colums = public_config.get("input_columns", {})
    if not input_colums:
        logging.warning("No input columns provided in the public config.")
    preprocessing = public_config.get("preprocessing", {})
    if not preprocessing:
        logging.warning("No preprocessing settings provided in the public config.")
    forecast_columns = public_config.get("forecast_columns", {})
    if not forecast_columns:
        logging.warning("No forecast columns provided in the public config.")
    cross_validation_data = public_config.get("cross_validation", {})
    if not cross_validation_data:
        logging.warning("No cross-validation settings provided in the public config.")
    forecast = public_config.get("forecast", {})
    if not forecast:
        logging.warning("No forecast settings provided in the public config.")
    metrics = public_config.get("metrics", {})
    if not metrics:
        logging.warning("No metrics settings provided in the public config.")
    lags = public_config.get("lags", [])
    if not lags:
        logging.warning("No lags provided in the public config.")

    log_wandb = public_config.get("log_wandb", False)
    wandb: dict = private_config.get("wandb", {})  #
    if not wandb and log_wandb:
        logging.warning("No W&B settings provided in the private config.")
    wandb.update({"log_wandb": log_wandb})

    return GlobalConfig(
        system=SystemConfig(
            GPU=system.get("GPU", 0), RANDOM_SEED=system.get("RANDOM_SEED", 42)
        ),
        filepaths=FilePathConfig(
            processed_data_dir=filepaths.get("processed_data_dir", "data/processed"),
            sku_stats_dir=filepaths.get("sku_stats_dir", "data/sku_stats"),
            cv_results_dir=filepaths.get("cv_results_dir", "data/cv_results"),
            eval_results_dir=filepaths.get("eval_results_dir", "data/eval_results"),
            eval_plots_dir=filepaths.get("eval_plots_dir", "data/eval_plots"),
            file_format=FileFormat[filepaths.get("file_format", "FEATHER")],
        ),
        datasets=DatasetConfig(names=[DatasetName[name] for name in dataset_names]),
        input_columns=InputColumnConfig(
            time_series_index=input_colums["time_series_index"],
            date=input_colums["date"],
            target=input_colums["target"],
            frequency=input_colums["frequency"],
        ),
        preprocessing=PreprocessingConfig(
            target_transform=TargetScalerType[preprocessing["target_transform"]],
        ),
        forecast_columns=ForecastColumnConfig(
            time_series_index=forecast_columns["time_series_index"],
            date=forecast_columns["date"],
            product_index=forecast_columns.get("product_index", "productID"),
            store_index=forecast_columns.get("store_index", "storeID"),
            target=forecast_columns["target"],
            cutoff=forecast_columns["cutoff"],
        ),
        cross_validation=CrossValidationConfig(
            data={
                DatasetName[name]: CrossValDatasetConfig(
                    test=CrossValWindowConfig(
                        n_windows=cv["test"]["n_windows"],
                        step_size=cv["test"]["step_size"],
                        refit=cv["test"]["refit"],
                    ),
                    val=CrossValWindowConfig(
                        n_windows=cv["val"]["n_windows"],
                        step_size=cv["val"]["step_size"],
                        refit=cv["val"]["refit"],
                    ),
                )
                for name, cv in cross_validation_data.items()
            }
        ),
        forecast=ForecastConfig(
            names=[ModelName[name] for name in forecast["models"]],
            model_config={
                Framework[fw]: forecast["model_config"][fw]
                for fw in forecast["model_config"]
            },
            lags_config=lags,
        ),
        metrics=MetricConfig(
            names=[MetricName[name] for name in public_config["metrics"]["metrics"]],
            seasonality=public_config["metrics"].get("seasonality", None),
            quantiles=QuantileConfig(**public_config["metrics"].get("quantiles", None)),
        ),
        wandb=WandbConfig(
            api_key=(wandb.get("api_key") if wandb else None),
            entity=(wandb.get("entity") if wandb else None),
            project=(wandb.get("project", "bench-forecast") if wandb else None),
            log_wandb=wandb.get("log_wandb", False) if wandb else False,
        ),
    )


def main():

    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    public_cfg_dict = load_config_dict(args.public_config)
    private_cfg_dict = load_config_dict(args.private_config)

    cfg = build_config(public_cfg_dict, private_cfg_dict)

    # System settings
    system_settings = SystemSettings(cfg.system)
    system_settings.configure_environment()
    system_settings.set_seed()
    # Ensure directories exist
    cfg.filepaths.ensure_directories_exist()

    # W&B orchestration
    wandb_orchestrator = WandbOrchestrator(cfg.wandb, public_cfg_dict)
    wandb_orchestrator.login()
    wandb_orchestrator.start_run()

    for dataset_name in cfg.datasets.names:

        # Load dataset
        dataset = DatasetFactory.create_dataset(dataset_name)
        cfg.set_dataset(dataset_name, dataset)

        # Preprocessing
        prep = Preprocessor(
            dataset,
            cfg.input_columns,
            cfg.preprocessing,
            cfg.forecast_columns,
            cfg.forecast,
            cfg.cross_validation,
        )
        prep.merge()
        prep.remove_skus(skus="not_at_min_date")
        df = prep.prepare_nixtla()

        # Calculate SKU statistics
        sku_stats = SKUStatistics(
            df=df,
            forecast_columns=cfg.forecast_columns,
            cross_validation=cfg.cross_validation,
            freq=cfg.forecast.freq,
            forecast=cfg.forecast,
        )
        sku_stats_df = sku_stats.compute_statistics()
        DataFrameHandler.write_dataframe(
            sku_stats_df, cfg.filepaths.sku_stats, cfg.filepaths.file_format
        )

        df = prep.preprocess_data(df)

        DataFrameHandler.write_dataframe(
            df, cfg.filepaths.processed_data, cfg.filepaths.file_format
        )

        # Cross-validation
        cross_validator = CrossValidator(cfg.forecast, cfg.forecast_columns)
        cv_df = cross_validator.cross_validate(
            df=df,
            cv_config=cfg.cross_validation,
        )

        DataFrameHandler.write_dataframe(
            cv_df, cfg.filepaths.cv_results, cfg.filepaths.file_format
        )

        # Evaluation
        evaluator = Evaluator(cfg.metrics, cfg.forecast_columns)
        eval_df = evaluator.evaluate(cv_df, train_df=df)
        metrics_summary = evaluator.summarize_metrics(eval_df)
        wandb_orchestrator.log_metrics(metrics_summary, dataset_name)

        DataFrameHandler.write_dataframe(
            eval_df, cfg.filepaths.eval_results, cfg.filepaths.file_format
        )

        wandb_orchestrator.log_artifact(
            name="evaluation-results",
            filepath=cfg.filepaths.eval_results,
            type_="results",
        )

        fig = EvaluationPlotter(
            eval_df,
            forecast_columns=cfg.forecast_columns,
            metric_config=cfg.metrics,
            ylim=(-0.5, 4),
        ).plot_error_distributions()


        fig.savefig(cfg.filepaths.eval_plots, dpi=300, bbox_inches="tight")


        wandb_orchestrator.log_image(
            alias="error_distribution_plot", filepath=cfg.filepaths.eval_plots
        )

        wandb_orchestrator.finish()


if __name__ == "__main__":
    main()
