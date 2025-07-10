import argparse
import logging
import yaml
import random
import numpy as np
from typing import Any
from src.configurations.file_path import FilePathConfig
from src.configurations.datasets import DatasetConfig
from src.configurations.input_column import InputColumnConfig
from src.configurations.forecast_column import ForecastColumnConfig
from src.configurations.cross_validation import CrossValidationConfig, CrossValWindowConfig, CrossValDatasetConfig
from src.configurations.forecasting import ForecastConfig
from src.configurations.metrics import MetricConfig
from src.configurations.enums import ModelName, MetricName, DatasetName, Framework, TargetScalerType
from src.configurations.wandb import WandbConfig
from src.configurations.global_cfg import GlobalConfig
from src.configurations.preprocessing import PreprocessingConfig
from src.utils.wandb_orchestrator import WandbOrchestrator
from src.dataset.dataset_factory import DatasetFactory
from src.preprocessing.nixtla_preprocessor import NixtlaPreprocessor
from src.forecasting.training import ForecastTrainer
from src.forecasting.evaluation import Evaluator, EvaluationPlotter


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


def set_seed(seed: int):
    """
    Set seeds for all relevant random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config(public_config: dict, private_config: dict) -> GlobalConfig:

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

    log_wandb = public_config.get("log_wandb", False)
    wandb: dict = private_config.get("wandb", {})  #
    if not wandb and log_wandb:
        logging.warning("No W&B settings provided in the private config.")
    wandb.update({"log_wandb": log_wandb})

    seed = public_config.get("seed", None)
    if not seed:
        logging.warning("No seed provided in the public config. Using default seed 42.")
        seed = 42

    return GlobalConfig(
        filepaths=FilePathConfig(
            eval_results_dir=filepaths.get("eval_results_dir", "results/eval_results"),
            eval_plots_dir=filepaths.get("eval_plots_dir", "results/eval_plots"),
        ),
        datasets=DatasetConfig(names=[DatasetName[name] for name in dataset_names]),
        input_columns=InputColumnConfig(
            sku_index=input_colums["sku_index"],
            date=input_colums["date"],
            target=input_colums["target"],
            frequency=input_colums["frequency"],
        ),
        preprocessing=PreprocessingConfig(
            target_transform=TargetScalerType[preprocessing["target_transform"]],
        ),
        forecast_columns=ForecastColumnConfig(
            sku_index=forecast_columns["sku_index"],
            date=forecast_columns["date"],
            target=forecast_columns["target"],
            cutoff=forecast_columns["cutoff"],
            base_exogenous=[col for col in forecast_columns["exog_vars"]],
            categorical=[col for col in forecast_columns.get("categorical", [])],
            static=[col for col in forecast_columns["static"]],
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
        ),
        metrics=MetricConfig(
            names=[MetricName[name] for name in public_config["metrics"]["metrics"]],
            seasonality=public_config["metrics"].get("seasonality", None),
        ),
        wandb=WandbConfig(
            api_key=(wandb.get("api_key") if wandb else None),
            entity=(wandb.get("entity") if wandb else None),
            project=(wandb.get("project", "bench-forecast") if wandb else None),
            log_wandb=wandb.get("log_wandb", False) if wandb else False,
        ),
        seed=seed,
    )


def set_seed(seed: int):
    """
    Set seeds for all relevant random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)


def main():

    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    public_cfg_dict = load_config_dict(args.public_config)
    private_cfg_dict = load_config_dict(args.private_config)

    cfg = build_config(public_cfg_dict, private_cfg_dict)

    # W&B orchestration
    wandb_orchestrator = WandbOrchestrator(cfg.wandb, public_cfg_dict)
    wandb_orchestrator.login()
    wandb_orchestrator.start_run()

    # initialize
    set_seed(cfg.seed)

    for dataset_name in cfg.datasets.names:

        # 1) Load dataset
        dataset = DatasetFactory.create_dataset(dataset_name)
        cfg.set_dataset(dataset_name, dataset)

        # 2) Preprocessing
        prep = NixtlaPreprocessor(
            dataset, cfg.input_columns, cfg.preprocessing,
            cfg.forecast_columns, cfg.forecast, cfg.cross_validation
        )
        prep.merge()
        prep.remove_skus(skus="not_at_min_date")
        df = prep.prepare_nixtla()
        df = prep.preprocess_data(df)

        # save df as feather file
        df.to_feather(cfg.filepaths.eval_results.replace(".feather", "processed_dataset.feather"))

        # 3) Cross-validation
        trainer = ForecastTrainer(cfg.forecast, cfg.forecast_columns)
        cv_df = trainer.cross_validate(
            df=df,
            cv_config=cfg.cross_validation,
        )

        # 4) Evaluation
        evaluator = Evaluator(cfg.metrics, cfg.forecast_columns)
        eval_df = evaluator.evaluate(cv_df, train_df=df)
        metrics_summary = evaluator.summarize_metrics(eval_df)
        wandb_orchestrator.log_metrics(metrics_summary, dataset_name)

        # 4) Save & log results
        eval_df.to_feather(cfg.filepaths.eval_results)
        wandb_orchestrator.log_artifact(
            name="evaluation-results",
            filepath=cfg.filepaths.eval_results,
            type_="results",
        )

        # 5) Plot & log
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
