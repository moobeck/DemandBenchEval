import logging
from src.configurations.core.global_cfg import GlobalConfig
from src.utils.wandb_orchestrator import WandbOrchestrator
from src.utils.dataframe import DataFrameHandler
from src.utils.system_settings import SystemSettings
from src.dataset.dataset_factory import DatasetFactory
from src.preprocessing.preprocessor import Preprocessor
from src.utils.statistics import SKUStatistics
from src.forecasting.cross_validation.cross_validation import CrossValidator
from src.forecasting.evaluation.evaluation import Evaluator, EvaluationPlotter
from src.utils.args_parser import ArgParser


def main():

    logging.basicConfig(level=logging.INFO)

    args = ArgParser.parse()
    public_cfg_dict = GlobalConfig.load_dict(args.public_config)
    private_cfg_dict = GlobalConfig.load_dict(args.private_config)

    cfg = GlobalConfig.build(public_cfg_dict, private_cfg_dict)

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
        df = prep.prepare_forecasting_data()

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
