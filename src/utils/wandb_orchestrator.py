import logging
import wandb
from src.configurations.utils.wandb import WandbConfig
from src.configurations.utils.enums import DatasetName


class WandbOrchestrator:
    """
    Encapsulates W&B initialization, logging, and finalization.
    """

    def __init__(self, config: WandbConfig, public_config: dict):
        self.config = config
        self.public_config = public_config
        self.tags = [
            str(m) for m in public_config.get("forecast", {}).get("models", [])
        ]
        self.run = None

    def login(self):
        if self.config.api_key:
            wandb.login(key=self.config.api_key)
        else:
            logging.info("No W&B API key provided; using default authentication.")

    def start_run(self):
        if self.config.log_wandb:

            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                config=self.public_config,
                tags=self.tags,
            )
            return self.run

    def log_artifact(self, name: str, filepath: str, type_: str):

        if self.run:
            logging.info(f"Logging artifact: {name} of type {type_}")
            art = wandb.Artifact(name, type=type_)
            art.add_file(filepath)
            self.run.log_artifact(art) if self.run else None

    def log_metrics(self, metrics: dict, dataset_name: DatasetName):

        if self.run:

            data = {dataset_name.value: metrics}

            logging.info(f"Logging metrics: {metrics}")
            # Log metrics to W&B
            wandb.log(data) if self.run else None
            self.run.log(data) if self.run else None


    def finish(self):
        if self.run:
            self.run.finish()
