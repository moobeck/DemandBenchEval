import logging
import wandb
from src.configurations.wandb import WandbConfig


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

    def _are_config_set(self):
        """
        Check if the W&B configuration is set.
        """
        return (
            self.config.api_key is not None
            or self.config.entity is not None
            or self.config.project is not None
        )

    def login(self):
        if self.config.api_key:
            wandb.login(key=self.config.api_key)
        else:
            logging.info("No W&B API key provided; using default authentication.")

    def start_run(self):
        if self._are_config_set():

            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                config=self.public_config,
                tags=self.tags,
            )
            return self.run

        logging.warning("W&B configuration is not set. Run will not be logged to W&B.")

    def log_artifact(self, name: str, filepath: str, type_: str):
        art = wandb.Artifact(name, type=type_)
        art.add_file(filepath)
        self.run.log_artifact(art) if self.run else None

    def log_metrics(self, metrics: dict):
        self.run.log(metrics) if self.run else None

    def log_image(self, alias: str, filepath: str):
        self.run.log({alias: wandb.Image(filepath)}) if self.run else None

    def finish(self):
        if self.run:
            self.run.finish()
