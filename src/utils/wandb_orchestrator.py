import logging
import os
from pathlib import Path
import wandb
from src.configurations.utils.wandb import WandbConfig
from src.configurations.utils.enums import DatasetName
from src.configurations.utils.enums import Framework


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

    @staticmethod
    def _load_key_from_envfile() -> str | None:
        """
        Look for WANDB_KEY in a .env file at repo root (three levels up from this file).
        Supports simple KEY=VALUE lines without interpolation.
        """
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.is_file():
            return None
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("WANDB_KEY="):
                    return line.split("WANDB_KEY=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            return None
        return None

    def login(self):
        # Priority: env var (WANDB_KEY), then config, then .env file fallback.
        key = os.getenv("WANDB_KEY") or self.config.api_key or self._load_key_from_envfile()
        if key:
            wandb.login(key=key)
        else:
            logging.info(
                "No W&B API key provided via env, config, or .env; using default authentication."
            )

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

    @staticmethod
    def _extract_best_config(results):
        """
        Attempt to extract the best hyperparameter config from different backends.
        Supports Ray Tune (ResultGrid), Optuna (Study), and simple mappings.
        """
        if results is None:
            return None

        # Ray Tune's ResultGrid
        if hasattr(results, "get_best_result"):
            try:
                best = results.get_best_result()
                return getattr(best, "config", None)
            except Exception:
                return None

        # Optuna Study
        if hasattr(results, "best_trial"):
            try:
                best_trial = results.best_trial
                if hasattr(best_trial, "params"):
                    return best_trial.params
                return getattr(best_trial, "config", None)
            except Exception:
                return None

        # Fallback to common attributes
        for attr in ("best_params", "best_config", "config"):
            cfg = getattr(results, attr, None)
            if cfg:
                return cfg

        return None

    def maybe_log_hyperparameters(self, frameworks: dict, task_name: str):
        if self.run:
            if Framework.NEURAL in frameworks and frameworks[Framework.NEURAL]:
                neural_engine = frameworks[Framework.NEURAL]

                models = getattr(neural_engine, "models", [])

                hyperparams = {}
                for model in models:
                    best_cfg = self._extract_best_config(
                        getattr(model, "results", None)
                    )
                    if best_cfg:
                        alias = getattr(model, "alias", str(model))
                        hyperparams[alias] = best_cfg

                if hyperparams:
                    logging.info(
                        f"Logging hyperparameters for task {task_name}: {hyperparams}"
                    )
                    wandb.config.update(
                        {f"{task_name}_neural_hyperparameters": hyperparams}
                    )
            else:
                logging.info(
                    f"No neural framework found or no models defined for task {task_name}. Skipping hyperparameter logging."
                )

    def finish(self):
        if self.run:
            self.run.finish()
