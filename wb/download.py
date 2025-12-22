from datetime import datetime, timezone
import os
import wandb
import logging



PROJECT = "bench-forecast"
TEAM = "d3_group"
PATH = f"{TEAM}/{PROJECT}"
CUTOFF = datetime(2025, 12, 10, 0, 0, 0, tzinfo=timezone.utc)

from enum import StrEnum


class ModelName(StrEnum):
    ARIMA = "arima"
    THETA = "theta"
    ETS = "ets"
    CES = "ces"
    CROSTON = "croston"
    LGBM = "lgbm"
    CATBOOST = "catboost"
    RF = "rf"
    TRANSFORMER = "transformer"
    MLP = "mlp"
    LSTM = "lstm"
    TIMESNET = "timesnet"
    FEDFORMER = "fedformer"
    TIDE = "tide"
    NHITS = "nhits"
    DEEPAR = "depar"
    NBEATS = "nbeats"
    BITCN = "bitcn"
    GRU = "gru"
    TCN = "tcn"
    TFT = "tft"
    PATCHTST = "patchtst"
    XLSTM = "xlstm"
    MOIRAI = "moirai"
    CHRONOS = "chronos"
    TABPFN = "tabpfn"



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Utils:
    @staticmethod
    def ensure_dir(dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)


class Run:
    def __init__(self, run: wandb.Run):
        self.run = run
        logger.info(
            "Initialized Run (run=%s, id=%s)",
            getattr(self.run, "name", "<unknown>"),
            getattr(self.run, "id", "<unknown>"),
        )

    @property
    def model_name(self) -> ModelName:
        tags = self.run.tags
        for tag in tags:
            if tag.lower() in ModelName:
                return ModelName(tag.lower())

    @property
    def artifacts(self) -> list[wandb.Artifact]:
        arts = self.run.logged_artifacts()
        logger.info(
            "Fetched artifacts for run (run=%s, id=%s, artifact_count=%d)",
            getattr(self.run, "name", "<unknown>"),
            getattr(self.run, "id", "<unknown>"),
            len(arts),
        )
        return (Artifact(artifact=a, model=self.model_name) for a in arts)


class ArtifactTypes:
    CV_RESULTS = "cv-results"
    EVAL_RESULTS = "evaluation-results"
    OTHER = "other"

    def from_str(type_str: str) -> str:
        if ArtifactTypes.CV_RESULTS in type_str:
            return ArtifactTypes.CV_RESULTS
        elif ArtifactTypes.EVAL_RESULTS in type_str:
            return ArtifactTypes.EVAL_RESULTS
        else:
            return ArtifactTypes.OTHER


class Artifact:
    def __init__(self, artifact: wandb.Artifact, model: ModelName):
        self.artifact = artifact
        self.local_dir = f"wb/artifacts/{self.type}/{model.value}"
        logger.info(
            "Initialized Artifact (artifact=%s, type=%s, local_dir=%s)",
            getattr(self.artifact, "name", "<unknown>"),
            self.type,
            self.local_dir,
        )

    @property
    def type(self) -> str:
        t = ArtifactTypes.from_str(self.artifact.name)
        logger.info("Resolved artifact type (artifact=%s -> type=%s)", self.artifact.name, t)
        return t

    @property
    def _is_of_interest(self) -> bool:
        is_interest = self.type in {ArtifactTypes.CV_RESULTS, ArtifactTypes.EVAL_RESULTS}
        logger.info(
            "Checked if artifact is of interest (artifact=%s, type=%s, is_of_interest=%s)",
            self.artifact.name,
            self.type,
            is_interest,
        )
        return is_interest

    def download_if_of_interest(self) -> None:
        logger.info("Attempting download_if_of_interest (artifact=%s)", self.artifact.name)
        if not self._is_of_interest:
            logger.info("Skipping download (artifact=%s) because it is not of interest", self.artifact.name)
            return

        Utils.ensure_dir(self.local_dir)
        logger.info("Downloading artifact (artifact=%s) to %s", self.artifact.name, self.local_dir)
        self.artifact.download(root=self.local_dir)
        logger.info("Download finished (artifact=%s, local_dir=%s)", self.artifact.name, self.local_dir)


class RunState:
    FINISHED = "finished"
    FAILED = "failed"
    RUNNING = "running"
    QUEUED = "queued"
    CRASHED = "crashed"


class Api:
    def __init__(self, project_path: str):
        self.api = wandb.Api()
        self.project_path = project_path
        logger.info("Initialized Api (project_path=%s)", self.project_path)

    def get_runs_before_cutoff(self, cutoff: datetime, only_completed: bool = True) -> list[Run]:
        cutoff_str = cutoff.isoformat()
        filters = {}

        if only_completed:
            filters["state"] = RunState.FINISHED

        # Get runs created after the cutoff date
        filters["createdAt"] = {"$gt": cutoff_str}

        runs = self.api.runs(self.project_path, filters=filters)

        logger.info(
            "Fetched runs after cutoff (project_path=%s, cutoff=%s, only_completed=%s, run_count=%d)",
            self.project_path,
            cutoff_str,
            only_completed,
            len(runs),
        )

        return (Run(run=r) for r in runs)


if __name__ == "__main__":
    api = Api(PATH)
    runs = api.get_runs_before_cutoff(CUTOFF, only_completed=True)

    for run in runs:
        for artifact in run.artifacts:
            artifact.download_if_of_interest()
