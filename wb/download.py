from datetime import datetime, timezone
import os
import wandb
import logging

from src.configurations.utils.enums import DatasetName, ModelName


PROJECT = "bench-forecast"
TEAM = "d3_group"
PATH = f"{TEAM}/{PROJECT}"
CUTOFF = datetime(2025, 12, 10, 0, 0, 0, tzinfo=timezone.utc)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Utils:
    @staticmethod
    def ensure_dir(dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)


class Run:
    def __init__(self, run: wandb.Run):
        self.run = run
        logger.info("Initialized Run (run=%s, id=%s)", self.name, self.id)

    @property
    def name(self) -> str:
        return self.run.name

    @property
    def id(self) -> str:
        return self.run.id

    @property
    def tag_names(self) -> list[str]:
        return [tag.lower() for tag in self.run.tags]

    @property
    def model_name(self) -> ModelName:
        for tag in self.tag_names:
            try:
                return ModelName(tag)
            except ValueError:
                continue

    @property
    def dataset_name(self) -> DatasetName | None:
        for tag in self.tag_names:
            try:
                return DatasetName(tag)
            except ValueError:
                continue

    @property
    def created_at(self) -> datetime | None:
        return self.run.created_at

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
        logger.info(
            "Resolved artifact type (artifact=%s -> type=%s)", self.artifact.name, t
        )
        return t

    @property
    def _is_of_interest(self) -> bool:
        is_interest = self.type in {
            ArtifactTypes.CV_RESULTS,
            ArtifactTypes.EVAL_RESULTS,
        }
        logger.info(
            "Checked if artifact is of interest (artifact=%s, type=%s, is_of_interest=%s)",
            self.artifact.name,
            self.type,
            is_interest,
        )
        return is_interest

    def download_if_of_interest(self) -> None:
        logger.info(
            "Attempting download_if_of_interest (artifact=%s)", self.artifact.name
        )
        if not self._is_of_interest:
            logger.info(
                "Skipping download (artifact=%s) because it is not of interest",
                self.artifact.name,
            )
            return

        Utils.ensure_dir(self.local_dir)
        logger.info(
            "Downloading artifact (artifact=%s) to %s",
            self.artifact.name,
            self.local_dir,
        )
        self.artifact.download(root=self.local_dir)
        logger.info(
            "Download finished (artifact=%s, local_dir=%s)",
            self.artifact.name,
            self.local_dir,
        )


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

    def get_runs_before_cutoff(
        self, cutoff: datetime, only_completed: bool = True
    ) -> list[Run]:
        cutoff_str = cutoff.isoformat()
        filters = {}

        if only_completed:
            filters["state"] = RunState.FINISHED

        # Get runs created after the cutoff date
        filters["createdAt"] = {"$gt": cutoff_str}

        fetched_runs = list(self.api.runs(self.project_path, filters=filters))

        logger.info(
            "Fetched runs after cutoff (project_path=%s, cutoff=%s, only_completed=%s, run_count=%d)",
            self.project_path,
            cutoff_str,
            only_completed,
            len(fetched_runs),
        )

        unique_runs: dict[tuple[str, str], Run] = {}
        for raw_run in fetched_runs:
            run = Run(run=raw_run)
            model = run.model_name
            if model is None:
                logger.info("Skipping run without model tag (id=%s)", run.id)
                continue

            dataset = run.dataset_name

            if dataset is None:
                logger.info(
                    "Run has no dataset tag (id=%s); using 'unknown' as dataset key",
                    run.id,
                )
                continue

            key = (model.value, dataset.value)

            existing = unique_runs.get(key)
            if existing is None or self._is_newer(run, existing):
                unique_runs[key] = run

        logger.info(
            "Selected newest runs per model (project_path=%s, selected_count=%d)",
            self.project_path,
            len(unique_runs),
        )

        return list(unique_runs.values())

    @staticmethod
    def _is_newer(candidate: Run, current: Run) -> bool:
        candidate_ts = Api._timestamp(candidate)
        current_ts = Api._timestamp(current)
        return candidate_ts > current_ts

    @staticmethod
    def _timestamp(run: Run) -> datetime:
        ts = run.created_at or datetime.min.replace(tzinfo=timezone.utc)
        return ts


if __name__ == "__main__":
    api = Api(PATH)
    runs = api.get_runs_before_cutoff(CUTOFF, only_completed=True)

    for run in runs:
        for artifact in run.artifacts:
            artifact.download_if_of_interest()
