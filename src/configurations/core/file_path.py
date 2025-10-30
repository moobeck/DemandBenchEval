from dataclasses import dataclass
from src.configurations.utils.enums import DatasetName, FileFormat
import os


@dataclass
class FilePathConfig:
    """
    A dataclass to store the paths of the input and output files.
    """

    processed_data_dir: str = None
    processed_data: str = None
    sku_stats_dir: str = None
    sku_stats: str = None
    cv_results_dir: str = None
    cv_results: str = None
    eval_results_dir: str = None
    eval_results: str = None
    eval_plots_dir: str = None
    eval_plots: str = None
    file_format: FileFormat = None

    def set_file_paths(self, dataset_name: DatasetName):
        """
        Sets the file paths based on the dataset name.
        """
        self.processed_data = (
            f"{self.processed_data_dir}/{dataset_name.value}.{self.file_format.value}"
        )
        self.sku_stats = (
            f"{self.sku_stats_dir}/{dataset_name.value}.{self.file_format.value}"
        )
        self.cv_results = (
            f"{self.cv_results_dir}/{dataset_name.value}.{self.file_format.value}"
        )
        self.eval_results = (
            f"{self.eval_results_dir}/{dataset_name.value}.{self.file_format.value}"
        )
        self.eval_plots = f"{self.eval_plots_dir}/{dataset_name.value}.png"

    def iter_directories(self):
        """
        Yield directory path attributes declared on this dataclass (attributes ending with '_dir').
        Skips None or empty values.
        """
        for name, val in self.__dict__.items():
            if name.endswith("_dir") and isinstance(val, str) and val.strip():
                yield val

    def missing_directories(self, dirs):
        """
        Given an iterable of directory paths, return a list of those that do not currently exist
        or are not directories.
        """

        missing = []
        for d in dirs:
            if not isinstance(d, str) or not d.strip():
                continue
            if not os.path.isdir(d):
                missing.append(d)
        return missing

    def create_directories(self, dirs, exist_ok: bool = True):
        """
        Create the directories from the provided iterable that are missing.
        """

        for d in self.missing_directories(dirs):
            os.makedirs(d, exist_ok=exist_ok)

    def ensure_directories_exist(self):
        """
        Ensure all directory attributes on this dataclass exist on disk.
        """
        dirs = list(self.iter_directories())
        self.create_directories(dirs)
