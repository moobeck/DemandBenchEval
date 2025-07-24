from dataclasses import dataclass
from src.configurations.enums import DatasetName


@dataclass
class FilePathConfig:
    """
    A dataclass to store the paths of the input and output files.
    """
    cv_results_dir: str = None
    cv_results: str = None
    eval_results_dir: str = None
    eval_plots_dir: str = None
    eval_results: str = None
    eval_plots: str = None

    def set_file_paths(self, dataset_name: DatasetName):
        """
        Sets the file paths based on the dataset name.
        """
        self.cv_results = f"{self.cv_results_dir}/{dataset_name.value}.feather"
        self.eval_results = f"{self.eval_results_dir}/{dataset_name.value}.feather"
        self.eval_plots = f"{self.eval_plots_dir}/{dataset_name.value}.png"
