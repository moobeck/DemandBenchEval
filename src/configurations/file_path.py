from dataclasses import dataclass


@dataclass(frozen=True)
class FilePathConfig:
    """
    A dataclass to store the paths of the input and output files.
    """

    train_data_features: str = None
    val_data_features: str = None
    train_data_target: str = None
    val_data_target: str = None
    eval_results: str = None
    eval_plots: str = None
