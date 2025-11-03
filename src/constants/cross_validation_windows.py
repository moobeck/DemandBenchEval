"""Default rolling-window settings for benchmark datasets."""
from dataclasses import dataclass


@dataclass
class TrainTestSplitConfig:
    rel_train_size: float
    rel_test_size: float


DEFAULT_TRAIN_TEST_SPLIT = TrainTestSplitConfig(
    rel_train_size=0.8, rel_test_size=0.2
)


