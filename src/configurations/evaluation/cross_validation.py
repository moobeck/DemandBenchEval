from dataclasses import dataclass, field
from ..utils.enums import FrequencyType
from src.utils.cross_validation import get_offset
import pandas as pd
from src.constants.tasks import Task


@dataclass
class CrossValidationConfig:
    n_windows: int = field(default_factory=lambda: 322)

    @property
    def step_size(self) -> int:
        return 1

    @property
    def refit(self) -> bool:
        return False

    def set_task(self, task: Task) -> None:
        """
        Sets the cross-validation configuration based on the task.
        """

        start_date, end_date = task.dataset.metadata.time_range
        frequency = task.dataset.metadata.frequency
        freq_map = {"daily": "D", "weekly": "W", "monthly": "MS"}
        date_range = pd.date_range(
            start=start_date, end=end_date, freq=freq_map[frequency]
        )
        num_time_points = len(date_range)

        rel_train_size = task.train_test_split.rel_train_size

        train_size = int(round(rel_train_size * num_time_points, 0))
        test_size = num_time_points - train_size

        self.n_windows = test_size - self.step_size + 1

    def get_cutoff_date(
        self,
        max_date: pd.Timestamp,
        freq: FrequencyType,
        horizon: int,
    ) -> pd.Timestamp:
        """
        Calculate the cutoff date for training data based on frequency.
        """

        n_windows = self.n_windows
        step_size = self.step_size

        if freq == FrequencyType.DAILY:
            offset = pd.Timedelta(days=get_offset(n_windows, step_size, horizon))
        elif freq == FrequencyType.WEEKLY:
            offset = pd.Timedelta(weeks=get_offset(n_windows, step_size, horizon))
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
        return max_date - offset


DEFAULT_CROSS_VALIDATION_CONFIG = CrossValidationConfig()
