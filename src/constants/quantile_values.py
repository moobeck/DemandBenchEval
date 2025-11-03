from typing import Tuple

Q_MIN = 10
Q_MAX = 90
Q_STEP = 10

DEFAULT_QUANTILE_VALUES: Tuple[float, ...] = tuple(q / 100 for q in range(Q_MIN, Q_MAX + 1, Q_STEP))