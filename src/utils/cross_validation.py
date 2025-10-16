def get_offset(n_windows: int, step_size: int, horizon: int) -> int:
    return horizon + step_size * (n_windows - 1)
