import math
import logging
from typing import Callable

import optuna
from optuna.trial import TrialState


def fail_on_nonfinite_trial(
    min_valid_loss: float = 1e-9,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """
    Optuna callback: if a trial completes with a non-finite or suspiciously low value,
    mark it as FAIL so it is ignored when selecting the best trial.

    Args:
        min_valid_loss: Minimum threshold for a valid loss. Values below this are
            considered degenerate (e.g., predicting all zeros, NaN converted to 0).
            Default is 1e-9 to catch 0.0 and near-zero values.

    A value of exactly 0.0 or near-zero is considered invalid because it typically
    indicates a degenerate model (e.g., predicting all zeros, or NaN converted to 0).
    """

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        val = trial.value
        if val is None or not math.isfinite(val) or val < min_valid_loss:
            logging.warning(
                f"Trial {trial.number} has invalid loss value {val}; marking as FAIL."
            )
            # Update state in storage so `study.best_trial` will skip it.
            study._storage.set_trial_state(trial._trial_id, TrialState.FAIL)

    return _callback
