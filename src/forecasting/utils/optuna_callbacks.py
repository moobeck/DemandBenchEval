import math
from typing import Callable

import optuna
from optuna.trial import TrialState


def fail_on_nonfinite_trial() -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """
    Optuna callback: if a trial completes with a non-finite value, mark it as FAILED
    so it is ignored when selecting the best trial.
    """

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        val = trial.value
        if val is None or not math.isfinite(val):
            # Update state in storage so `study.best_trial` will skip it.
            study._storage.set_trial_state(trial._trial_id, TrialState.FAIL)

    return _callback
