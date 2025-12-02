import json
import os
import threading
from typing import Callable

import optuna


def make_trial_logger(path: str) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """
    Create an Optuna callback that appends trial info to a JSONL file.
    The callback is passed to NeuralForecast Auto models via the `callbacks` argument.
    """
    lock = threading.Lock()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        record = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        with lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    return _callback
