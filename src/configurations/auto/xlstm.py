
from neuralforecast.models import xLSTM
from neuralforecast.auto import BaseAuto, MAE, cpu_count, BasicVariantGenerator, tune
import torch


class AutoxLSTM(BaseAuto):

    default_config = {
        "input_size_multiplier": [-1, 4, 16, 64],
        "inference_input_size_multiplier": [-1],
        "h": None,
        "encoder_hidden_size": tune.choice([16, 32, 64, 128]),
        "encoder_n_blocks": tune.randint(1, 4),
        "decoder_hidden_size": tune.choice([16, 32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "max_steps": tune.choice([500, 1000]),
        "batch_size": tune.choice([16, 32]),
        "loss": None,
        "random_seed": tune.randint(1, 20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        # Define search space, input/output sizes
        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoxLSTM, self).__init__(
            cls_model=xLSTM,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["inference_input_size"] = tune.choice(
            [h * x for x in config["inference_input_size_multiplier"]]
        )
        del config["input_size_multiplier"], config["inference_input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config