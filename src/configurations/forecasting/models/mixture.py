from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions import (
    Normal,
    MixtureSameFamily,
    Categorical,
)
import math
import logging
from typing import Dict, Any
from neuralforecast.losses.pytorch import (
    quantiles_to_outputs,
    level_to_outputs,
    weighted_average,
)
from src.configurations.forecasting.utils.quantile import QuantileUtils


class TruncatedNormal(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.interval(0.0, 1.0)
    has_rsample = False
    MIN_CLAMP_VALUE = 1e-10
    N_SAMPLES = 1000

    def __init__(self, loc, scale, low=0.0, high=1.0, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.base_dist = Normal(loc, scale, validate_args=False)
        batch_shape = self.loc.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        base_log_prob = self.base_dist.log_prob(value)
        low_cdf = self.base_dist.cdf(self.low)
        high_cdf = self.base_dist.cdf(self.high)
        Z = torch.clamp(high_cdf - low_cdf, min=self.MIN_CLAMP_VALUE)
        log_prob = base_log_prob - torch.log(Z)
        inside = (value >= self.low) & (value <= self.high)
        return torch.where(inside, log_prob, torch.tensor(-float("inf")).to(log_prob))

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            low_cdf = self.base_dist.cdf(self.low)
            high_cdf = self.base_dist.cdf(self.high)
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
            u = low_cdf + u * (high_cdf - low_cdf)
            u = torch.clamp(u, min=self.MIN_CLAMP_VALUE, max=1.0 - self.MIN_CLAMP_VALUE)
            sample = self.base_dist.icdf(u)
        return sample

    @property
    def mean(self):
        samples = self.sample(sample_shape=(self.N_SAMPLES,))
        return torch.mean(samples, dim=0)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TruncatedNormal, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.low = self.low
        new.high = self.high
        new.base_dist = Normal(new.loc, new.scale)
        super(TruncatedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


class TGMM(nn.Module):
    def __init__(
        self,
        n_components=1,
        level=[80, 90],
        quantiles=None,
        num_samples=1000,
        return_params=False,
        batch_correlation=False,
        horizon_correlation=False,
        weighted=False,
    ):
        super(TGMM, self).__init__()
        if quantiles is not None:
            qs, self.output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(qs)
        else:
            qs, self.output_names = level_to_outputs(level)
            qs = torch.Tensor(qs)
        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.num_samples = num_samples
        self.batch_correlation = batch_correlation
        self.horizon_correlation = horizon_correlation
        self.weighted = weighted
        self.return_params = return_params

        mu_names = [f"-mu-{i}" for i in range(1, n_components + 1)]
        std_names = [f"-std-{i}" for i in range(1, n_components + 1)]
        if weighted:
            weight_names = [f"-weight-{i}" for i in range(1, n_components + 1)]
            self.param_names = [
                i for j in zip(mu_names, std_names, weight_names) for i in j
            ]
        else:
            self.param_names = [i for j in zip(mu_names, std_names) for i in j]
        if self.return_params:
            self.output_names = self.output_names + self.param_names
        self.output_names.insert(0, "")
        self.n_outputs = 2 + weighted
        self.n_components = n_components
        self.outputsize_multiplier = self.n_outputs * n_components
        self.is_distribution_output = True
        self.has_predicted = False

    def _domain_map(self, output: torch.Tensor):
        output = output.reshape(
            output.shape[0], output.shape[1], -1, self.outputsize_multiplier
        )
        return torch.tensor_split(output, self.n_outputs, dim=-1)

    def _scale_decouple(
        self,
        output,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        eps: float = 0.2,
    ):
        if self.weighted:
            means, stds, weights = output
            if self.horizon_correlation:
                weights = weights.mean(dim=1, keepdim=True).expand(
                    -1, means.shape[1], -1, -1
                )
            weights = F.softmax(weights, dim=-1)
        else:
            means, stds = output
        stds = F.softplus(stds)
        if (loc is not None) and (scale is not None):
            if loc.ndim == 3:
                loc = loc.unsqueeze(-1)
                scale = scale.unsqueeze(-1)
            means = (means * scale) + loc
            stds = (stds + eps) * scale
        if self.weighted:
            return (means, stds, weights)
        else:
            return (means, stds)

    def _get_distribution(self, distr_args) -> Distribution:
        if self.weighted:
            means, stds, weights = distr_args
        else:
            means, stds = distr_args
            weights = torch.full_like(means, fill_value=1 / self.n_components)
        mix = Categorical(weights)
        components = TruncatedNormal(loc=means, scale=stds, low=0.0, high=1.0)
        distr = MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )
        self.distr_mean = distr.mean
        return distr

    def sample(self, distr_args: torch.Tensor, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = self.num_samples
        distr = self._get_distribution(distr_args=distr_args)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(1, 2, 3, 0)
        sample_mean = torch.mean(samples, dim=-1, keepdim=True)
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=-1)
        quants = quants.permute(1, 2, 3, 0)
        return samples, sample_mean, quants

    def _update_quantile(self, q: Optional[List[float]] = None):
        if q is not None:
            self.quantiles = nn.Parameter(
                torch.tensor(q, dtype=torch.float32), requires_grad=False
            )
            self.output_names = (
                [""]
                + [f"_ql{q_i}" for q_i in q]
                + self.return_params * self.param_names
            )
            self.has_predicted = True
        elif q is None and self.has_predicted:
            self.quantiles = nn.Parameter(
                torch.tensor([0.5], dtype=torch.float32), requires_grad=False
            )
            self.output_names = ["", "-median"] + self.return_params * self.param_names

    def __call__(
        self,
        y: torch.Tensor,
        distr_args: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        distr = self._get_distribution(distr_args=distr_args)
        x = distr._pad(y)
        log_prob_x = distr.component_distribution.log_prob(x)
        log_mix_prob = torch.log_softmax(distr.mixture_distribution.logits, dim=-1)
        if self.batch_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=0, keepdim=True)
        if self.horizon_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=1, keepdim=True)
        loss_values = -torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)
        return weighted_average(loss_values, weights=mask)


class MixtureLossFactory:
    """
    Factory class to create mixture-specific loss functions based on configuration.
    """

    @staticmethod
    def create_loss(mixture_config: Dict[str, Any]) -> Any:
        """
        Create a mixture-specific loss function based on the mixture_config.

        Args:
            mixture_config (Dict[str, Any]): Configuration dictionary for the mixture model.

        Returns:
            An instance of a loss function (TGMM or MAE).

        Notes:
            - If mixture_config is empty or None, returns MAE loss.
            - If mixture_config contains TGMM settings, returns a TGMM instance.
        """

        if "TGMM" in mixture_config:
            tgmm_config = mixture_config["TGMM"]
            logging.info(f"Creating TGMM loss with configuration: {tgmm_config}")

            # Extract TGMM parameters with defaults
            n_components = tgmm_config.get("num_components", 1)
            horizon_correlation = tgmm_config.get("horizon_correlation", True)
            weighted = tgmm_config.get("weighted", True)
            return_params = tgmm_config.get("return_params", True)
            quantiles = tgmm_config.get("quantiles", None)

            # Create and return TGMM instance
            tgmm_loss = TGMM(
                n_components=n_components,
                horizon_correlation=horizon_correlation,
                weighted=weighted,
                return_params=return_params,
                level=QuantileUtils.quantiles_to_level(quantiles),
            )
            logging.info(
                f"Successfully created TGMM loss with {n_components} components"
            )
            return tgmm_loss
        else:
            raise ValueError(
                f"Mixture configuration found but no supported mixture type: {list(mixture_config.keys())}. "
                "Supported types: ['TGMM']"
            )
