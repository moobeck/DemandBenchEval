from typing import Optional, Union, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions import (
    Normal,
    MixtureSameFamily,
    Categorical,
    AffineTransform,
    TransformedDistribution,
)
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.uniform import Uniform
import math


def quantiles_to_outputs(quantiles):
    output_names = []
    for q in quantiles:
        if q < 0.50:
            output_names.append(f"-lo-{np.round(100-200*q,2)}")
        elif q > 0.50:
            output_names.append(f"-hi-{np.round(100-200*(1-q),2)}")
        else:
            output_names.append("-median")
    return quantiles, output_names

def level_to_outputs(level):
    qs = sum([[50 - l / 2, 50 + l / 2] for l in level], [])
    output_names = sum([[f"-lo-{l}", f"-hi-{l}"] for l in level], [])

    sort_idx = np.argsort(qs)
    quantiles = np.array(qs)[sort_idx]

    # Add default median
    quantiles = np.concatenate([np.array([50]), quantiles])
    quantiles = torch.Tensor(quantiles) / 100
    output_names = list(np.array(output_names)[sort_idx])
    output_names.insert(0, "-median")

    return quantiles, output_names

def weighted_average(
    x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim)

def divideno_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    div = a / b
    return torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)

def weightedmean(losses, weights):
    return divideno_nan(torch.sum(losses * weights), torch.sum(weights))


class TruncatedNormal(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.interval(0., 1.)
    has_rsample = False

    def __init__(self, loc, scale, low=0., high=1., validate_args=None):
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
        Z = torch.clamp(high_cdf - low_cdf, min=1e-10)
        log_prob = base_log_prob - torch.log(Z)
        inside = (value >= self.low) & (value <= self.high)
        return torch.where(inside, log_prob, torch.tensor(-float('inf')).to(log_prob))

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            low_cdf = self.base_dist.cdf(self.low)
            high_cdf = self.base_dist.cdf(self.high)
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
            u = low_cdf + u * (high_cdf - low_cdf)
            u = torch.clamp(u, min=1e-10, max=1.-1e-10)
            sample = self.base_dist.icdf(u)
        return sample

    @property
    def mean(self):
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        sqrt_2pi = math.sqrt(2 * math.pi)
        phi_a = torch.exp(-0.5 * a**2) / sqrt_2pi
        phi_b = torch.exp(-0.5 * b**2) / sqrt_2pi
        Phi_a = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        Phi_b = 0.5 * (1 + torch.erf(b / math.sqrt(2)))
        Z = Phi_b - Phi_a
        Z = torch.clamp(Z, min=0.1)
        mean = self.loc + self.scale * (phi_a - phi_b) / Z
        return mean

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

    def domain_map(self, output: torch.Tensor):
        output = output.reshape(
            output.shape[0], output.shape[1], -1, self.outputsize_multiplier
        )
        return torch.tensor_split(output, self.n_outputs, dim=-1)

    def scale_decouple(
        self,
        output,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        eps: float = 0.2,
    ):
        if self.weighted:
            means, stds, weights = output
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

    def get_distribution(self, distr_args) -> Distribution:
        if self.weighted:
            means, stds, weights = distr_args
        else:
            means, stds = distr_args
            weights = torch.full_like(means, fill_value=1 / self.n_components)
        mix = Categorical(weights)
        components = TruncatedNormal(loc=means, scale=stds, low=0., high=1.)
        distr = MixtureSameFamily(
            mixture_distribution=mix, component_distribution=components
        )
        self.distr_mean = distr.mean
        return distr

    def sample(self, distr_args: torch.Tensor, num_samples: Optional[int] = None):
        if num_samples is None:
            num_samples = self.num_samples
        distr = self.get_distribution(distr_args=distr_args)
        samples = distr.sample(sample_shape=(num_samples,))
        samples = samples.permute(1, 2, 3, 0)
        sample_mean = torch.mean(samples, dim=-1, keepdim=True)
        quantiles_device = self.quantiles.to(distr_args[0].device)
        quants = torch.quantile(input=samples, q=quantiles_device, dim=-1)
        quants = quants.permute(1, 2, 3, 0)
        return samples, sample_mean, quants

    def update_quantile(self, q: Optional[List[float]] = None):
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
        distr = self.get_distribution(distr_args=distr_args)
        x = distr._pad(y)
        log_prob_x = distr.component_distribution.log_prob(x)
        log_mix_prob = torch.log_softmax(distr.mixture_distribution.logits, dim=-1)
        if self.batch_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=0, keepdim=True)
        if self.horizon_correlation:
            log_prob_x = torch.sum(log_prob_x, dim=1, keepdim=True)
        loss_values = -torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)
        return weighted_average(loss_values, weights=mask)