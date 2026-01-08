
from statsforecast.models import (
    _demand,
    _intervals,
    _ses_forecast,
    _tsb,
    _TS,
    _add_fitted_pi,
    _expand_fitted_demand,
    _expand_fitted_intervals
)
import numpy as np
from typing import Optional, Dict, List

from statsforecast.utils import (
    _calculate_sigma,
    _calculate_intervals,
    _ensure_float,
    _naive,
    _quantiles,
    _repeat_val,
    _repeat_val_seas,
    _seasonal_naive,
    CACHE,
    ConformalIntervals,
    NOGIL,
)

# %% ../../nbs/src/core/models.ipynb XXX
def _croston_sba_params(
    y: np.ndarray,
    h: int,
    fitted: bool,
    alpha_d: float,
    alpha_i: float,
) -> Dict[str, np.ndarray]:
    """
    Croston SBA with explicit smoothing params for:
      alpha_d: non-zero demand size SES
      alpha_i: inter-demand interval SES

    Forecast: 0.95 * z_hat / p_hat
    """
    y = _ensure_float(y)

    # demand
    yd = _demand(y)
    if not yd.size:
        # fallback for all-zeros / no-demand series
        out = _naive(y=y, h=h, fitted=fitted)
        out["mean"] *= 0.95
        if fitted and "fitted" in out:
            out["fitted"] *= 0.95
        return out

    ydf, ydft = _ses_forecast(yd, alpha_d)  # forecast level, fitted levels (len(ydft)=len(yd)?)
    # intervals
    yi = _intervals(y)
    yif, yift = _ses_forecast(yi, alpha_i)

    mean_val = 0.95 * (ydf / yif if yif != 0.0 else ydf)
    out = {"mean": _repeat_val(val=mean_val, h=h)}

    if fitted:
        # Expand fitted demand + interval sequences back to original timeline
        # We mimic your TSB pattern: append last observed to align expansion
        ydft_full = _expand_fitted_demand(np.append(ydft, yd), y)
        yift_full = _expand_fitted_intervals(np.append(yift, yi), y)

        # protect division
        yift_full = np.where(yift_full == 0.0, 1.0, yift_full)
        out["fitted"] = 0.95 * (ydft_full / yift_full)

        # match convention: first fitted is nan (like your TSB and Croston variants)
        if out["fitted"].size:
            out["fitted"][0] = np.nan

    return out


# %% ../../nbs/src/core/models.ipynb XXX
class AutoCrostonSBA(_TS):
    """
    Auto-tuned Croston SBA.

    Searches (alpha_d, alpha_i) over a grid on fit(), minimizing insample MSE of fitted values.
    Stores best parameters and exposes forward() to apply them to new series.
    """

    def __init__(
        self,
        alpha_grid: Optional[np.ndarray] = None,
        alias: str = "AutoCrostonSBA",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.only_conformal_intervals = True

        if alpha_grid is None:
            # Croston literature often uses smaller alphas; keep grid compact and practical
            self.alpha_grid = np.round(np.linspace(0.05, 0.30, 26), 4)
        else:
            self.alpha_grid = np.asarray(alpha_grid, dtype=np.float64)

        self.alpha_d_: Optional[float] = None
        self.alpha_i_: Optional[float] = None

    def _select_params(self, y: np.ndarray) -> Dict[str, float]:
        best = {"alpha_d": float(self.alpha_grid[0]), "alpha_i": float(self.alpha_grid[0])}
        best_mse = np.inf

        for a_d in self.alpha_grid:
            for a_i in self.alpha_grid:
                res = _croston_sba_params(y=y, h=1, fitted=True, alpha_d=float(a_d), alpha_i=float(a_i))
                fitted_vals = res["fitted"]
                mask = ~np.isnan(fitted_vals)
                if not mask.any():
                    continue
                err = y[mask] - fitted_vals[mask]
                mse = float(np.mean(err * err))
                if mse < best_mse:
                    best_mse = mse
                    best = {"alpha_d": float(a_d), "alpha_i": float(a_i)}
        return best

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        y = _ensure_float(y)

        params = self._select_params(y)
        self.alpha_d_ = params["alpha_d"]
        self.alpha_i_ = params["alpha_i"]

        self.model_ = _croston_sba_params(
            y=y,
            h=1,
            fitted=True,
            alpha_d=self.alpha_d_,
            alpha_i=self.alpha_i_,
        )
        self.model_["sigma"] = _calculate_sigma(y - self.model_["fitted"], y.size)
        self._store_cs(y=y, X=X)
        return self

    def predict(self, h: int, X: Optional[np.ndarray] = None, level: Optional[List[int]] = None):
        mean = _repeat_val(val=self.model_["mean"][0], h=h)
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = self._add_predict_conformal_intervals(res, level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            res = _add_fitted_pi(res=res, se=self.model_["sigma"], level=level)
        return res

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        if self.alpha_d_ is None or self.alpha_i_ is None:
            raise Exception("You have to use the `fit` method first")

        y = _ensure_float(y)
        res = _croston_sba_params(
            y=y,
            h=h,
            fitted=fitted,
            alpha_d=self.alpha_d_,
            alpha_i=self.alpha_i_,
        )
        res = dict(res)

        if level is None:
            return res

        level = sorted(level)
        if self.prediction_intervals is not None:
            res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")

        if fitted:
            sigma = _calculate_sigma(y - res["fitted"], y.size)
            res = _add_fitted_pi(res=res, se=sigma, level=level)

        return res

    def forward(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        """
        Apply the already-selected (alpha_d_, alpha_i_) to a new series y.
        No parameter search here.
        """
        return self.forecast(y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted)



# %% ../../nbs/src/core/models.ipynb 
class AutoTSB(_TS):
    """
    Auto-tuned TSB with optional *native* (parametric) prediction intervals
    using a Bernoulli gate + SES/ETS(ANN)-style Normal innovations.

    Predictive model (per horizon step):
        I ~ Bernoulli(p)
        Y = 0                      if I = 0
        Y = max(0, mu + eps)       if I = 1   (optional truncation)
        eps ~ Normal(0, sigma^2)

    Notes:
    - This is closest in spirit to ETS/SES additive-error (ANN) for the size component.
    - Quantiles are computed by Monte Carlo simulation (mixture quantiles have no simple closed form).
    - Conformal intervals still supported and can be used instead (or forced).
    """

    def __init__(
        self,
        alpha_grid: Optional[np.ndarray] = None,
        alias: str = "AutoTSB",
        prediction_intervals: Optional[ConformalIntervals] = None,
        # --- new knobs for "ETS-like" native intervals ---
        native_intervals: bool = True,
        n_sim: int = 10_000,
        random_state: Optional[int] = 0,
        truncate_at_zero: bool = True,
        # If True and prediction_intervals is provided, prefer conformal over native
        prefer_conformal: bool = True,
    ):
        self.alias = alias
        self.prediction_intervals = prediction_intervals

        # previously this class only did conformal; now it can do native too
        self.only_conformal_intervals = not native_intervals

        self.native_intervals = native_intervals
        self.n_sim = int(n_sim)
        self.random_state = random_state
        self.truncate_at_zero = truncate_at_zero
        self.prefer_conformal = prefer_conformal

        if alpha_grid is None:
            self.alpha_grid = np.round(np.linspace(0.02, 0.50, 49), 4)
        else:
            self.alpha_grid = np.asarray(alpha_grid, dtype=np.float64)

        self.alpha_d_: Optional[float] = None
        self.alpha_p_: Optional[float] = None

    def _select_params(self, y: np.ndarray) -> Dict[str, float]:
        best = {"alpha_d": float(self.alpha_grid[0]), "alpha_p": float(self.alpha_grid[0])}
        best_mse = np.inf

        for a_d in self.alpha_grid:
            for a_p in self.alpha_grid:
                res = _tsb(y=y, h=1, fitted=True, alpha_d=float(a_d), alpha_p=float(a_p))
                fitted_vals = res["fitted"]
                mask = ~np.isnan(fitted_vals)
                if not mask.any():
                    continue
                err = y[mask] - fitted_vals[mask]
                mse = float(np.mean(err * err))
                if mse < best_mse:
                    best_mse = mse
                    best = {"alpha_d": float(a_d), "alpha_p": float(a_p)}
        return best

    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        y = _ensure_float(y)

        params = self._select_params(y)
        self.alpha_d_ = params["alpha_d"]
        self.alpha_p_ = params["alpha_p"]

        self.model_ = _tsb(y=y, h=1, fitted=True, alpha_d=self.alpha_d_, alpha_p=self.alpha_p_)

        # sigma estimated from residuals (ETS/SES-like)
        self.model_["sigma"] = _calculate_sigma(y - self.model_["fitted"], y.size)

        self._store_cs(y=y, X=X)
        return self

    def _simulate_native_pi(
        self,
        mu: np.ndarray,
        p: np.ndarray,
        sigma: float,
        level: List[int],
    ) -> Dict[str, np.ndarray]:
        """
        Monte Carlo quantiles for the Bernoulli x Normal-mixture.
        Returns dict with lo-{l}, hi-{l}.
        """
        if self.n_sim <= 0:
            raise ValueError("n_sim must be > 0 for native intervals.")

        rng = np.random.default_rng(self.random_state)

        h = mu.size
        p = np.clip(p, 0.0, 1.0)

        # Draw gates: shape (n_sim, h)
        gates = rng.random((self.n_sim, h)) < p[None, :]

        # Draw eps: shape (n_sim, h)
        eps = rng.normal(loc=0.0, scale=float(sigma), size=(self.n_sim, h))

        # size when gate=1
        y_pos = mu[None, :] + eps
        if self.truncate_at_zero:
            y_pos = np.maximum(0.0, y_pos)

        sim = np.where(gates, y_pos, 0.0)

        out: Dict[str, np.ndarray] = {}
        # quantiles per horizon
        # For level l: lower=(100-l)/2, upper=100-(100-l)/2
        for l in level:
            alpha = (100.0 - float(l)) / 100.0
            q_lo = alpha / 2.0
            q_hi = 1.0 - alpha / 2.0
            out[f"lo-{l}"] = np.quantile(sim, q_lo, axis=0)
            out[f"hi-{l}"] = np.quantile(sim, q_hi, axis=0)
        return out

    def predict(self, h: int, X: Optional[np.ndarray] = None, level: Optional[List[int]] = None):
        # Point forecast (TSB mean)
        mean = _repeat_val(self.model_["mean"][0], h=h)
        res = {"mean": mean}
        if level is None:
            return res

        level = sorted(level)

        # prefer conformal if requested and available
        if self.prediction_intervals is not None and self.prefer_conformal:
            return self._add_predict_conformal_intervals(res, level)

        # native intervals (ETS-like) if enabled
        if self.native_intervals:
            # We need p and mu for the horizon.
            # In many TSB implementations, the mean is p * z (z=size-level).
            # If your _tsb returns components (like prob/size), use them here.
            # Otherwise, we approximate:
            #   - assume p is last smoothed probability (if available) else infer from mean and last size.
            #   - assume mu is conditional mean size (mean / p) if p>0, else 0.
            mod = self.model_

            # Try to use stored components if present (best)
            if "prob" in mod:
                p = _repeat_val(float(mod["prob"][0]), h=h)
            elif "p" in mod:
                p = _repeat_val(float(mod["p"][0]), h=h)
            else:
                # fallback: estimate p from empirical non-zero rate (crude but workable)
                # (better: store p in _tsb)
                y_hist = getattr(self, "y_", None)
                if y_hist is None:
                    p_last = 0.0
                else:
                    p_last = float(np.mean(np.asarray(y_hist) > 0))
                p = _repeat_val(p_last, h=h)

            # conditional size mean mu
            # If _tsb exposes it, use it; else use mean / max(p, eps)
            if "size" in mod:
                mu = _repeat_val(float(mod["size"][0]), h=h)
            elif "z" in mod:
                mu = _repeat_val(float(mod["z"][0]), h=h)
            else:
                p_safe = np.maximum(p, 1e-12)
                mu = mean / p_safe

            sigma = float(mod.get("sigma", 0.0))
            res.update(self._simulate_native_pi(mu=mu, p=p, sigma=sigma, level=level))
            return res

        # if we get here, user asked for intervals but none configured
        if self.prediction_intervals is not None:
            return self._add_predict_conformal_intervals(res, level)

        raise Exception(
            "Prediction intervals requested but neither native_intervals=True nor prediction_intervals provided."
        )

    def predict_in_sample(self, level: Optional[List[int]] = None):
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            res = _add_fitted_pi(res=res, se=self.model_["sigma"], level=level)
        return res

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        if self.alpha_d_ is None or self.alpha_p_ is None:
            raise Exception("You have to use the `fit` method first")

        y = _ensure_float(y)

        # keep y for the crude p fallback in predict()
        self.y_ = y

        res = _tsb(y=y, h=h, fitted=fitted, alpha_d=self.alpha_d_, alpha_p=self.alpha_p_)
        res = dict(res)

        if level is None:
            return res

        level = sorted(level)

        # prefer conformal if requested and available
        if self.prediction_intervals is not None and self.prefer_conformal:
            res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
        elif self.native_intervals:
            # need mean path; and p/mu if available
            mean = res["mean"]

            # try to get p and mu from _tsb outputs if present
            if "prob" in res:
                p = res["prob"]
            elif "p" in res:
                p = res["p"]
            else:
                # fallback: empirical non-zero rate
                p = _repeat_val(float(np.mean(y > 0)), h=h)

            if "size" in res:
                mu = res["size"]
            elif "z" in res:
                mu = res["z"]
            else:
                p_safe = np.maximum(p, 1e-12)
                mu = mean / p_safe

            # sigma: recompute from fitted residuals if available; else use stored
            if "fitted" in res:
                sigma = float(_calculate_sigma(y - res["fitted"], y.size))
            else:
                # compute from a 1-step fitted model (cheap) as fallback
                mod1 = _tsb(y=y, h=1, fitted=True, alpha_d=self.alpha_d_, alpha_p=self.alpha_p_)
                sigma = float(_calculate_sigma(y - mod1["fitted"], y.size))

            res.update(self._simulate_native_pi(mu=mu, p=p, sigma=sigma, level=level))
        else:
            if self.prediction_intervals is not None:
                res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
            else:
                raise Exception("You must pass `prediction_intervals` or enable native_intervals.")

        if fitted:
            sigma_fit = _calculate_sigma(y - res["fitted"], y.size)
            res = _add_fitted_pi(res=res, se=sigma_fit, level=level)

        return res

    def forward(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        """
        Apply the already-selected (alpha_d_, alpha_p_) to a new series y.
        No parameter search here.
        """
        return self.forecast(y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted)
