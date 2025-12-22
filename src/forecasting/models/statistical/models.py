
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


# %% ../../nbs/src/core/models.ipynb XXX
class AutoTSB(_TS):
    """
    Auto-tuned TSB.

    Searches (alpha_d, alpha_p) over a grid on fit(), minimizing insample MSE of fitted values.
    Stores best parameters and exposes forward() to apply them to new series.
    """

    def __init__(
        self,
        alpha_grid: Optional[np.ndarray] = None,
        alias: str = "AutoTSB",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.only_conformal_intervals = True

        if alpha_grid is None:
            # For probability smoothing, a wider useful range than Croston is common
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
        self.model_["sigma"] = _calculate_sigma(y - self.model_["fitted"], y.size)
        self._store_cs(y=y, X=X)
        return self

    def predict(self, h: int, X: Optional[np.ndarray] = None, level: Optional[List[int]] = None):
        mean = _repeat_val(self.model_["mean"][0], h=h)
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
        if self.alpha_d_ is None or self.alpha_p_ is None:
            raise Exception("You have to use the `fit` method first")

        y = _ensure_float(y)
        res = _tsb(y=y, h=h, fitted=fitted, alpha_d=self.alpha_d_, alpha_p=self.alpha_p_)
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
        Apply the already-selected (alpha_d_, alpha_p_) to a new series y.
        No parameter search here.
        """
        return self.forecast(y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted)
