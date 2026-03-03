
from statsforecast.models import (
    _demand,
    _intervals,
    _ses_forecast,
    _probability,
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



import numpy as np
from typing import Optional, List, Dict

# --- UPDATED: return probability + size components so we can build a native mixture distribution
def _tsb(
    y: np.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: int,  # fitted values
    alpha_d: float,
    alpha_p: float,
) -> Dict[str, np.ndarray]:
    if (y == 0).all():
        res = {
            "mean": np.zeros(h, dtype=y.dtype),
            "prob": np.zeros(h, dtype=np.float64),  # p_t
            "size": np.zeros(h, dtype=np.float64),  # mu_t (conditional on demand)
        }
        if fitted:
            res["fitted"] = np.zeros_like(y)
            res["fitted"][0] = np.nan
        return res

    y = _ensure_float(y)
    yd = _demand(y)        # positive sizes (typically non-zeros)
    yp = _probability(y)   # 0/1 occurrence indicator

    ypf, ypft = _ses_forecast(yp, alpha_p)  # last-level for prob + fitted
    ydf, ydft = _ses_forecast(yd, alpha_d)  # last-level for size + fitted

    p_last = float(ypf)
    mu_last = float(ydf)

    res = {
        "mean": _repeat_val(val=p_last * mu_last, h=h),
        "prob": _repeat_val(val=p_last, h=h),
        "size": _repeat_val(val=mu_last, h=h),
    }

    if fitted:
        ydft = _expand_fitted_demand(np.append(ydft, yd), y)
        res["fitted"] = ypft * ydft

    return res


# %% ../../nbs/src/core/models.ipynb XXX
class AutoTSB(_TS):
    """
    Auto-tuned TSB with optional native (parametric) prediction intervals:
      Y = 0 w.p. (1-p)
      Y = max(0, mu + eps) w.p. p
      eps ~ Normal(0, sigma^2)

    This is the closest analogue to ETS/SES(ANN) additive-error for intermittent demand.
    """

    def __init__(
        self,
        alpha_grid: Optional[np.ndarray] = None,
        alias: str = "AutoTSB",
        prediction_intervals: Optional[ConformalIntervals] = None,
        # native intervals knobs
        native_intervals: bool = True,
        n_sim: int = 10_000,
        random_state: Optional[int] = 0,
        truncate_at_zero: bool = True,
        prefer_conformal: bool = True,
    ):
        self.alias = alias
        self.prediction_intervals = prediction_intervals

        self.native_intervals = native_intervals
        self.n_sim = int(n_sim)
        self.random_state = random_state
        self.truncate_at_zero = truncate_at_zero
        self.prefer_conformal = prefer_conformal

        # now we can do native, so this is no longer "only conformal" unless user disables it
        self.only_conformal_intervals = not native_intervals

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

        # ETS/SES-like innovation scale from residuals
        self.model_["sigma"] = _calculate_sigma(y - self.model_["fitted"], y.size)

        self._store_cs(y=y, X=X)
        return self

    def _native_pi(self, p: np.ndarray, mu: np.ndarray, sigma: float, level: List[int]) -> Dict[str, np.ndarray]:
        if self.n_sim <= 0:
            raise ValueError("n_sim must be > 0 for native intervals.")
        rng = np.random.default_rng(self.random_state)

        h = mu.size
        p = np.clip(p, 0.0, 1.0)

        gates = rng.random((self.n_sim, h)) < p[None, :]
        eps = rng.normal(loc=0.0, scale=float(sigma), size=(self.n_sim, h))

        y_pos = mu[None, :] + eps
        if self.truncate_at_zero:
            y_pos = np.maximum(0.0, y_pos)

        sim = np.where(gates, y_pos, 0.0)

        out: Dict[str, np.ndarray] = {}
        for l in level:
            alpha = (100.0 - float(l)) / 100.0
            q_lo = alpha / 2.0
            q_hi = 1.0 - alpha / 2.0
            out[f"lo-{l}"] = np.quantile(sim, q_lo, axis=0)
            out[f"hi-{l}"] = np.quantile(sim, q_hi, axis=0)
        return out

    def predict(self, h: int, X: Optional[np.ndarray] = None, level: Optional[List[int]] = None):
        mean = _repeat_val(self.model_["mean"][0], h=h)
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)

        if self.prediction_intervals is not None and self.prefer_conformal:
            return self._add_predict_conformal_intervals(res, level)

        if self.native_intervals:
            p = _repeat_val(self.model_["prob"][0], h=h)
            mu = _repeat_val(self.model_["size"][0], h=h)
            sigma = float(self.model_["sigma"])
            res.update(self._native_pi(p=p, mu=mu, sigma=sigma, level=level))
            return res

        if self.prediction_intervals is not None:
            return self._add_predict_conformal_intervals(res, level)

        raise Exception("You must pass `prediction_intervals` or enable native_intervals to compute intervals.")

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
        res = dict(_tsb(y=y, h=h, fitted=fitted, alpha_d=self.alpha_d_, alpha_p=self.alpha_p_))

        if level is None:
            return res

        level = sorted(level)

        if self.prediction_intervals is not None and self.prefer_conformal:
            res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
        elif self.native_intervals:
            # sigma from residuals (if fitted returned, use it; else do a 1-step fitted pass)
            if "fitted" in res:
                sigma = float(_calculate_sigma(y - res["fitted"], y.size))
            else:
                mod1 = _tsb(y=y, h=1, fitted=True, alpha_d=self.alpha_d_, alpha_p=self.alpha_p_)
                sigma = float(_calculate_sigma(y - mod1["fitted"], y.size))

            res.update(self._native_pi(p=res["prob"], mu=res["size"], sigma=sigma, level=level))
        else:
            if self.prediction_intervals is not None:
                res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
            else:
                raise Exception("You must pass `prediction_intervals` or enable native_intervals to compute intervals.")

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
        return self.forecast(y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted)



class SeasonalNaive(_TS):

    def __init__(
        self,
        season_length: int,
        alias: str = "SeasonalNaive",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        r"""Seasonal naive model.

        A method similar to the naive, but uses the last known observation of the same period (e.g. the same month of the previous year) in order to capture seasonal variations.

        References
        ----------
        [Rob J. Hyndman and George Athanasopoulos (2018). "forecasting principles and practice, Simple Methods"](https://otexts.com/fpp3/simple-methods.html#seasonal-na%C3%AFve-method).

        Parameters
        ----------
        season_length : int
            Number of observations per unit of time. Ex: 24 Hourly data.
        alias : str
            Custom name of the model.
        prediction_intervals : Optional[ConformalIntervals]
            Information to compute conformal prediction intervals.
            By default, the model will compute the native prediction
            intervals.
        """
        self.season_length = season_length
        self.alias = alias
        self.prediction_intervals = prediction_intervals

    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
    ):
        r"""Fit the SeasonalNaive model.

        Fit an SeasonalNaive to a time series (numpy array) `y`.

        Parameters
        ----------
        y : numpy.array
            Clean time series of shape (t, ).
        X: array-like
            Optional exogenous of shape (t, n_x).

        Returns
        -------
        self :
            SeasonalNaive fitted model.
        r"""
        y = _ensure_float(y)
        mod = _seasonal_naive(
            y=y,
            season_length=self.season_length,
            h=self.season_length,
            fitted=True,
        )
        mod = dict(mod)
        residuals = y - mod["fitted"]
        mod["sigma"] = _calculate_sigma(residuals, len(y) - self.season_length)
        self.model_ = mod
        self._store_cs(y=y, X=X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        r"""Predict with fitted Naive.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X: array-like
            Optional exogenous of shape (h, n_x).
        level: List[float]
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        forecasts : dict
            Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        mean = _repeat_val_seas(season_vals=self.model_["mean"], h=h)
        res = {"mean": mean}

        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = self._add_predict_conformal_intervals(res, level)
        else:
            k = np.floor(np.arange(h) / self.season_length)
            sigma = self.model_["sigma"]
            sigmah = sigma * np.sqrt(k + 1)
            pred_int = _calculate_intervals(res, level, h, sigmah)
            res = {**res, **pred_int}
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        r"""Access fitted SeasonalNaive insample predictions.

        Parameters
        ----------
        level : List[float]
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        forecasts : dict
            Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        r"""
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            level = sorted(level)
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
        r"""Memory Efficient SeasonalNaive predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Parameters
        ----------
        y : numpy.array
            Clean time series of shape (n, ).
        h : int
            Forecast horizon.
        X : array-like
            Optional insample exogenous of shape (t, n_x).
        X_future : array-like
            Optional exogenous of shape (h, n_x).
        level : List[float]
            Confidence levels (0-100) for prediction intervals.
        fitted : bool
            Whether or not to return insample predictions.

        Returns
        -------
        forecasts : dict
            Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        y = _ensure_float(y)
        out = _seasonal_naive(
            y=y,
            h=h,
            fitted=fitted or (level is not None),
            season_length=self.season_length,
        )
        res = {"mean": out["mean"]}
        if fitted:
            res["fitted"] = out["fitted"]
        if level is not None:
            level = sorted(level)
            if self.prediction_intervals is not None:
                res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
            else:
                k = np.floor(np.arange(h) / self.season_length)
                residuals = y - out["fitted"]
                sigma = _calculate_sigma(residuals, len(y) - self.season_length)
                sigmah = sigma * np.sqrt(k + 1)
                pred_int = _calculate_intervals(out, level, h, sigmah)
                res = {**res, **pred_int}
            if fitted:
                residuals = y - out["fitted"]
                sigma = _calculate_sigma(residuals, len(y) - self.season_length)
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
        r"""Apply SeasonalNaive to new series.

        Parameters
        ----------
        y : numpy.array
            Clean time series of shape (n, ).
        h : int
            Forecast horizon.
        X : array-like
            Optional insample exogenous of shape (t, n_x).
        X_future : array-like
            Optional exogenous of shape (h, n_x).
        level : List[float]
            Confidence levels (0-100) for prediction intervals.
        fitted : bool
            Whether or not to return insample predictions.
        Returns
        -------
        forecasts : dict
            Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions
        """
        return self.forecast(y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted)


def _historic_average(
    y: np.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
) -> Dict[str, np.ndarray]:
    fcst = {"mean": _repeat_val(val=y.mean(), h=h)}
    if fitted:
        fitted_vals = _repeat_val(val=y.mean(), h=len(y))
        fcst["fitted"] = fitted_vals
    return fcst



class HistoricAverage(_TS):
    def __init__(
        self,
        alias: str = "HistoricAverage",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        r"""HistoricAverage model.

        Also known as mean method. Uses a simple average of all past observations.
        Assuming there are $t$ observations, the one-step forecast is given by:

        ``` math
        \hat{y}_{t+1} = \frac{1}{t} \sum_{j=1}^t y_j
        ```

        References:
            - [Rob J. Hyndman and George Athanasopoulos (2018). "Forecasting principles and practice, Simple Methods"](https://otexts.com/fpp3/simple-methods.html).

        Args:
            alias (str): Custom name of the model.
            prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals.
                By default, the model will compute the native prediction
                intervals.
        """
        self.alias = alias
        self.prediction_intervals = prediction_intervals

    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
    ):
        r"""Fit the HistoricAverage model.

        Fit an HistoricAverage to a time series (numpy array) `y`.

        Args:
            y (numpy.array): Clean time series of shape (t, ).
            X (array-like): Optional exogenous of shape (t, n_x).

        Returns:
            self: HistoricAverage fitted model.
        r"""
        y = _ensure_float(y)
        mod = _historic_average(y, h=1, fitted=True)
        mod = dict(mod)
        residuals = y - mod["fitted"]
        mod["sigma"] = _calculate_sigma(residuals, len(residuals) - 1)
        mod["n"] = len(y)
        self.model_ = mod
        self._store_cs(y=y, X=X)
        return self


    def predict(
        self,
        h: int,
        X: Optional[np.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        r"""Predict with fitted HistoricAverage.

        Args:
            h (int): Forecast horizon.
            X (Optional[np.ndarray], optional): Optional exogenous of shape (h, n_x). Defaults to None.
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        mean = _repeat_val(val=self.model_["mean"][0], h=h)
        res = {"mean": mean}

        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = self._add_predict_conformal_intervals(res, level)
        else:
            sigma = self.model_["sigma"]
            sigmah = sigma * np.sqrt(1 + (1 / self.model_["n"]))
            pred_int = _calculate_intervals(res, level, h, sigmah)
            res = {**res, **pred_int}

        return res

    def predict_in_sample(self, level: Optional[List[int]] = None):
        r"""Access fitted HistoricAverage insample predictions.

        Args:
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions.
        """
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            sigma = self.model_["sigma"]
            sigmah = sigma * np.sqrt(1 + (1 / self.model_["n"]))
            res = _add_fitted_pi(res, se=sigmah, level=level)
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
        r"""Memory Efficient HistoricAverage predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (np.ndarray): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (Optional[np.ndarray], optional): Optional insample exogenous of shape (t, n_x). Defaults to None.
            X_future (Optional[np.ndarray], optional): Optional exogenous of shape (h, n_x). Defaults to None.
            level (Optional[List[int]], optional): Confidence levels (0-100) for prediction intervals. Defaults to None.
            fitted (bool, optional): Whether or not to return insample predictions. Defaults to False.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        y = _ensure_float(y)
        out = _historic_average(y=y, h=h, fitted=fitted or (level is not None))
        res = {"mean": out["mean"]}

        if fitted:
            res["fitted"] = out["fitted"]
        if level is not None:
            level = sorted(level)
            if self.prediction_intervals is not None:
                res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
            else:
                residuals = y - out["fitted"]
                sigma = _calculate_sigma(residuals, len(residuals) - 1)
                sigmah = sigma * np.sqrt(1 + (1 / len(y)))
                pred_int = _calculate_intervals(out, level, h, sigmah)
                res = {**res, **pred_int}
            if fitted:
                residuals = y - out["fitted"]
                sigma = _calculate_sigma(residuals, len(residuals) - 1)
                sigmah = sigma * np.sqrt(1 + (1 / len(y)))
                res = _add_fitted_pi(res=res, se=sigmah, level=level)
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
        r"""Apply HistoricAverage to a new series.

        Parameters
        ----------
        y : np.ndarray
            Clean time series of shape (n, ).
        h : int
            Forecast horizon.
        X : Optional[np.ndarray]
            Optional insample exogenous of shape (n, n_x).
        X_future : Optional[np.ndarray]
            Optional exogenous of shape (h, n_x).
        level : Optional[List[int]]
            Confidence levels (0-100) for prediction intervals.
        fitted : bool
            Whether to return insample predictions.

        Returns
        -------
        dict
            Dictionary with entries `mean` (and optionally `fitted` and intervals).
        """
        return self.forecast(
            y=y,
            h=h,
            X=X,
            X_future=X_future,
            level=level,
            fitted=fitted,
        )

