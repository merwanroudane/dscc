"""Rolling out-of-sample minimum-variance backtest (Sections 5.2-5.4).

Each month: estimate on the last ``window`` days, forecast the daily covariance
matrices for the next ``horizon`` days, average them into one monthly matrix,
form global minimum-variance weights, hold them for the month, then roll forward
by ``horizon`` days.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._model import DSCC, DSCCResults
from .dcc import DCC
from .portfolio import (covariance_summaries, gmv_weights, holding_period_returns, pairwise_tests,
                        performance, turnover_leverage)

__all__ = ["DSCCForecaster", "DCCForecaster", "SampleCovForecaster", "EqualWeight",
           "rolling_backtest", "BacktestResults"]


class DSCCForecaster:
    """DSCC forecaster for :func:`rolling_backtest`.

    ``refit_every`` months the parameters are re-estimated (warm start from the
    previous estimate, plus ``n_starts - 1`` random starts); in between, the
    latest estimate is re-filtered on the new window. The paper re-estimates every
    month; ``refit_every=1`` reproduces that at a higher computing cost.
    """

    def __init__(self, dynamics="egarch", init="average", mode="paper", refit_every=12,
                 n_starts=1, maxiter=200, complement="householder", label=None):
        self.dynamics, self.init, self.mode = dynamics, init, mode
        self.refit_every, self.n_starts, self.maxiter = refit_every, n_starts, maxiter
        self.complement = complement
        self.theta = None
        self.label = label or f"DSCC-{dynamics}" + ("-av" if init == "average" else "")

    def __call__(self, window: pd.DataFrame, month: int, horizon: int):
        m = DSCC(window, self.dynamics, complement=self.complement)
        if self.theta is None or month % self.refit_every == 0:
            r = m.fit(start=self.theta, n_starts=self.n_starts, maxiter=self.maxiter, se=False)
            self.theta = r.theta
        else:
            r = DSCCResults(m, self.theta, se=False)
        return r.forecast(horizon, mode=self.mode, init=self.init).mean_sigma


class DCCForecaster:
    label = "DCC"

    def __call__(self, window, month, horizon):
        return DCC(window).fit().forecast(horizon).mean(0)


class SampleCovForecaster:
    label = "Sample"

    def __call__(self, window, month, horizon):
        return np.cov(window.to_numpy(float), rowvar=False)


class EqualWeight:
    label = "1/N"

    def __call__(self, window, month, horizon):
        return None


@dataclass
class BacktestResults:
    weights: dict
    daily: pd.DataFrame
    monthly: pd.DataFrame
    sigmas: dict
    month_returns: list
    timing: dict = field(default_factory=dict)

    def performance_table(self) -> pd.DataFrame:
        """Table 2: AV, SD (annualised, percent) and IR."""
        return pd.DataFrame({k: performance(self.daily[k].to_numpy()) for k in self.daily}).T

    def turnover_table(self, replication: bool = False) -> pd.DataFrame:
        """Table 4: TO, GL, PL."""
        return pd.DataFrame({k: turnover_leverage(np.array(w), self.month_returns, replication)
                             for k, w in self.weights.items()}).T

    def pairwise(self, lags=None) -> pd.DataFrame:
        """Table 3: HAC one-sided tests on monthly return differences (1/N excluded)."""
        cols = [c for c in self.monthly if c != "1/N"]
        return pairwise_tests(self.monthly[cols], lags)

    def covariance_table(self) -> pd.DataFrame:
        """Tables 5-6: means of v, Delta, d, Delta' and kappa (min/max for Delta', kappa)."""
        out = {}
        for k, S in self.sigmas.items():
            if not S:
                continue
            c = covariance_summaries(S, self.weights[k])
            out[k] = {"v": c.v.mean(), "Delta": c.Delta.mean(), "d": c.d.mean(),
                      "Delta' mean": c.Delta_prime.mean(), "Delta' min": c.Delta_prime.min(),
                      "Delta' max": c.Delta_prime.max(), "kappa mean": c.kappa.mean(),
                      "kappa min": c.kappa.min(), "kappa max": c.kappa.max()}
        return pd.DataFrame(out).T

    def wealth(self, initial: float = 100.0) -> pd.DataFrame:
        """Value of ``initial`` invested at the start (Figures 4-5)."""
        return initial * (1 + self.daily).cumprod()


def rolling_backtest(returns: pd.DataFrame, forecasters, window: int = 1260, horizon: int = 21,
                     holding: str = "buy_and_hold", in_percent: bool = True,
                     n_months: int | None = None, verbose: bool = True) -> BacktestResults:
    """Run the monthly-rebalanced GMV backtest for several covariance forecasters.

    Parameters
    ----------
    returns : (T, N) daily returns with a DatetimeIndex.
    forecasters : list of forecasters (objects with ``label`` and ``__call__``).
    window : estimation window in days (1,260 = five years of 21-day months).
    horizon : holding period / forecast horizon in days (21).
    holding : ``"buy_and_hold"`` (paper text) or ``"constant"`` (authors' code).
    in_percent : the returns are in percent (as in the French library); models are
        estimated on them as given, portfolio returns use decimals.
    n_months : evaluate only the last ``n_months`` months.
    """
    R = returns.astype(float)
    if R.isna().any().any():
        raise ValueError("returns contain NaN; fill or drop them first")
    T, N = R.shape
    starts = list(range(0, T - window - horizon + 1, horizon))
    if n_months is not None:
        starts = starts[-int(n_months):]
    scale = 100.0 if in_percent else 1.0
    labels = [f.label for f in forecasters]
    W = {l: [] for l in labels}
    S = {l: [] for l in labels}
    daily = {l: [] for l in labels}
    idx, month_R, timing = [], [], {l: 0.0 for l in labels}
    for mth, s in enumerate(starts):
        win = R.iloc[s:s + window]
        oos = R.iloc[s + window:s + window + horizon].to_numpy() / scale
        month_R.append(oos)
        idx.extend(R.index[s + window:s + window + horizon])
        for f in forecasters:
            t0 = time.time()
            Sig = f(win, mth, horizon)
            timing[f.label] += time.time() - t0
            w = np.full(N, 1.0 / N) if Sig is None else gmv_weights(Sig)
            if Sig is not None:
                S[f.label].append(Sig)
            W[f.label].append(w)
            daily[f.label].append(holding_period_returns(w, oos, holding))
        if verbose:
            print(f"month {mth + 1}/{len(starts)}  {R.index[s + window].date()}", flush=True)
    dd = pd.DataFrame({l: np.concatenate(daily[l]) for l in labels}, index=pd.DatetimeIndex(idx))
    monthly = pd.DataFrame({l: [np.prod(1 + d) - 1 for d in daily[l]] for l in labels})
    return BacktestResults(W, dd, monthly, S, month_R, timing)
