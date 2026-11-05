"""Global minimum-variance portfolios and the evaluation criteria of Section 5.

Abadir and Rockinger (2026), Sections 5.3-5.4, following De Nard, Ledoit and
Wolf (2021, J. Financial Econometrics 19, 236-257).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ._representation import correlation_from_cov

__all__ = [
    "gmv_weights", "factor_covariance", "holding_period_returns", "performance",
    "turnover_leverage", "newey_west_lags", "hac_mean_test", "pairwise_tests",
    "covariance_summaries", "HACTest",
]


def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """w = Sigma^{-1} iota / (iota' Sigma^{-1} iota)  (Section 5.3)."""
    Sigma = np.asarray(Sigma, float)
    x = np.linalg.solve(Sigma, np.ones(Sigma.shape[0]))
    return x / x.sum()


def factor_covariance(B: np.ndarray, Sigma_f: np.ndarray, Sigma_u: np.ndarray) -> np.ndarray:
    """Sigma_r = B' Sigma_f B + Sigma_u for the factor model (5.1); B is k x N."""
    return B.T @ Sigma_f @ B + Sigma_u


def holding_period_returns(w: np.ndarray, R: np.ndarray, holding: str = "buy_and_hold") -> np.ndarray:
    """Daily portfolio returns over one holding period.

    Parameters
    ----------
    w : (N,) weights chosen at the start of the period.
    R : (H, N) simple returns over the period (decimal or percent, used as given
        for ``"constant"``; must be decimal for ``"buy_and_hold"``).
    holding : ``"buy_and_hold"`` (Section 5.4: weights drift with prices during
        the month) or ``"constant"`` (weights reset every day, r_p,t = R_t w --
        what the authors' ``B_1_Table2.m`` computes).
    """
    w = np.asarray(w, float); R = np.asarray(R, float)
    if holding == "constant":
        return R @ w
    if holding != "buy_and_hold":
        raise ValueError("holding must be 'buy_and_hold' or 'constant'")
    growth = np.vstack([np.ones(R.shape[1]), np.cumprod(1.0 + R, axis=0)])   # (H+1, N)
    value = growth @ w                                                       # portfolio value path, V_0 = 1
    return value[1:] / value[:-1] - 1.0


def performance(daily: np.ndarray, periods: int = 252, scale: float = 100.0) -> dict:
    """AV, SD and IR of daily out-of-sample returns (Section 5.4 a-c).

    AV = periods * mean, SD = sqrt(periods) * std (ddof = 1), IR = AV / SD;
    AV and SD are multiplied by ``scale`` (percent when returns are decimal).
    """
    r = np.asarray(daily, float)
    av = periods * r.mean(axis=0)
    sd = np.sqrt(periods) * r.std(axis=0, ddof=1)
    return {"AV": scale * av, "SD": scale * sd, "IR": av / sd}


def turnover_leverage(W: np.ndarray, R_list, replication: bool = False) -> dict:
    """TO, GL and PL (Section 5.4 / Table 4).

    Parameters
    ----------
    W : (M, N) weights chosen at the start of each of M months.
    R_list : sequence of M (H, N) arrays of daily simple returns (decimal), month by month.
    replication : reproduce the authors' ``B_3_Table4.m``: the hold portfolio of
        month h is w_{h-1} drifted with the returns of month h (instead of h - 1),
        growth is exp(sum r) (instead of prod(1 + r)), and the mean of TO runs
        over all M months with TO_1 = 0.

    Returns
    -------
    dict with TO = (1/(M-1)) sum_h ||w_{h+1} - w_h^hold||_1, GL = mean ||w_h||_1,
    PL = mean share of negative weights.
    """
    W = np.asarray(W, float)
    M, N = W.shape
    to = np.zeros(M)
    for h in range(1, M):
        Rm = np.asarray(R_list[h] if replication else R_list[h - 1], float)
        g = np.exp(Rm.sum(0)) if replication else np.prod(1.0 + Rm, axis=0)
        hold = W[h - 1] * g
        hold = hold / hold.sum()
        to[h] = np.abs(W[h] - hold).sum()
    TO = to.mean() if replication else to[1:].mean()
    return {"TO": TO, "GL": np.abs(W).sum(1).mean(), "PL": (W < 0).mean()}


def newey_west_lags(T: int) -> int:
    """floor(4 (T/100)^{2/9}): the automatic Bartlett bandwidth used by the authors (5 lags for T = 447)."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


@dataclass
class HACTest:
    alpha: float
    se: float
    tstat: float
    pvalue: float      # one-sided, H0: alpha <= 0
    lags: int
    nobs: int


def hac_mean_test(d: np.ndarray, lags: int | None = None) -> HACTest:
    """Regress d_t = alpha + e_t and test H0: alpha <= 0 with a Newey-West (Bartlett) variance.

    The p-value is one-sided, from a t distribution with T - 1 degrees of freedom,
    as in the authors' ``ols_hac.m`` (Table 3).
    """
    d = np.asarray(d, float).ravel()
    T = d.size
    L = newey_west_lags(T) if lags is None else int(lags)
    u = d - d.mean()
    S = u @ u
    for l in range(1, L + 1):
        S += 2 * (1 - l / (L + 1)) * (u[l:] @ u[:-l])
    se = np.sqrt(S) / T
    t = d.mean() / se
    return HACTest(d.mean(), se, t, float(stats.t.sf(t, T - 1)), L, T)


def pairwise_tests(monthly: pd.DataFrame, lags: int | None = None) -> pd.DataFrame:
    """Table 3: for every pair, Model 1 = the one with the higher cumulative return.

    ``monthly`` holds one column of monthly returns per model.
    """
    cols = list(monthly.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if monthly[b].sum() > monthly[a].sum():
                a, b = b, a
            r = hac_mean_test(monthly[a].to_numpy() - monthly[b].to_numpy(), lags)
            rows.append({"Model 1": a, "Model 2": b, "alpha": r.alpha, "t-ratio": r.tstat,
                         "p-value": r.pvalue, "lags": r.lags})
    return pd.DataFrame(rows)


def covariance_summaries(Sigmas, weights=None) -> pd.DataFrame:
    """Per-period v_h, Delta_h (5.2), d_h, Delta'_h (5.3) and kappa_h (Tables 5-6).

    v_h = tr(Sigma)/N;  Delta_h = tr(Sigma)/N - |Sigma|^{1/N};
    d_h = ||w_h - iota/N||_1;  Delta'_h = 1 - |Omega|^{1/N};  kappa_h = cond(Omega).
    """
    rows = []
    for h, S in enumerate(Sigmas):
        S = np.asarray(S, float)
        N = S.shape[0]
        Om = correlation_from_cov(S)
        gv = np.exp(np.linalg.slogdet(S)[1] / N)
        gvo = np.exp(np.linalg.slogdet(Om)[1] / N)
        w = gmv_weights(S) if weights is None else np.asarray(weights[h], float)
        rows.append({"v": np.trace(S) / N, "Delta": np.trace(S) / N - gv,
                     "d": np.abs(w - 1.0 / N).sum(), "Delta_prime": 1.0 - gvo,
                     "kappa": np.linalg.cond(Om)})
    return pd.DataFrame(rows)
