"""DCC(1,1) baseline of Engle (2002), two-step Gaussian QMLE, with multistep forecasts.

It is the comparison model of Abadir and Rockinger (2026), eq. (2.17). The forecast
recursion is the one the authors' code uses for the DCC block:
h_{T+1} = omega + a u_T^2 + b h_T, h_{T+s} = omega + (a+b) h_{T+s-1};
Q_{T+1} = (1-alpha-beta) Qbar + alpha e_T e_T' + beta Q_T,
Q_{T+s} = (1-alpha-beta) Qbar + (alpha+beta) Q_{T+s-1}.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from ._fast import dcc_nll, garch11_h
from ._representation import correlation_from_cov

__all__ = ["garch11_fit", "DCC", "DCCResults"]


def _garch11_nll(p, u):
    w, a, b = p
    if w <= 0 or a < 0 or b < 0 or a + b >= 1:
        return 1e12
    h = garch11_h(u, float(w), float(a), float(b), float(u.var()))
    return 0.5 * np.sum(np.log(h) + u * u / h)


def garch11_fit(u: np.ndarray):
    """Gaussian QMLE of a GARCH(1,1) on a de-meaned series; returns (params, h)."""
    u = np.asarray(u, float)
    v = u.var()
    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.80), (0.03, 0.95)):
        r = optimize.minimize(_garch11_nll, [v * (1 - a0 - b0), a0, b0], args=(u,), method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    w, a, b = best.x
    return np.array([w, a, b]), garch11_h(u, float(w), float(a), float(b), float(v))


def _dcc_nll(p, e, Qbar):
    a, b = p
    if a < 0 or b < 0 or a + b >= 0.9999:
        return 1e12
    return dcc_nll(e, Qbar, float(a), float(b))


class DCC:
    """Engle (2002) DCC(1,1) with GARCH(1,1) margins.

    Examples
    --------
    >>> import numpy as np
    >>> from dscc.dcc import DCC
    >>> r = DCC(np.random.default_rng(0).standard_normal((80, 2))).fit()
    >>> r.forecast(5).shape
    (5, 2, 2)
    """

    def __init__(self, data, demean: bool = True):
        if isinstance(data, pd.DataFrame):
            self.names = [str(c) for c in data.columns]
            Y = data.to_numpy(float)
        else:
            Y = np.asarray(data, float)
            self.names = [f"x{i + 1}" for i in range(Y.shape[1])]
        if not np.all(np.isfinite(Y)):
            raise ValueError("data contains NaN or infinite values")
        self.mean_ = Y.mean(0) if demean else np.zeros(Y.shape[1])
        self.X = Y - self.mean_

    def fit(self) -> "DCCResults":
        X = self.X
        T, k = X.shape
        garch, H = [], np.empty((T, k))
        for i in range(k):
            p, h = garch11_fit(X[:, i])
            garch.append(p); H[:, i] = h
        e = np.ascontiguousarray(X / np.sqrt(H))
        Qbar = e.T @ e / T
        best = None
        for s in ((0.02, 0.95), (0.05, 0.90), (0.01, 0.98)):
            r = optimize.minimize(_dcc_nll, s, args=(e, Qbar), method="Nelder-Mead",
                                  options={"xatol": 1e-7, "fatol": 1e-7})
            if best is None or r.fun < best.fun:
                best = r
        return DCCResults(self, np.array(garch), best.x, H, e, Qbar)


class DCCResults:
    def __init__(self, model, garch, ab, H, e, Qbar):
        self.model, self.garch, self.ab = model, garch, ab
        self.h, self.e, self.Qbar = H, e, Qbar
        a, b = ab
        T, k = e.shape
        Q = np.empty((T, k, k)); Q[0] = Qbar
        for t in range(1, T):
            Q[t] = (1 - a - b) * Qbar + a * np.outer(e[t - 1], e[t - 1]) + b * Q[t - 1]
        self.Q = Q
        cols = [f"{p}[{n}]" for n in model.names for p in ("omega", "a", "b")]
        self.params = pd.Series(np.r_[garch.ravel(), ab], index=cols + ["alpha_dcc", "beta_dcc"])

    def correlation(self, t=None):
        R = np.array([correlation_from_cov(q) for q in self.Q])
        return R if t is None else R[t]

    def sigma(self, t=None):
        R = self.correlation()
        s = np.sqrt(self.h)
        S = R * s[:, :, None] * s[:, None, :]
        return S if t is None else S[t]

    def forecast(self, H: int = 21) -> np.ndarray:
        """(H, k, k) forecasts of Sigma_{T+1..T+H}."""
        a, b = self.ab
        w, ga, gb = self.garch.T
        X, e, Qbar = self.model.X, self.e, self.Qbar
        k = X.shape[1]
        hT, uT = self.h[-1], X[-1]
        hf = np.empty((H, k)); Qf = np.empty((H, k, k))
        hf[0] = w + ga * uT ** 2 + gb * hT
        Qf[0] = (1 - a - b) * Qbar + a * np.outer(e[-1], e[-1]) + b * self.Q[-1]
        for s in range(1, H):
            hf[s] = w + (ga + gb) * hf[s - 1]
            Qf[s] = (1 - a - b) * Qbar + (a + b) * Qf[s - 1]
        R = np.array([correlation_from_cov(q) for q in Qf])
        sd = np.sqrt(hf)
        return R * sd[:, :, None] * sd[:, None, :]
