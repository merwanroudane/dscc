"""Optional numba kernels for the three hot loops (DSH filter, GARCH(1,1), DCC likelihood).

If numba is not installed, or ``DSCC_NO_NUMBA=1`` is set, the pure-NumPy code paths
are used; both paths give identical results (tested).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import os

import numpy as np

HAVE_NUMBA = False
if os.environ.get("DSCC_NO_NUMBA", "0") != "1":
    try:
        from numba import njit
        HAVE_NUMBA = True
    except Exception:  # pragma: no cover
        HAVE_NUMBA = False

__all__ = ["HAVE_NUMBA", "dsh_loop", "garch11_h", "dcc_nll"]


def _dsh_loop_py(X, Xd, expU, D0, D11, D21, a, c, egarch, delta_fixed):
    T, k = X.shape
    state = np.zeros((T, k)); delta = np.zeros(T, np.int64); contrib = np.zeros(T)
    for t in range(1, T):
        xp = Xd[t - 1] if delta[t - 1] else X[t - 1]
        z = expU[t - 1].T @ xp
        if egarch:
            zs = z / np.sqrt(np.exp(state[t - 1]))
            d = zs - c
            state[t] = D0 + D11 * state[t - 1] + D21 * (np.abs(d) - a * d)
            lam = np.exp(state[t]); logdet = state[t].sum()
        else:
            state[t] = D0 + D11 * state[t - 1] + D21 * z * z
            lam = state[t]
            if np.any(lam <= 0):
                return state, delta, contrib, False
            logdet = np.log(lam).sum()
        E = expU[t]
        w0 = E.T @ X[t]; w1 = E.T @ Xd[t]
        f0 = np.sum(w0 * w0 / lam); f1 = np.sum(w1 * w1 / lam)
        if delta_fixed[0] >= 0:
            delta[t] = delta_fixed[t]
            contrib[t] = logdet + (f1 if delta[t] else f0)
        elif f1 < f0:
            delta[t] = 1; contrib[t] = logdet + f1
        else:
            contrib[t] = logdet + f0
    return state, delta, contrib, True


def _garch11_h_py(u, w, a, b, h0):
    T = u.size
    h = np.empty(T); h[0] = h0
    for t in range(1, T):
        h[t] = w + a * u[t - 1] ** 2 + b * h[t - 1]
    return h


def _dcc_nll_py(e, Qbar, a, b):
    T, k = e.shape
    Q = Qbar.copy()
    nll = 0.0
    for t in range(T):
        if t > 0:
            Q = (1 - a - b) * Qbar + a * np.outer(e[t - 1], e[t - 1]) + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        L = np.linalg.cholesky(R)
        y = np.linalg.solve(L, e[t])
        nll += 2.0 * np.log(np.diag(L)).sum() + y @ y - e[t] @ e[t]
    return 0.5 * nll


if HAVE_NUMBA:
    @njit(cache=True, error_model="numpy")
    def _dsh_loop_nb(X, Xd, expU, D0, D11, D21, a, c, egarch, delta_fixed):
        T, k = X.shape
        state = np.zeros((T, k)); delta = np.zeros(T, np.int64); contrib = np.zeros(T)
        z = np.empty(k)
        for t in range(1, T):
            use_d = delta[t - 1] == 1
            for i in range(k):
                s = 0.0
                for j in range(k):
                    s += expU[t - 1, j, i] * (Xd[t - 1, j] if use_d else X[t - 1, j])
                z[i] = s
            logdet = 0.0
            for i in range(k):
                if egarch:
                    zs = z[i] / np.sqrt(np.exp(state[t - 1, i]))
                    dd = zs - c[i]
                    state[t, i] = D0[i] + D11[i] * state[t - 1, i] + D21[i] * (abs(dd) - a[i] * dd)
                    logdet += state[t, i]
                else:
                    state[t, i] = D0[i] + D11[i] * state[t - 1, i] + D21[i] * z[i] * z[i]
                    if state[t, i] <= 0.0:
                        return state, delta, contrib, False
                    logdet += np.log(state[t, i])
            f0 = 0.0; f1 = 0.0
            for i in range(k):
                lam = np.exp(state[t, i]) if egarch else state[t, i]
                s0 = 0.0; s1 = 0.0
                for j in range(k):
                    s0 += expU[t, j, i] * X[t, j]
                    s1 += expU[t, j, i] * Xd[t, j]
                f0 += s0 * s0 / lam; f1 += s1 * s1 / lam
            if delta_fixed[0] >= 0:
                delta[t] = delta_fixed[t]
                contrib[t] = logdet + (f1 if delta[t] == 1 else f0)
            elif f1 < f0:
                delta[t] = 1; contrib[t] = logdet + f1
            else:
                contrib[t] = logdet + f0
        return state, delta, contrib, True

    @njit(cache=True, error_model="numpy")
    def _garch11_h_nb(u, w, a, b, h0):
        T = u.size
        h = np.empty(T); h[0] = h0
        for t in range(1, T):
            h[t] = w + a * u[t - 1] ** 2 + b * h[t - 1]
        return h

    @njit(cache=True, error_model="numpy")
    def _dcc_nll_nb(e, Qbar, a, b):
        T, k = e.shape
        Q = Qbar.copy()
        nll = 0.0
        for t in range(T):
            if t > 0:
                for i in range(k):
                    for j in range(k):
                        Q[i, j] = (1 - a - b) * Qbar[i, j] + a * e[t - 1, i] * e[t - 1, j] + b * Q[i, j]
            R = np.empty((k, k))
            for i in range(k):
                for j in range(k):
                    R[i, j] = Q[i, j] / np.sqrt(Q[i, i] * Q[j, j])
            L = np.linalg.cholesky(R)
            y = np.linalg.solve(L, e[t])
            ld = 0.0
            for i in range(k):
                ld += 2.0 * np.log(L[i, i])
            nll += ld + np.dot(y, y) - np.dot(e[t], e[t])
        return 0.5 * nll

    dsh_loop, garch11_h, dcc_nll = _dsh_loop_nb, _garch11_h_nb, _dcc_nll_nb
else:  # pragma: no cover
    dsh_loop, garch11_h, dcc_nll = _dsh_loop_py, _garch11_h_py, _dcc_nll_py
