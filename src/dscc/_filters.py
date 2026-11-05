"""Likelihood filters for the DSH / DSCC class.

One pass through the data computes, for every t, the dynamic skew-symmetric
component Upsilon_t (eq. 2.9), the eigenvalues Lambda_t (GARCH eq. 2.5 or
EGARCH eq. 2.6-2.7), the in-sample MLE of delta_t, and the Gaussian
log-likelihood (2.13) of Abadir and Rockinger (2026) / (3.11) of Abadir (2023).

Parameter-vector layout (identical to the authors' replication code, so their
stored estimates load unchanged), with k variables and l = k(k-1)/2:

    GARCH  : [D0 (k), D11 (k), D21 (k), mu (l), alpha (l), beta (l)]           3k + 3l
    EGARCH : [D0 (k), D11 (k), D21 (k), mu (l), alpha (l), beta (l), a (k), c (k)]  5k + 3l

Timing conventions (verified line by line against the authors' code):

* t = 1 is a dead start-up: Lambda_1 = O (GARCH) or log Lambda_1 = O (EGARCH),
  Upsilon_1 = O, delta_1 = 0, and t = 1 does not enter the likelihood
  (Abadir 2023, after eq. 4.8).
* Lambda_t uses z_{t-1} = e^{-Upsilon_{t-1}} P^{delta_{t-1}} x_{t-1}   (eq. 2.4-2.5).
* delta_t is chosen sequentially: given Lambda_t and Upsilon_t, the t-th summand
  of (2.13) is maximised over delta_t in {0, 1}; ties keep delta_t = 0.
  (Abadir 2023, Section 4.1: "each of the summands ... maximized with respect
  to delta_t"; 2T likelihood evaluations instead of 2^T.)

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._fast import HAVE_NUMBA, dsh_loop
from ._representation import n_offdiag

__all__ = ["FilterOutput", "unpack_theta", "n_params", "dsh_filter", "g_abs"]


@dataclass
class FilterOutput:
    """Output of one filtering pass (all arrays are time-major)."""

    negloglik: float            # 0.5 * sum_t [ sum_i log lam_it + x_t' Sigma_t^{-1} x_t ]
    contributions: np.ndarray   # (T,) the per-t terms (before the 0.5 factor)
    delta: np.ndarray           # (T,) int, in-sample MLE of delta_t
    lam: np.ndarray             # (T, k) eigenvalues Lambda_t (levels)
    ups: np.ndarray             # (T, l) free elements of Upsilon_t
    expU: np.ndarray            # (T, k, k) e^{Upsilon_t}


def n_params(k: int, dynamics: str) -> int:
    l = n_offdiag(k)
    return 3 * k + 3 * l + (2 * k if dynamics == "egarch" else 0)


def unpack_theta(theta: np.ndarray, k: int, dynamics: str) -> dict:
    theta = np.asarray(theta, dtype=float).ravel()
    l = n_offdiag(k)
    if theta.size != n_params(k, dynamics):
        raise ValueError(
            f"theta has {theta.size} elements; {dynamics} with k={k} needs {n_params(k, dynamics)}"
        )
    out = {
        "D0": theta[0:k],
        "D11": theta[k:2 * k],
        "D21": theta[2 * k:3 * k],
        "mu": theta[3 * k:3 * k + l],
        "alpha": theta[3 * k + l:3 * k + 2 * l],
        "beta": theta[3 * k + 2 * l:3 * k + 3 * l],
    }
    if dynamics == "egarch":
        out["a"] = theta[3 * k + 3 * l:4 * k + 3 * l]
        out["c"] = theta[4 * k + 3 * l:5 * k + 3 * l]
    return out


def g_abs(v: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Asymmetric absolute-value function g_i(v) = |v_i - c_i| - a_i (v_i - c_i), eq. (2.7)."""
    d = v - c
    return np.abs(d) - a * d


def dsh_filter(
    theta: np.ndarray,
    X: np.ndarray,
    xi: np.ndarray,
    dynamics: str = "garch",
    store_expU: bool = True,
    delta_fixed: np.ndarray | None = None,
    use_fast: bool = True,
) -> FilterOutput:
    """Run the DSH filter and return the negative log-likelihood and paths.

    Parameters
    ----------
    theta : parameter vector (layout in the module docstring).
    X : (T, k) de-meaned data x_t.
    xi : (T, l) data innovations xi_t = vec(log Q_t), eq. (2.11).
    dynamics : {"garch", "egarch"} for eq. (2.5) or eq. (2.6).
    store_expU : keep e^{Upsilon_t} for every t (needed for forecasting).
    delta_fixed : optional (T,) 0/1 path. When given, delta_t is held at these
        values instead of being maximised; the likelihood is then smooth in theta,
        which is what the profile-likelihood standard errors use.

    Returns
    -------
    FilterOutput

    Examples
    --------
    >>> import numpy as np
    >>> from dscc._representation import xi_from_data
    >>> rng = np.random.default_rng(0); X = rng.standard_normal((30, 2))
    >>> theta = np.r_[[0.02]*2, [0.85]*2, [0.10]*2, -1.0, 0.1, -0.02]
    >>> out = dsh_filter(theta, X, xi_from_data(X))
    >>> bool(np.isfinite(out.negloglik))
    True
    """
    X = np.asarray(X, dtype=float)
    T, k = X.shape
    l = n_offdiag(k)
    p = unpack_theta(theta, k, dynamics)
    D0, D11, D21 = p["D0"], p["D11"], p["D21"]
    mu, alpha, beta = p["mu"], p["alpha"], p["beta"]
    const_v = mu * (1.0 - alpha)
    egarch = dynamics == "egarch"
    if egarch:
        a, c = p["a"], p["c"]

    Xd = X.copy()
    Xd[:, [0, 1]] = X[:, [1, 0]]          # P x_t  (P swaps the first two variates)

    # (2.9): upsilon_t = (1-alpha) mu + alpha upsilon_{t-1} + beta xi_{t-1}, upsilon_1 = 0.
    # Upsilon_t depends neither on Lambda nor on delta, so the whole path and all
    # matrix exponentials are computed once, before the Lambda/delta recursion.
    v = ups_path(const_v, alpha, beta, xi, T)
    expU = expm_skew_batch(v, k)

    if use_fast and HAVE_NUMBA:
        dfix = (np.full(1, -1, np.int64) if delta_fixed is None
                else np.ascontiguousarray(delta_fixed, dtype=np.int64))
        zk = np.zeros(k)
        state, delta, contrib, ok = dsh_loop(
            np.ascontiguousarray(X), np.ascontiguousarray(Xd), expU,
            np.ascontiguousarray(D0, float), np.ascontiguousarray(D11, float),
            np.ascontiguousarray(D21, float),
            np.ascontiguousarray(a, float) if egarch else zk,
            np.ascontiguousarray(c, float) if egarch else zk, egarch, dfix)
        if not ok:
            return FilterOutput(np.inf, np.full(T, np.inf), delta, state, v, expU)
        lam = np.exp(state) if egarch else state
        return FilterOutput(0.5 * contrib.sum(), contrib, delta, lam, v,
                            expU if store_expU else None)

    state = np.zeros((T, k))               # lam_t (garch) or log lam_t (egarch)
    delta = np.zeros(T, dtype=int)
    contrib = np.zeros(T)

    for t in range(1, T):
        # z_{t-1} = e^{-Upsilon_{t-1}} P^{delta_{t-1}} x_{t-1}
        x_prev = Xd[t - 1] if delta[t - 1] else X[t - 1]
        z = expU[t - 1].T @ x_prev
        if egarch:
            # (2.6): log Lambda_t = D0 + D11 log Lambda_{t-1} + D21 g(Lambda_{t-1}^{-1/2} z_{t-1})
            zs = z / np.sqrt(np.exp(state[t - 1]))
            state[t] = D0 + D11 * state[t - 1] + D21 * g_abs(zs, a, c)
            lam_t = np.exp(state[t])
            logdet = state[t].sum()
        else:
            # (2.5): Lambda_t = D0 + D11 Lambda_{t-1} + D21 dg(z_{t-1} z_{t-1}')
            state[t] = D0 + D11 * state[t - 1] + D21 * z * z
            lam_t = state[t]
            if np.any(lam_t <= 0):
                contrib[t:] = np.inf
                return FilterOutput(np.inf, contrib, delta, state, v, expU)
            logdet = np.log(lam_t).sum()

        E = expU[t]
        # quadratic forms x' Sigma_t^{-1} x for delta = 0 and delta = 1
        w0 = E.T @ X[t]
        w1 = E.T @ Xd[t]
        f0 = np.sum(w0 * w0 / lam_t)
        f1 = np.sum(w1 * w1 / lam_t)
        if delta_fixed is not None:
            delta[t] = int(delta_fixed[t])
            contrib[t] = logdet + (f1 if delta[t] else f0)
        elif f1 < f0:
            delta[t] = 1
            contrib[t] = logdet + f1
        else:
            contrib[t] = logdet + f0

    lam = np.exp(state) if egarch else state
    return FilterOutput(0.5 * contrib.sum(), contrib, delta, lam, v,
                        expU if store_expU else None)


def ups_path(const_v, alpha, beta, xi, T: int) -> np.ndarray:
    """Path of the free elements of Upsilon_t from eq. (2.9), with upsilon_1 = 0."""
    l = np.size(alpha)
    v = np.zeros((T, l))
    for t in range(1, T):
        v[t] = const_v + alpha * v[t - 1] + beta * xi[t - 1]
    return v


def expm_skew_batch(v: np.ndarray, k: int) -> np.ndarray:
    """e^{Upsilon_t} for a batch of skew-symmetric matrices.

    For skew-symmetric U, H = iU is Hermitian: H = V diag(w) V^H, hence
    e^U = V diag(e^{-iw}) V^H (exact; Lemma 2.1 of Abadir 2023). One batched
    Hermitian eigendecomposition replaces T calls to a general expm.
    """
    T = v.shape[0]
    U = np.zeros((T, k, k))
    i, j = np.triu_indices(k, 1)
    U[:, i, j] = v
    U[:, j, i] = -v
    w, V = np.linalg.eigh(1j * U)
    E = (V * np.exp(-1j * w)[:, None, :]) @ np.conj(np.swapaxes(V, 1, 2))
    return E.real
