"""DSCC / DSCH model: estimation, filtering and multistep forecasting.

Abadir, K. M. and M. Rockinger (2026). Dynamic spectral conditional correlations.
Econometrics Journal. https://doi.org/10.1093/ectj/utag017
Abadir, K. M. (2023). Explicit minimal representation of variance matrices, and its
implication for dynamic volatility models. Econometrics Journal 26, 88-104.
https://doi.org/10.1093/ectj/utac023

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ._filters import FilterOutput, dsh_filter, expm_skew_batch, g_abs, n_params, unpack_theta
from ._representation import (
    correlation_from_cov,
    n_offdiag,
    permutation_matrix,
    xi_from_data,
)

__all__ = ["DSCC", "DSCCResults", "DSCCForecast", "default_start", "param_names"]


# --------------------------------------------------------------------------- helpers
def param_names(k: int, dynamics: str, names=None) -> list[str]:
    names = list(names) if names is not None else [f"x{i + 1}" for i in range(k)]
    out = [f"D0[{n}]" for n in names] + [f"D11[{n}]" for n in names] + [f"D21[{n}]" for n in names]
    pairs = [f"{names[i]},{names[j]}" for i in range(k) for j in range(i + 1, k)]
    out += [f"mu[{p}]" for p in pairs] + [f"alpha[{p}]" for p in pairs] + [f"beta[{p}]" for p in pairs]
    if dynamics == "egarch":
        out += [f"a[{n}]" for n in names] + [f"c[{n}]" for n in names]
    return out


def default_start(X: np.ndarray, dynamics: str = "garch") -> np.ndarray:
    """Starting values from eq. (5.1) of Abadir (2023).

    GARCH: (D0, D11, D21) = (0.02, 0.85, 0.10) per eigenvalue and
    (mu, alpha, beta) = (-1, 0.1, -0.02) per skew-symmetric element.
    EGARCH (not given in the paper): D11 = 0.85, D21 = 0.10, a = 0, c = 0 and D0
    set so that the implied mean of log Lambda matches log of the sample variances.
    """
    k = X.shape[1]
    l = n_offdiag(k)
    ups = np.r_[np.full(l, -1.0), np.full(l, 0.1), np.full(l, -0.02)]
    if dynamics == "garch":
        return np.r_[np.full(k, 0.02), np.full(k, 0.85), np.full(k, 0.10), ups]
    lv = np.log(np.var(X, axis=0))
    eg = np.sqrt(2.0 / np.pi)             # E|eps| for a standard normal
    D0 = 0.15 * lv - 0.10 * eg
    return np.r_[D0, np.full(k, 0.85), np.full(k, 0.10), ups, np.zeros(k), np.zeros(k)]


def _bounds(k: int, dynamics: str):
    l = n_offdiag(k)
    big = None
    if dynamics == "garch":
        b = [(1e-6, big)] * k + [(0.0, 0.9999)] * k + [(0.0, 0.9999)] * k
    else:
        b = [(big, big)] * k + [(-0.9999, 0.9999)] * k + [(big, big)] * k
    b += [(-np.pi, np.pi)] * l + [(-0.9999, 0.9999)] * l + [(big, big)] * l
    if dynamics == "egarch":
        b += [(-1.0, 1.0)] * k + [(big, big)] * k
    return b


def _constraints(k: int, dynamics: str):
    if dynamics != "garch":
        return []
    n = n_params(k, "garch")
    A = np.zeros((k, n))
    for i in range(k):
        A[i, k + i] = -1.0       # -(D11_i + D21_i) + 0.9999 >= 0
        A[i, 2 * k + i] = -1.0
    return [{"type": "ineq", "fun": lambda th, A=A: A @ th + 0.9999, "jac": lambda th, A=A: A}]


def _num_hessian(f, x, eps=1e-4):
    n = x.size
    H = np.zeros((n, n))
    h = eps * np.maximum(1.0, np.abs(x))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        fpp = f(x + ei); fmm = f(x - ei)
        H[i, i] = (fpp - 2 * f0 + fmm) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(n); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


# --------------------------------------------------------------------------- model
class DSCC:
    """Dynamic spectral conditional heteroskedasticity / correlation model.

    Sigma_t = P^{delta_t} e^{Upsilon_t} Lambda_t e^{-Upsilon_t} P^{delta_t}      (2.12)
    with Lambda_t from GARCH(1,1) (2.5) or EGARCH(1,1) (2.6)-(2.7), the free
    elements of Upsilon_t from ARMA(1,1) (2.9) driven by xi_t = vec(log Q_t)
    (2.10)-(2.11), and the conditional correlation Omega_t from (2.14).

    Parameters
    ----------
    data : (T, k) array or DataFrame of returns. Column names are carried into
        every table.
    dynamics : {"garch", "egarch"}
        Eigenvalue dynamics: eq. (2.5) ("DSCC-lambda_garch") or eq. (2.6)
        ("DSCC-lambda_egarch").
    demean : bool, default True
        Subtract the sample mean, as in Abadir (2023, Section 4.2) and the
        authors' code.
    complement : {"householder", "eig"}
        How the orthogonal complement Q2t is formed (see
        :func:`dscc.orthogonal_from_data`). Irrelevant for k = 2.
    xi : optional (T, l) array
        Pre-computed xi_t. Supplying the series produced by the authors'
        MATLAB code reproduces their likelihood exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from dscc import DSCC
    >>> X = np.random.default_rng(1).standard_normal((60, 2))
    >>> res = DSCC(X).fit(maxiter=30)
    >>> res.sigma(-1).shape
    (2, 2)
    """

    def __init__(self, data, dynamics: str = "garch", demean: bool = True,
                 complement: str = "householder", xi=None):
        if dynamics not in ("garch", "egarch"):
            raise ValueError("dynamics must be 'garch' or 'egarch'")
        if isinstance(data, pd.DataFrame):
            self.names = [str(c) for c in data.columns]
            self.index = data.index
            Y = data.to_numpy(dtype=float)
        else:
            Y = np.asarray(data, dtype=float)
            if Y.ndim != 2:
                raise ValueError("data must be a (T, k) array")
            self.names = [f"x{i + 1}" for i in range(Y.shape[1])]
            self.index = pd.RangeIndex(Y.shape[0])
        T, k = Y.shape
        if k < 2:
            raise ValueError("data must have at least k = 2 columns")
        if T <= 10 * k:
            raise ValueError(f"data has T = {T} rows; at least {10 * k + 1} are needed for k = {k}")
        if not np.all(np.isfinite(Y)):
            raise ValueError("data contains NaN or infinite values; clean or fill them first")
        self.k, self.T, self.dynamics = k, T, dynamics
        self.mean_ = Y.mean(axis=0) if demean else np.zeros(k)
        self.X = Y - self.mean_
        self.complement = complement
        if xi is None:
            self.xi = xi_from_data(self.X, complement)
        else:
            xi = np.asarray(xi, dtype=float)
            if xi.shape != (T, n_offdiag(k)):
                raise ValueError(f"xi must have shape {(T, n_offdiag(k))}, got {xi.shape}")
            self.xi = xi
        self.param_names = param_names(k, dynamics, self.names)

    # ---- likelihood -------------------------------------------------------
    def loglike(self, theta) -> float:
        """Gaussian log-likelihood (2.13), constant term included."""
        out = dsh_filter(theta, self.X, self.xi, self.dynamics, store_expU=False)
        return -out.negloglik - 0.5 * (self.T - 1) * self.k * np.log(2 * np.pi)

    def filter(self, theta) -> FilterOutput:
        return dsh_filter(theta, self.X, self.xi, self.dynamics)

    # ---- estimation -------------------------------------------------------
    def fit(self, start=None, n_starts: int = 1, seed: int | None = 0,
            method: str = "SLSQP", maxiter: int = 500, se: bool = True,
            polish: bool = True, polish_maxiter: int = 20, verbose: bool = False) -> "DSCCResults":
        """Maximum-likelihood estimation of theta and in-sample delta_t.

        Parameters
        ----------
        start : starting vector; defaults to eq. (5.1) of Abadir (2023).
        n_starts : total number of starts. Extra starts are random draws in a wide
            admissible box around ``start``, as in Section 5.5 of Abadir and
            Rockinger (2026) (one deterministic start plus nine random ones).
        seed : seed of the random starts.
        method : ``scipy.optimize.minimize`` method ("SLSQP" honours the GARCH
            stationarity constraints; "Powell" / "Nelder-Mead" are derivative-free
            fall-backs, bounds only).
        se : compute profile-likelihood standard errors (delta held at its MLE).
        polish : after the gradient run, polish the best point found with the
            derivative-free Powell method (robust to the jumps the delta_t search
            creates in the likelihood surface). ``polish_maxiter`` caps its sweeps.

        Notes
        -----
        The returned estimate is the best admissible point evaluated across all
        starts and both stages, so the fit can never be worse than its start.
        """
        k, dyn = self.k, self.dynamics
        x0 = default_start(self.X, dyn) if start is None else np.asarray(start, float).ravel()
        if x0.size != n_params(k, dyn):
            raise ValueError(f"start must have {n_params(k, dyn)} elements")
        bounds, cons = _bounds(k, dyn), _constraints(k, dyn)
        rng = np.random.default_rng(seed)
        starts = [x0] + [_random_start(x0, k, dyn, rng) for _ in range(max(0, n_starts - 1))]

        lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
        tracker = {"fun": np.inf, "x": None}

        def obj(th):
            # The delta_t search makes the likelihood piecewise smooth, so gradient
            # methods can step to worse points; every evaluation is tracked and the
            # best admissible point ever seen is what the run returns.
            th = np.clip(th, lo, hi)
            if dyn == "garch" and np.any(th[k:2 * k] + th[2 * k:3 * k] > 0.9999):
                return 1e12
            v = dsh_filter(th, self.X, self.xi, dyn, store_expU=False).negloglik
            v = v if np.isfinite(v) else 1e12
            if v < tracker["fun"]:
                tracker["fun"], tracker["x"] = v, th.copy()
            return v

        best_fun, best_x, best_opt, runs = np.inf, None, None, []
        for s_i, s in enumerate(starts):
            tracker["fun"], tracker["x"] = np.inf, None
            obj(s)
            kw = {"maxiter": maxiter}
            if method == "SLSQP":
                r = optimize.minimize(obj, s, method="SLSQP", bounds=bounds, constraints=cons, options=kw)
            else:
                r = optimize.minimize(obj, s, method=method, bounds=bounds, options=kw)
            if polish:
                r2 = optimize.minimize(obj, tracker["x"], method="Powell", bounds=bounds,
                                       options={"maxiter": polish_maxiter, "xtol": 1e-6, "ftol": 1e-9})
                r = r2 if r2.fun <= r.fun else r
            fun, x = tracker["fun"], tracker["x"]
            runs.append({"start": s_i, "neg_loglik": fun, "converged": bool(r.success), "message": str(r.message)})
            if verbose:
                print(f"start {s_i + 1}/{len(starts)}: -loglik(core) = {fun:.4f}  ({r.message})")
            if fun < best_fun:
                best_fun, best_x, best_opt = fun, x, r
        return DSCCResults(self, best_x, best_opt, runs, se=se)


def _random_start(x0, k, dyn, rng):
    l = n_offdiag(k)
    th = x0.copy()
    if dyn == "garch":
        th[:k] = rng.uniform(0.005, 0.2, k)
        d11 = rng.uniform(0.6, 0.95, k)
        th[k:2 * k] = d11
        th[2 * k:3 * k] = rng.uniform(0.01, 1 - d11 - 1e-3)
    else:
        th[:k] = x0[:k] + rng.normal(0, 0.2, k)
        th[k:2 * k] = rng.uniform(0.6, 0.95, k)
        th[2 * k:3 * k] = rng.uniform(0.02, 0.3, k)
        th[3 * k + 3 * l:4 * k + 3 * l] = rng.uniform(-0.5, 0.5, k)
        th[4 * k + 3 * l:] = rng.normal(0, 0.05, k)
    th[3 * k:3 * k + l] = rng.uniform(-np.pi, np.pi, l)
    th[3 * k + l:3 * k + 2 * l] = rng.uniform(-0.5, 0.9, l)
    th[3 * k + 2 * l:3 * k + 3 * l] = rng.normal(0, 0.1, l)
    return th


# --------------------------------------------------------------------------- results
class DSCCResults:
    """Fitted DSCC model.

    Attributes
    ----------
    params : pandas.Series of estimates (names follow the variables).
    bse, tvalues, pvalues : profile-likelihood standard errors and Wald statistics.
    loglik, aic, bic : fit statistics.
    delta : in-sample MLE of delta_t;  psi : estimate of Pr(delta = 0).
    lam : (T, k) eigenvalues;  ups : (T, l) skew-symmetric elements.
    """

    def __init__(self, model: DSCC, theta, opt=None, runs=None, se: bool = True):
        self.model = model
        self.theta = np.asarray(theta, float)
        self.opt, self.runs = opt, runs or []
        self.fout = model.filter(self.theta)
        self.k, self.T, self.dynamics = model.k, model.T, model.dynamics
        self.params = pd.Series(self.theta, index=model.param_names, name="coef")
        self.loglik = -self.fout.negloglik - 0.5 * (self.T - 1) * self.k * np.log(2 * np.pi)
        m = self.theta.size + (self.T - 1)          # theta plus one binary delta_t per t
        self.n_params = self.theta.size
        self.aic = -2 * self.loglik + 2 * self.theta.size
        self.bic = -2 * self.loglik + np.log(self.T - 1) * self.theta.size
        self.delta = self.fout.delta
        self.psi = 1.0 - self.delta[1:].mean()      # Pr(delta = 0), dead start t = 1 excluded
        self.lam, self.ups, self.expU = self.fout.lam, self.fout.ups, self.fout.expU
        self.bse = pd.Series(np.nan, index=self.params.index, name="std err")
        if se:
            self._compute_se()
        self.tvalues = (self.params / self.bse).rename("t")
        self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(self.tvalues)), index=self.params.index, name="P>|t|")
        del m

    def _compute_se(self):
        d = self.delta
        f = lambda th: dsh_filter(th, self.model.X, self.model.xi, self.dynamics,
                                  store_expU=False, delta_fixed=d).negloglik
        try:
            H = _num_hessian(f, self.theta)
            cov = np.linalg.pinv(0.5 * (H + H.T))
            se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
        except Exception:                                  # pragma: no cover
            se = np.full(self.theta.size, np.nan)
        # parameters at a bound have no Wald standard error
        for i, (lo, hi) in enumerate(_bounds(self.k, self.dynamics)):
            if (lo is not None and self.theta[i] - lo < 1e-6) or (hi is not None and hi - self.theta[i] < 1e-6):
                se[i] = np.nan
        self.bse[:] = se

    # ---- conditional moments ---------------------------------------------
    def sigma(self, t: int | None = None) -> np.ndarray:
        """Conditional variance Sigma_t (2.12); all t if ``t`` is None -> (T, k, k)."""
        lam, E, d = self.lam, self.expU, self.delta
        P = permutation_matrix(self.k)
        idx = range(self.T) if t is None else [t]
        out = []
        for s in idx:
            S = (E[s] * lam[s]) @ E[s].T
            if d[s]:
                S = P @ S @ P
            out.append(S)
        out = np.array(out)
        return out if t is None else out[0]

    def correlation(self, t: int | None = None) -> np.ndarray:
        """Conditional correlation Omega_t (2.14)."""
        S = self.sigma(t)
        if S.ndim == 2:
            return correlation_from_cov(S)
        return np.array([correlation_from_cov(s) for s in S])

    def std_resid(self) -> np.ndarray:
        """epsilon_t = Sigma_t^{-1/2} x_t (2.2), using Sigma^{-1/2} from (2.3)."""
        P = permutation_matrix(self.k)
        out = np.zeros_like(self.model.X)
        for t in range(1, self.T):
            x = P @ self.model.X[t] if self.delta[t] else self.model.X[t]
            E = self.expU[t]
            w = E @ ((E.T @ x) / np.sqrt(self.lam[t]))
            out[t] = P @ w if self.delta[t] else w
        return out

    # ---- forecasting ------------------------------------------------------
    def forecast(self, H: int = 21, mode: str = "paper", init: str = "last",
                 xi_window: int | None = None) -> "DSCCForecast":
        """Multistep forecasts Sigma_{T+h}, h = 1..H (Section 4).

        Parameters
        ----------
        H : horizon (21 = one trading month in the paper).
        mode : {"paper", "replication"}
            ``"paper"`` (default) follows Section 4: Upsilon_{T+h} is iterated with
            (2.9), Lambda^0 and Lambda^1 with (4.1) (data term frozen at z_T), the
            psi-weighted Lambda feeds the next step, and
            Sigma_{T+h} = psi Sigma^0 + (1-psi) Sigma^1 for every h.
            ``"replication"`` reproduces the authors' MATLAB scripts exactly:
            e^{Upsilon} is frozen at its time-T value for all h (the forecast of
            Upsilon is computed but not used there), and the h = 1 weights are
            the last in-sample delta_T rather than psi.
        init : {"last", "average"}
            ``"average"`` is the "-av" variant: the data-dependent term uses the
            average over the last ``xi_window`` days of e^{Upsilon_t}, of
            Lambda_t^{-1/2} (EGARCH) and of x_t, and xi_t is averaged over the same
            window, exactly as in the authors' ``A_4_EGARCHAV.m``.
        xi_window : averaging window, default ``H``.

        Returns
        -------
        DSCCForecast with ``sigma`` (H, k, k), ``correlation`` and ``mean_sigma``
        (the average over the horizon used for monthly portfolio allocation).
        """
        if mode not in ("paper", "replication"):
            raise ValueError("mode must be 'paper' or 'replication'")
        if init not in ("last", "average"):
            raise ValueError("init must be 'last' or 'average'")
        k, T, dyn = self.k, self.T, self.dynamics
        p = unpack_theta(self.theta, k, dyn)
        P = permutation_matrix(k)
        X = self.model.X
        Xd = X.copy(); Xd[:, [0, 1]] = X[:, [1, 0]]
        w = H if xi_window is None else int(xi_window)
        TT = T - 1
        psi1 = 1.0 - self.psi if mode == "paper" else self.delta.mean()   # Pr(delta = 1)
        # xi used to iterate (2.9) forward
        xi_bar = self.model.xi[TT - w + 1:TT + 1].mean(0) if init == "average" else self.model.xi[TT]
        if mode == "replication" and init == "last":
            xi_bar = self.model.xi[TT - w:TT + 1].mean(0)      # authors' A_2/A_3 window (w + 1 points)
        # data-dependent term at T
        if init == "average":
            ET = self.expU[TT - w + 1:TT + 1].mean(0)
            x0 = X[TT - w + 1:TT + 1].mean(0); x1 = Xd[TT - w + 1:TT + 1].mean(0)
            lam_isq = (1.0 / np.sqrt(self.lam[TT - w + 1:TT + 1])).mean(0)
        else:
            ET = self.expU[TT]
            x0, x1 = X[TT], Xd[TT]
            lam_isq = 1.0 / np.sqrt(self.lam[TT])
        z0, z1 = ET.T @ x0, ET.T @ x1
        if dyn == "garch":
            q0 = p["D21"] * z0 * z0; q1 = p["D21"] * z1 * z1
            state_T = self.lam[TT]
        else:
            q0 = p["D21"] * g_abs(lam_isq * z0, p["a"], p["c"])
            q1 = p["D21"] * g_abs(lam_isq * z1, p["a"], p["c"])
            state_T = np.log(self.lam[TT])
        # Upsilon path over the horizon, eq. (2.9) with xi frozen at xi_bar
        const_v = p["mu"] * (1 - p["alpha"])
        vf = np.zeros((H, n_offdiag(k)))
        prev = self.ups[TT]
        for h in range(H):
            vf[h] = const_v + p["alpha"] * prev + p["beta"] * xi_bar
            prev = vf[h]
        Ef = expm_skew_batch(vf, k) if mode == "paper" else np.repeat(ET[None], H, 0)

        sig = np.zeros((H, k, k)); L0 = np.zeros((H, k)); L1 = np.zeros((H, k))
        state = state_T
        for h in range(H):
            s0 = p["D0"] + p["D11"] * state + q0          # (4.1) with delta = 0
            s1 = p["D0"] + p["D11"] * state + q1          # (4.1) with delta = 1
            wgt = float(self.delta[TT]) if (mode == "replication" and h == 0) else psi1
            state = wgt * s1 + (1 - wgt) * s0
            l0, l1 = (s0, s1) if dyn == "garch" else (np.exp(s0), np.exp(s1))
            L0[h], L1[h] = l0, l1
            E = Ef[h]
            S0 = (E * l0) @ E.T
            S1 = P @ ((E * l1) @ E.T) @ P
            sig[h] = wgt * S1 + (1 - wgt) * S0
        return DSCCForecast(sig, L0, L1, vf, 1.0 - psi1, mode, init, self.model.names)

    # ---- tables -----------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        ci = 1.959963984540054 * self.bse
        return pd.DataFrame({"coef": self.params, "std err": self.bse, "t": self.tvalues,
                             "P>|t|": self.pvalues, "[0.025": self.params - ci, "0.975]": self.params + ci})

    def summary(self) -> str:
        from ._tables import results_summary
        return results_summary(self)

    def to_latex(self, caption="DSCC maximum-likelihood estimates", label="tab:dscc") -> str:
        from ._tables import to_latex
        return to_latex(self, caption, label)

    def to_markdown(self) -> str:
        from ._tables import to_markdown
        return to_markdown(self)

    def plot_correlations(self, **kw):
        from ._plots import plot_correlations
        return plot_correlations(self, **kw)

    def plot_eigenvalues(self, **kw):
        from ._plots import plot_eigenvalues
        return plot_eigenvalues(self, **kw)

    def __repr__(self):
        return f"<DSCCResults {self.dynamics} k={self.k} T={self.T} loglik={self.loglik:.3f}>"


@dataclass
class DSCCForecast:
    sigma: np.ndarray
    lam0: np.ndarray
    lam1: np.ndarray
    ups: np.ndarray
    psi: float
    mode: str
    init: str
    names: list = field(default_factory=list)

    @property
    def correlation(self) -> np.ndarray:
        return np.array([correlation_from_cov(s) for s in self.sigma])

    @property
    def mean_sigma(self) -> np.ndarray:
        """Average of the daily forecasts over the horizon (Section 5.2)."""
        return self.sigma.mean(0)
