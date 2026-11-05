"""Span of correlation models: elliptope volumes and slice coverage (Section 3).

Abadir and Rockinger (2026) benchmark how much of the space E_k of positive-
definite correlation matrices a dynamic model can reach. For k = 3 they slice
the elliptope along one correlation, measure the area of the convex hull of the
simulated model correlations in every slice, and compare it with the maximal
area allowed by positive definiteness (Table 1, Figures 1-3).

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special
from scipy.signal import lfilter
from scipy.spatial import ConvexHull

from ._representation import compose_sigma, correlation_from_cov, n_offdiag

__all__ = [
    "elliptope_volume", "ellipse_slice_area", "slab_projection_area", "elliptope_boundary",
    "equicorrelation", "toeplitz_corr", "uniform_perturbation",
    "simulate_dcc", "offdiag_lower", "slice_coverage", "coverage_table", "dscc_span_draws",
]


# ------------------------------------------------------------------ analytics
def elliptope_volume(k: int) -> float:
    """Exact volume of E_k in the k(k-1)/2 off-diagonal coordinates.

    Uses the normalising constant of the LKJ density with eta = 1
    (Lewandowski, Kurowicka and Joe 2009, J. Multivariate Analysis 100, 1989-2001):
    vol(E_k) = 2^{sum_{i=1}^{k-1} (k-i)^2} prod_{i=1}^{k-1} B((k-i+1)/2, (k-i+1)/2)^{k-i}.
    For k = 3 this is pi^2/2, eq. (3.2).
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    logv = 0.0
    for i in range(1, k):
        m = k - i
        logv += m * m * np.log(2.0) + m * special.betaln((m + 1) / 2.0, (m + 1) / 2.0)
    return float(np.exp(logv))


def ellipse_slice_area(c: float | np.ndarray) -> np.ndarray:
    """Area of the section of E_3 at a fixed correlation c: pi * sqrt(1 - c^2).

    At fixed omega_3 = c, |Omega| > 0 is the ellipse
    omega_1^2 + omega_2^2 - 2 c omega_1 omega_2 < 1 - c^2, cf. eq. (3.1),
    whose area is pi (1 - c^2) / sqrt(1 - c^2). Integrating over c in (-1, 1)
    returns V_3 = pi^2 / 2.
    """
    c = np.asarray(c, dtype=float)
    return np.pi * np.sqrt(np.clip(1.0 - c * c, 0.0, None))


def elliptope_boundary(w1, w2):
    """Upper and lower boundary of E_3: omega_3 = w1 w2 +/- sqrt((1-w1^2)(1-w2^2)), eq. (3.1)."""
    w1, w2 = np.asarray(w1, float), np.asarray(w2, float)
    r = np.sqrt(np.clip((1 - w1 ** 2) * (1 - w2 ** 2), 0, None))
    return w1 * w2 - r, w1 * w2 + r


def slab_projection_area(lb: float, ub: float, n_c: int = 401, n_theta: int = 721) -> float:
    """Area of the projection of the slab {Omega in E_3 : lb < omega_slice < ub} on the other two
    correlations.

    This is the exact maximal area the convex hull of points in that slab can
    reach (the elliptope is convex, so the projection is convex and equals the
    hull of the section ellipses). It is never smaller than the paper's benchmark
    pi sqrt(1 - lb^2), because the section ellipses rotate and stretch along the
    slab. Offered as ``benchmark="projection"`` in :func:`slice_coverage`.
    """
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    pts = []
    for c in np.linspace(lb, ub, n_c):
        s = np.sqrt(max(1 - c * c, 0.0))
        # ellipse u^2(1-c) + v^2(1+c) = 1 - c^2 in the rotated frame (axes along the diagonals)
        a_ = s / np.sqrt(1 - c) if c < 1 else 0.0      # = sqrt(1 + c)
        b_ = s / np.sqrt(1 + c) if c > -1 else 0.0     # = sqrt(1 - c)
        u, v = a_ * np.cos(th), b_ * np.sin(th)
        pts.append(np.c_[(u + v) / np.sqrt(2), (u - v) / np.sqrt(2)])
    return float(ConvexHull(np.vstack(pts)).volume)


# ------------------------------------------------------------------ designs
def equicorrelation(k: int, rho: float) -> np.ndarray:
    S = np.full((k, k), float(rho))
    np.fill_diagonal(S, 1.0)
    return S


def toeplitz_corr(k: int, rho: float) -> np.ndarray:
    i = np.arange(k)
    return float(rho) ** np.abs(i[:, None] - i[None, :])


def uniform_perturbation(k: int, gamma: float, rng=None, symmetric: bool = True) -> np.ndarray:
    """'Uniform' design of Abadir, Distaso and Zikes (2014): rho_ij = gamma U(0,1).

    ``symmetric=False`` reproduces the authors' Table 1 script, which draws
    rho_ij and rho_ji independently (the resulting S is not symmetric).
    """
    rng = np.random.default_rng(rng)
    S = np.eye(k)
    for i in range(k - 1):
        for j in range(i + 1, k):
            S[i, j] = gamma * rng.uniform()
            S[j, i] = S[i, j] if symmetric else gamma * rng.uniform()
    return S


# ------------------------------------------------------------------ simulation
def offdiag_lower(R: np.ndarray) -> np.ndarray:
    """Lower-triangular off-diagonal elements, column-major: (2,1), (3,1), (3,2), ...

    For k = 3 and Omega = [[1, w1, w2], [w1, 1, w3], [w2, w3, 1]] this returns
    (w1, w2, w3), the coordinates used in Section 3.
    """
    R = np.asarray(R)
    k = R.shape[-1]
    j, i = np.triu_indices(k, 1)        # i > j, ordered by column j
    order = np.lexsort((i, j))
    return R[..., i[order], j[order]]


def simulate_dcc(S: np.ndarray, alpha: float, beta: float, T: int,
                 innovations: str = "cdcc", rng=None, burn: int = 0) -> np.ndarray:
    """Simulate a DCC(1,1) correlation path; returns (T, k, k) correlation matrices.

    Q_t = (1 - alpha - beta) S + alpha u_{t-1} u_{t-1}' + beta Q_{t-1},
    R_t = dg(Q_t)^{-1/2} Q_t dg(Q_t)^{-1/2},   Q_1 = S.

    innovations
        ``"cdcc"`` (default): Aielli (2013, eq. 4) corrected DCC,
        u_t = dg(Q_t)^{1/2} e_t with e_t ~ N(0, R_t), i.e. u_t ~ N(0, Q_t) --
        the design the paper describes ("we simulate its dynamics as in Aielli").
        ``"engle"``: Engle (2002), u_t = e_t ~ N(0, R_t), eq. (2.17).
        ``"iid"``: u_t ~ N(0, I_k) independently of the state -- what the authors'
        ``simDcc.m`` actually does; use it to reproduce Table 1.
    """
    rng = np.random.default_rng(rng)
    S = np.asarray(S, float)
    k = S.shape[0]
    n = T + burn
    if innovations == "iid":
        eps = rng.standard_normal((n, k))
        outer = eps[:, :, None] * eps[:, None, :]
        drive = (1 - alpha - beta) * S[None] + alpha * outer
        drive[0] = S                                        # Q_1 = S
        Q = lfilter([1.0], [1.0, -beta], drive, axis=0)    # Q_t = beta Q_{t-1} + drive_t
        Q[0] = S
    elif innovations in ("cdcc", "engle"):
        Q = np.empty((n, k, k))
        Q[0] = S
        for t in range(1, n):
            q = Q[t - 1]
            if innovations == "cdcc":
                L = np.linalg.cholesky(0.5 * (q + q.T))
            else:
                d = 1 / np.sqrt(np.diag(q))
                L = np.linalg.cholesky(q * np.outer(d, d))
            u = L @ rng.standard_normal(k)
            Q[t] = (1 - alpha - beta) * S + alpha * np.outer(u, u) + beta * q
    else:
        raise ValueError("innovations must be 'cdcc', 'engle' or 'iid'")
    d = 1 / np.sqrt(np.einsum("tii->ti", Q))
    R = Q * d[:, :, None] * d[:, None, :]
    return R[burn:]


def slice_coverage(points: np.ndarray, n_slices: int = 8, slice_on: int = 2,
                   benchmark: str = "paper") -> pd.DataFrame:
    """Convex-hull coverage of E_3 slice by slice (Table 1 of the paper).

    Parameters
    ----------
    points : (n, 3) array of (w1, w2, w3), e.g. ``offdiag_lower(R)``.
    n_slices : number of equal slices of (0, 1) (the paper uses 8).
    slice_on : which coordinate defines the slices; 2 = omega_3 as in the paper's
        text (the authors' script slices on index 0 = omega_1; for the symmetric
        designs of Table 1 the two coincide in distribution).
    benchmark : ``"paper"`` uses pi sqrt(1 - lb^2) at the lower edge of each
        slice (the authors' code); ``"midpoint"`` uses the slice midpoint;
        ``"projection"`` uses :func:`slab_projection_area`, the exact maximum.

    Returns
    -------
    DataFrame with columns lb, ub, max_cover, hull_area, ratio and n_points.
    """
    points = np.asarray(points, float)
    other = [i for i in range(3) if i != slice_on]
    edges = np.linspace(0, 1, n_slices + 1)
    rows = []
    for lb, ub in zip(edges[:-1], edges[1:]):
        m = (points[:, slice_on] > lb) & (points[:, slice_on] < ub)
        P = points[m][:, other]
        area = ConvexHull(P).volume if m.sum() >= 3 else np.nan
        if benchmark == "paper":
            mx = float(ellipse_slice_area(lb))
        elif benchmark == "midpoint":
            mx = float(ellipse_slice_area(0.5 * (lb + ub)))
        elif benchmark == "projection":
            mx = slab_projection_area(lb, ub)
        else:
            raise ValueError("benchmark must be 'paper', 'midpoint' or 'projection'")
        rows.append((lb, ub, mx, area, area / mx, int(m.sum())))
    return pd.DataFrame(rows, columns=["lb", "ub", "max_cover", "hull_area", "ratio", "n_points"])


def coverage_table(T: int = 400_000, alpha: float = 0.3, apb: float = 0.998,
                   innovations: str = "cdcc", benchmark: str = "paper", seed: int = 123,
                   slice_on: int = 2, symmetric_uniform: bool = True) -> pd.DataFrame:
    """Reproduce Table 1: DCC hull areas for the equicorrelation, Toeplitz and Uniform designs."""
    rng = np.random.default_rng(seed)
    designs = ([("Equicorrelation", f"rho={r}", equicorrelation(3, r)) for r in (0.0, 0.3, 0.5, 0.7)]
               + [("Toeplitz", f"rho={r}", toeplitz_corr(3, r)) for r in (0.5, 0.75, 0.95)]
               + [("Uniform", f"gamma={g}", uniform_perturbation(3, g, rng, symmetric_uniform)) for g in (0.05, 0.10)])
    out = None
    for fam, lab, S in designs:
        R = simulate_dcc(S, alpha, apb - alpha, T, innovations, rng)
        cov = slice_coverage(offdiag_lower(R), 8, slice_on, benchmark)
        if out is None:
            out = pd.DataFrame({("", "Domain"): [f"({a:.3f}, {b:.3f})" for a, b in zip(cov.lb, cov.ub)],
                                ("", "Max cover"): cov.max_cover})
        out[(fam, lab)] = cov.hull_area.to_numpy()
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def dscc_span_draws(n_cuts: int = 41, bound: float = 3.0, lam=(0.1, 5.0, 40.0)) -> np.ndarray:
    """Correlations implied by e^Upsilon Lambda e^{-Upsilon} on a grid of Upsilon (k = 3).

    Mirrors the authors' ``MExpDraws.m``: a uniform grid of the three free
    elements in [-bound, bound], a fixed Lambda, and the resulting (w1, w2, w3).
    """
    g = np.linspace(-bound, bound, n_cuts)
    V = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    lam = np.asarray(lam, float)
    out = np.empty_like(V)
    for i, v in enumerate(V):
        # authors' grid: upsilon(2,1)=x1, (3,1)=x2, (3,2)=x3, i.e. upper elements (-x1, -x2, -x3)
        out[i] = offdiag_lower(correlation_from_cov(compose_sigma(lam, -v)))
    return out
