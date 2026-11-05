"""Explicit minimal representation of variance matrices.

Implements Abadir (2023, Econometrics Journal 26, 88-104), Lemma 2.1-2.2 and
Proposition 2.1, as used by Abadir and Rockinger (2026) for DSCC:

    Sigma = P^delta  e^Upsilon  Lambda  e^{-Upsilon}  P^delta          (2.1)/(2.3)

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg as sla

__all__ = [
    "n_offdiag",
    "skew_from_vec",
    "vec_from_skew",
    "permutation_matrix",
    "compose_sigma",
    "decompose_sigma",
    "correlation_from_cov",
    "orthogonal_from_data",
    "xi_from_data",
]


def n_offdiag(k: int) -> int:
    """Number of free skew-symmetric elements, l = k(k-1)/2."""
    return k * (k - 1) // 2


def _triu(k: int):
    # Ordering used throughout: row-major upper triangle (i < j), i.e. the
    # loop ``for i in 1..k-1: for j in i+1..k`` of the authors' code.
    return np.triu_indices(k, 1)


def skew_from_vec(v: np.ndarray, k: int) -> np.ndarray:
    """Build the k x k skew-symmetric matrix Upsilon from its l free elements.

    The free elements fill the upper triangle (i < j) and the lower triangle is
    their negative. The paper writes the free elements below the diagonal
    (upsilon_ij, i > j); the two conventions are the reparameterisation
    upsilon -> -upsilon and span the same space (Lemma 2.2). The upper-triangle
    convention is kept so that the authors' stored parameter vectors load as is.
    """
    v = np.asarray(v, dtype=float)
    U = np.zeros((k, k))
    i, j = _triu(k)
    U[i, j] = v
    U[j, i] = -v
    return U


def vec_from_skew(U: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew_from_vec` (upper-triangle elements, row-major)."""
    k = U.shape[0]
    return np.asarray(U)[_triu(k)].copy()


def permutation_matrix(k: int) -> np.ndarray:
    """P := diag( [[0,1],[1,0]], I_{k-2} )  (Proposition 2.1)."""
    P = np.eye(k)
    P[[0, 1]] = P[[1, 0]]
    return P


def compose_sigma(lam: np.ndarray, ups: np.ndarray, delta: int = 0) -> np.ndarray:
    """Sigma = P^delta e^Upsilon diag(lam) e^{-Upsilon} P^delta, eq. (2.1).

    Parameters
    ----------
    lam : (k,) positive eigenvalues.
    ups : (l,) free elements of Upsilon, or a (k, k) skew-symmetric matrix.
    delta : 0 or 1.
    """
    lam = np.asarray(lam, dtype=float)
    k = lam.size
    U = ups if np.ndim(ups) == 2 else skew_from_vec(ups, k)
    E = sla.expm(U)
    S = (E * lam) @ E.T
    if delta:
        S = S[[1, 0] + list(range(2, k))][:, [1, 0] + list(range(2, k))]
    return 0.5 * (S + S.T)


def decompose_sigma(Sigma: np.ndarray):
    """Recover (lam, upsilon, delta) from a positive-definite Sigma.

    The spectral decomposition Sigma = T Lambda T' is computed, delta is set to 1
    when det(T) = -1 (so that P T is proper orthogonal), and Upsilon is the
    principal matrix logarithm of the proper-orthogonal factor (Lemma 2.2(ii)).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    k = Sigma.shape[0]
    lam, T = np.linalg.eigh(Sigma)
    delta = 0
    if np.linalg.det(T) < 0:
        delta = 1
        T = permutation_matrix(k) @ T
    U = sla.logm(T)
    if np.iscomplexobj(U):
        U = U.real
    U = 0.5 * (U - U.T)
    return lam, vec_from_skew(U), delta


def correlation_from_cov(Sigma: np.ndarray) -> np.ndarray:
    """Omega = dg(Sigma)^{-1/2} Sigma dg(Sigma)^{-1/2}, eq. (2.14)."""
    d = 1.0 / np.sqrt(np.diag(Sigma))
    return Sigma * np.outer(d, d)


def _householder_complement(q1: np.ndarray) -> np.ndarray:
    """Proper-orthogonal Q = (q1, Q2) with first column q1 (deterministic).

    A Householder reflector H with H e1 = q1 is built (numerically stable sign
    choice); its columns are orthonormal and its first column is q1. As a
    reflector has determinant -1, the last column is negated so that |Q| = +1.
    """
    k = q1.size
    e1 = np.zeros(k)
    e1[0] = 1.0
    s = 1.0 if q1[0] >= 0 else -1.0
    u = q1 + s * e1          # reflects  -s*q1 ... handled by the sign fix below
    nu = u @ u
    if nu < 1e-300:
        Q = np.eye(k)
    else:
        H = np.eye(k) - 2.0 * np.outer(u, u) / nu   # H e1 = -s q1
        Q = H.copy()
        Q[:, 0] = -s * H[:, 0]                      # first column = q1
    if np.linalg.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]
    return Q


def orthogonal_from_data(x: np.ndarray, method: str = "householder") -> np.ndarray:
    """Proper-orthogonal eigenvector matrix Q_t of x_t x_t', eq. (2.10).

    Parameters
    ----------
    x : (k,) observation.
    method : {"householder", "eig"}
        ``"householder"`` (default): Q_t = (q1t, Q2t), q1t = x/||x||, with a
        deterministic Householder complement Q2t. Reproducible on any platform.
        ``"eig"``: the authors' replication convention -- eigenvectors of the
        rank-one matrix x x' in ascending-eigenvalue order (so q1t is the LAST
        column), first column negated if the determinant is -1. For k >= 3 the
        null-space basis returned by LAPACK is implementation-defined, so this
        does not reproduce MATLAB's numbers; pass ``xi=`` to the model for an
        exact replication.

    Notes
    -----
    For k = 2 the complement is unique and both methods agree with eq. (4.4) of
    Abadir (2023) up to the column order.
    """
    x = np.asarray(x, dtype=float)
    k = x.size
    nx = np.linalg.norm(x)
    if nx == 0.0:
        return np.eye(k)
    if method == "householder":
        return _householder_complement(x / nx)
    if method == "eig":
        _, Q = np.linalg.eigh(np.outer(x, x))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q
    raise ValueError(f"method must be 'householder' or 'eig', got {method!r}")


def xi_from_data(X: np.ndarray, method: str = "householder") -> np.ndarray:
    """Data-driven innovations xi_t = vec(log Q_t), eq. (2.11).

    The principal matrix logarithm is used, as stated after eq. (3.6) of Abadir
    (2023). Returns a (T, l) array, upper-triangle ordering.
    """
    X = np.asarray(X, dtype=float)
    T, k = X.shape
    iu = _triu(k)
    out = np.empty((T, n_offdiag(k)))
    n_complex = 0
    for t in range(T):
        L = sla.logm(orthogonal_from_data(X[t], method))
        if np.iscomplexobj(L):
            n_complex += int(np.abs(L.imag).max() > 1e-8)
            L = L.real
        out[t] = L[iu]
    if n_complex:
        warnings.warn(
            f"{n_complex} observation(s) gave an orthogonal Q_t with eigenvalue -1, "
            "where the principal logarithm is not unique; the real part was kept.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out
