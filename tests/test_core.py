"""Core tests: representation, filter, bivariate closed form, elliptope analytics."""
import numpy as np
import pytest
from scipy import integrate, linalg as sla

import dscc
from dscc import elliptope as el
from dscc._filters import dsh_filter, expm_skew_batch


# ---------------------------------------------------------------- representation
@pytest.mark.parametrize("k", [2, 3, 5])
def test_compose_decompose_roundtrip(k):
    rng = np.random.default_rng(k)
    A = rng.standard_normal((k, k))
    S = A @ A.T + k * np.eye(k)
    lam, ups, d = dscc.decompose_sigma(S)
    assert np.allclose(dscc.compose_sigma(lam, ups, d), S, atol=1e-10)


@pytest.mark.parametrize("method", ["householder", "eig"])
@pytest.mark.parametrize("k", [2, 3, 5])
def test_orthogonal_from_data_is_proper_and_spans_x(method, k):
    x = np.random.default_rng(7).standard_normal(k)
    Q = dscc.orthogonal_from_data(x, method)
    assert np.allclose(Q.T @ Q, np.eye(k), atol=1e-12)
    assert np.isclose(np.linalg.det(Q), 1.0)
    # x x' = Q diag(x'x, 0) Q'   eq. (2.10)
    D = np.zeros((k, k))
    col = 0 if method == "householder" else k - 1
    D[col, col] = x @ x
    assert np.allclose(Q @ D @ Q.T, np.outer(x, x), atol=1e-12)


def test_xi_bivariate_is_principal_angle():
    """k = 2: principal log gives atan2; Abadir (2023, eq. 4.5) arcsin only agrees when x1 > 0."""
    X = np.random.default_rng(1).standard_normal((50, 2))
    xi = dscc.xi_from_data(X)
    assert np.allclose(xi[:, 0], -np.arctan2(X[:, 1], X[:, 0]))      # upper-triangle convention
    pos = X[:, 0] > 0
    assert np.allclose(xi[pos, 0], -np.arcsin(X[pos, 1] / np.hypot(X[pos, 0], X[pos, 1])))


def test_expm_skew_batch_equals_scipy():
    v = np.random.default_rng(3).standard_normal((40, 10))
    E = expm_skew_batch(v, 5)
    for t in range(40):
        assert np.allclose(E[t], sla.expm(dscc.skew_from_vec(v[t], 5)), atol=1e-12)


# ---------------------------------------------------------------- filter
def _bivariate_closed_form(theta, X, xi):
    """Direct implementation of Abadir (2023) eqs (4.2), (4.3), (4.7)."""
    D0, D1, D2 = theta[0:2], theta[2:4], theta[4:6]
    mu, al, be = theta[6:9]
    lam = np.zeros(2); v = vp = 0.0; d = 0; nll = 0.0
    for t in range(1, X.shape[0]):
        v = mu * (1 - al) + al * v + be * xi[t - 1, 0]
        xp = X[t - 1][::-1] if d else X[t - 1]
        E_prev = np.array([[np.cos(vp), np.sin(vp)], [-np.sin(vp), np.cos(vp)]])
        z = E_prev.T @ xp
        lam = D0 + D1 * lam + D2 * z ** 2
        c, s = np.cos(-v), np.sin(-v)
        l1, l2 = 1 / lam
        Si = np.array([[l1 * c * c + l2 * s * s, (l1 - l2) * s * c],
                       [(l1 - l2) * s * c, l1 * s * s + l2 * c * c]])
        f0 = X[t] @ Si @ X[t]; xs = X[t][::-1]; f1 = xs @ Si @ xs
        d = int(f1 < f0)
        nll += np.log(lam).sum() + (f1 if d else f0)
        vp = v
    return 0.5 * nll


def test_general_filter_matches_bivariate_closed_form():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 2))
    th = np.r_[0.05, 0.03, 0.85, 0.9, 0.1, 0.07, 0.4, 0.3, 0.2]
    xi = dscc.xi_from_data(X)
    assert np.isclose(dsh_filter(th, X, xi).negloglik, _bivariate_closed_form(th, X, xi), rtol=0, atol=1e-9)


def test_filter_sigma_reconstruction_and_likelihood():
    """Each summand equals log|Sigma_t| + x' Sigma_t^{-1} x with the reported delta_t (2.13)."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((80, 3))
    th = dscc.default_start(X)
    m = dscc.DSCC(X, demean=False)
    r = dscc.DSCCResults(m, th, se=False)
    S = r.sigma()
    for t in (5, 40, 79):
        ref = np.linalg.slogdet(S[t])[1] + X[t] @ np.linalg.solve(S[t], X[t])
        assert np.isclose(r.fout.contributions[t], ref, atol=1e-9)
        assert np.allclose(np.diag(r.correlation(t)), 1.0)


@pytest.mark.parametrize("dyn", ["garch", "egarch"])
def test_numba_and_numpy_paths_agree(dyn):
    X = np.random.default_rng(8).standard_normal((300, 4))
    th = dscc.default_start(X, dyn)
    xi = dscc.xi_from_data(X)
    a = dsh_filter(th, X, xi, dyn, use_fast=True)
    b = dsh_filter(th, X, xi, dyn, use_fast=False)
    assert np.isclose(a.negloglik, b.negloglik, rtol=0, atol=1e-8)
    assert np.array_equal(a.delta, b.delta)


def test_delta_fixed_reproduces_free_filter():
    X = np.random.default_rng(2).standard_normal((60, 3))
    th = dscc.default_start(X, "egarch")
    xi = dscc.xi_from_data(X)
    a = dsh_filter(th, X, xi, "egarch")
    b = dsh_filter(th, X, xi, "egarch", delta_fixed=a.delta)
    assert np.isclose(a.negloglik, b.negloglik)


def test_std_resid_whitens():
    X = np.random.default_rng(4).standard_normal((50, 3))
    m = dscc.DSCC(X, demean=False)
    r = dscc.DSCCResults(m, dscc.default_start(X), se=False)
    e = r.std_resid()
    S = r.sigma()
    for t in (3, 30):
        L = sla.sqrtm(S[t]).real
        assert np.allclose(L @ e[t], X[t], atol=1e-9)


def test_input_validation():
    with pytest.raises(ValueError):
        dscc.DSCC(np.ones((5, 2)))
    bad = np.random.default_rng(0).standard_normal((100, 2)); bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        dscc.DSCC(bad)
    with pytest.raises(ValueError):
        dscc.DSCC(np.random.default_rng(0).standard_normal((100, 2)), dynamics="bekk")


def test_dataframe_equals_ndarray():
    import pandas as pd
    X = np.random.default_rng(9).standard_normal((70, 2))
    th = dscc.default_start(X)
    a = dscc.DSCC(X).loglike(th)
    b = dscc.DSCC(pd.DataFrame(X, columns=["a", "b"])).loglike(th)
    assert a == b


def test_forecast_shapes_and_psd():
    X = np.random.default_rng(11).standard_normal((120, 3))
    r = dscc.DSCCResults(dscc.DSCC(X), dscc.default_start(X), se=False)
    for mode in ("paper", "replication"):
        for init in ("last", "average"):
            f = r.forecast(H=5, mode=mode, init=init)
            assert f.sigma.shape == (5, 3, 3)
            assert np.all(np.linalg.eigvalsh(f.sigma) > 0)


def test_fit_never_worse_than_start():
    X = np.random.default_rng(21).standard_normal((150, 3))
    m = dscc.DSCC(X)
    s = dscc.default_start(m.X)
    r = m.fit(start=s, maxiter=15, se=False, polish_maxiter=2)
    assert r.loglik >= m.loglike(s) - 1e-9


# ---------------------------------------------------------------- elliptope
def test_elliptope_volumes():
    assert np.isclose(el.elliptope_volume(2), 2.0)
    assert np.isclose(el.elliptope_volume(3), np.pi ** 2 / 2)          # eq. (3.2)
    val = integrate.quad(lambda c: el.ellipse_slice_area(c), -1, 1)[0]
    assert np.isclose(val, np.pi ** 2 / 2)


def test_table1_max_cover_column():
    c = el.slice_coverage(np.random.default_rng(0).uniform(0.0, 0.99, (5000, 3)))
    paper = [3.14, 3.12, 3.04, 2.91, 2.72, 2.45, 2.08, 1.52]
    assert np.allclose(c.max_cover.round(2), paper)


def test_projection_benchmark_dominates_paper_benchmark():
    for lb in (0.0, 0.5, 0.875):
        assert el.slab_projection_area(lb, lb + 0.125) >= el.ellipse_slice_area(lb) - 1e-9


def test_table1_dcc_column_rho03_iid():
    """Authors' simulator (iid innovations), alpha=0.3, alpha+beta=0.998: Table 1, rho = 0.3."""
    R = el.simulate_dcc(el.equicorrelation(3, 0.3), 0.3, 0.698, 400_000, "iid", rng=123)
    c = el.slice_coverage(el.offdiag_lower(R), 8, slice_on=0)
    paper = np.array([2.77, 2.73, 2.64, 2.47, 2.29, 1.95, 1.55, 0.85])
    assert np.all(np.abs(c.hull_area.to_numpy()[:7] - paper[:7]) < 0.06)   # Monte Carlo error
    assert np.all(c.hull_area < c.max_cover)


def test_simulated_dcc_are_correlation_matrices():
    R = el.simulate_dcc(el.toeplitz_corr(3, 0.75), 0.05, 0.9, 500, "cdcc", rng=1)
    assert np.allclose(np.einsum("tii->ti", R), 1.0)
    assert np.all(np.linalg.eigvalsh(R) > 0)
