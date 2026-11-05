"""Portfolio, DCC, colours, tables, data parser and backtest tests."""
import numpy as np
import pandas as pd
import pytest

import dscc
from dscc import portfolio as pf


def test_gmv_weights_minimise_variance():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)); S = A @ A.T + np.eye(4)
    w = pf.gmv_weights(S)
    assert np.isclose(w.sum(), 1)
    for _ in range(50):
        v = rng.standard_normal(4); v -= v.mean()          # feasible direction (sum zero)
        assert (w + 0.01 * v) @ S @ (w + 0.01 * v) >= w @ S @ w - 1e-12


def test_holding_returns():
    R = np.array([[0.01, -0.02], [0.03, 0.00]])
    w = np.array([0.5, 0.5])
    assert np.allclose(pf.holding_period_returns(w, R, "constant"), R @ w)
    bh = pf.holding_period_returns(w, R, "buy_and_hold")
    total = 0.5 * 1.01 * 1.03 + 0.5 * 0.98 * 1.0
    assert np.isclose(np.prod(1 + bh), total)


def test_performance_definitions():
    r = np.random.default_rng(1).normal(0.0004, 0.01, 1000)
    p = pf.performance(r)
    assert np.isclose(p["AV"], 100 * 252 * r.mean())
    assert np.isclose(p["SD"], 100 * np.sqrt(252) * r.std(ddof=1))
    assert np.isclose(p["IR"], p["AV"] / p["SD"])


def test_newey_west_lags_paper():
    assert pf.newey_west_lags(447) == 5        # "resulting in five lags being chosen" (footnote 4)


def test_hac_zero_lags_equals_iid_t():
    d = np.random.default_rng(2).normal(0.1, 1, 300)
    r = pf.hac_mean_test(d, lags=0)
    se_iid = d.std(ddof=0) / np.sqrt(d.size)
    assert np.isclose(r.se, se_iid)


def test_turnover_constant_prices_is_rebalancing_distance():
    W = np.array([[0.5, 0.5], [0.6, 0.4], [0.6, 0.4]])
    R = [np.zeros((3, 2))] * 3
    out = pf.turnover_leverage(W, R)
    assert np.isclose(out["TO"], (0.2 + 0.0) / 2)
    assert np.isclose(out["GL"], 1.0) and out["PL"] == 0


def test_covariance_summaries_scalar_matrix():
    S = 2.0 * np.eye(5)
    c = pf.covariance_summaries([S])
    assert np.isclose(c.Delta[0], 0) and np.isclose(c.Delta_prime[0], 0)
    assert np.isclose(c.d[0], 0) and np.isclose(c.kappa[0], 1)


def test_dcc_baseline_runs_and_is_pd():
    X = np.random.default_rng(3).standard_normal((300, 3))
    r = dscc.DCC(X).fit()
    f = r.forecast(4)
    assert f.shape == (4, 3, 3) and np.all(np.linalg.eigvalsh(f) > 0)
    a, b = r.ab
    assert a >= 0 and b >= 0 and a + b < 1


def test_parula_stops_and_colorscale():
    assert len(dscc.parula_colors(64)) == 64
    assert dscc.parula_colors(64)[0] == "#3e26a8"
    cs = dscc.resolve_colorscale("Parula", 8)
    assert cs[0][0] == 0.0 and cs[-1][0] == 1.0 and len(cs) == 8


def test_french_csv_parser():
    txt = ("This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 202401 CRSP database.\n"
           "\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n19630701,   -0.67,    0.01,   -0.35,    0.03,    0.11,   0.012\n"
           "19630702,    0.79,   -0.31,    0.24,   -0.08,   -0.25,   0.012\n\n Copyright 2024\n")
    df = dscc.datasets.read_french_csv(txt)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert df.shape == (2, 6) and df.iloc[0, 0] == -0.67


def test_tables_latex_markdown():
    X = np.random.default_rng(4).standard_normal((60, 2))
    r = dscc.DSCCResults(dscc.DSCC(X), dscc.default_start(X), se=False)
    assert "\\toprule" in r.to_latex() and "| Parameter |" in r.to_markdown()
    assert "Dynamic Spectral" in r.summary()


def test_small_backtest_end_to_end():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2000-01-03", periods=200)
    R = pd.DataFrame(rng.normal(0.02, 1.0, (200, 2)), index=idx, columns=["a", "b"])
    fc = [dscc.EqualWeight(), dscc.SampleCovForecaster(),
          dscc.DSCCForecaster("garch", init="last", refit_every=100, maxiter=5)]
    bt = dscc.rolling_backtest(R, fc, window=120, horizon=20, verbose=False)
    assert bt.daily.shape == (80, 3)
    assert set(bt.performance_table().columns) == {"AV", "SD", "IR"}
    assert bt.turnover_table().loc["1/N", "GL"] == pytest.approx(1.0)
    assert len(bt.pairwise()) == 1
