"""Complete DSCC application on real data: the Fama-French five daily factors.

This is the five-factor block that Abadir and Rockinger (2026, Section 5) model with
DSCC. The script
  1. estimates DSCC-lambda_garch and DSCC-lambda_egarch on the first five-year
     window of the paper (2 Jan 1980 onwards, 1,260 days) and prints the tables;
  2. draws the conditional correlations, eigenvalues and the last correlation matrix;
  3. reproduces Section 3 (Figure 1, Table 1, Figure 2) on simulated DCC paths;
  4. runs a monthly-rebalanced minimum-variance backtest on the five factor
     portfolios with DSCC-egarch-av, DSCC-garch, DCC, the sample covariance and 1/N,
     and prints Tables 2-6 and Figure 4.

Run from a terminal:
    python ff5_dscc_example.py                 # downloads the data once
    python ff5_dscc_example.py --csv FF5Factors.CSV --months 60 --out results

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dscc
from dscc import elliptope as el


def main(csv=None, months=60, out="results", sim_T=400_000, n_starts=1, refit_every=12):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    # ------------------------------------------------------------ 1. data
    ff = dscc.datasets.load_ff5_daily(path=csv, start="1980-01-02", end="2022-03-31")
    factors = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    print(f"Fama-French 5 factors: {factors.index[0].date()} to {factors.index[-1].date()}, "
          f"T = {len(factors)} days (percent returns)\n")
    window = factors.iloc[:1260]

    # ------------------------------------------------------------ 2. estimation
    fits = {}
    for dyn in ("garch", "egarch"):
        t0 = time.time()
        res = dscc.DSCC(window, dynamics=dyn).fit(n_starts=n_starts)
        fits[dyn] = res
        print(res.summary())
        print(f"(estimated in {time.time() - t0:.0f} s)\n")
        (out / f"table_params_{dyn}.tex").write_text(res.to_latex(
            caption=f"DSCC-$\\lambda_{{{dyn}}}$ estimates, FF5 factors 1980--1984", label=f"tab:{dyn}"))
        (out / f"table_params_{dyn}.md").write_text(res.to_markdown())

    res = fits["egarch"]
    dscc.save_figure(res.plot_correlations(), str(out / "fig_correlations"))
    dscc.save_figure(res.plot_eigenvalues(), str(out / "fig_eigenvalues"))
    dscc.save_figure(dscc.plot_correlation_heatmap(res.correlation(-1), res.model.names),
                     str(out / "fig_corr_heatmap"))
    fc = res.forecast(21, init="average")
    print("21-day average forecast of the factor correlation matrix (DSCC-egarch-av):")
    print(pd.DataFrame(dscc.correlation_from_cov(fc.mean_sigma), index=res.model.names,
                       columns=res.model.names).round(3), "\n")

    # ------------------------------------------------------------ 3. span (Section 3)
    print(f"Elliptope volume V3 = {el.elliptope_volume(3):.6f} (pi^2/2 = {np.pi ** 2 / 2:.6f})")
    dscc.save_figure(dscc.plot_elliptope(), str(out / "fig1_elliptope"))
    for innov in ("iid", "cdcc"):
        tab = el.coverage_table(T=sim_T, innovations=innov, slice_on=0 if innov == "iid" else 2,
                                symmetric_uniform=(innov != "iid"))
        print(f"\nTable 1 -- DCC convex-hull surface per slice, innovations = '{innov}'")
        print(tab.round(2).to_string(index=False))
        tab.to_csv(out / f"table1_{innov}.csv", index=False)
    R = el.simulate_dcc(el.equicorrelation(3, 0.3), 0.3, 0.698, sim_T, "iid", rng=123)
    dscc.save_figure(dscc.plot_coverage_grid(el.offdiag_lower(R), slice_on=0), str(out / "fig2_coverage"))

    # ------------------------------------------------------------ 4. portfolio backtest
    print(f"\nRolling GMV backtest on the five factor portfolios, last {months} months ...")
    forecasters = [
        dscc.EqualWeight(),
        dscc.SampleCovForecaster(),
        dscc.DCCForecaster(),
        dscc.DSCCForecaster("garch", init="last", refit_every=refit_every),
        dscc.DSCCForecaster("egarch", init="average", refit_every=refit_every),
    ]
    bt = dscc.rolling_backtest(factors, forecasters, n_months=months, verbose=False)
    print("\nTable 2 -- out-of-sample performance (annualised, percent)")
    print(bt.performance_table().round(2))
    print("\nTable 3 -- one-sided HAC tests of monthly return differences")
    print(bt.pairwise().round(3).to_string(index=False))
    print("\nTable 4 -- turnover and leverage")
    print(bt.turnover_table().round(2))
    print("\nTables 5-6 -- covariance and correlation summaries")
    print(bt.covariance_table().round(3))
    print("\ncomputing time per model (s):", {k: round(v) for k, v in bt.timing.items()})
    dscc.save_figure(dscc.plot_wealth(bt), str(out / "fig4_wealth"))
    bt.daily.to_csv(out / "oos_daily_returns.csv")
    bt.performance_table().to_csv(out / "table2_performance.csv")
    bt.pairwise().to_csv(out / "table3_pairwise.csv", index=False)
    bt.turnover_table().to_csv(out / "table4_turnover.csv")
    bt.covariance_table().to_csv(out / "table56_covariance.csv")
    return fits, bt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None, help="local FF5 CSV (else download from the French library)")
    p.add_argument("--months", type=int, default=60, help="number of out-of-sample months")
    p.add_argument("--out", default="results", help="output folder for tables and figures")
    p.add_argument("--sim-T", type=int, default=400_000, help="length of the DCC simulations")
    p.add_argument("--n-starts", type=int, default=1, help="number of optimisation starts")
    a = p.parse_args()
    main(a.csv, a.months, a.out, a.sim_T, a.n_starts)
