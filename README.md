# dscc

[![PyPI](https://img.shields.io/pypi/v/dscc.svg)](https://pypi.org/project/dscc/)
[![Python](https://img.shields.io/pypi/pyversions/dscc.svg)](https://pypi.org/project/dscc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://merwanroudane.github.io/dscc/)

**dscc** implements the *Dynamic Spectral Conditional Correlation* (DSCC) model of
Abadir and Rockinger (2026, *Econometrics Journal*,
[doi:10.1093/ectj/utag017](https://doi.org/10.1093/ectj/utag017)), built on the explicit
minimal representation of variance matrices of Abadir (2023, *Econometrics Journal* 26,
88–104, [doi:10.1093/ectj/utac023](https://doi.org/10.1093/ectj/utac023)). You get
estimation, multistep forecasts, the elliptope-coverage benchmark, a DCC baseline and a
complete minimum-variance portfolio evaluation, with tables and figures ready for a
journal.

📖 **Documentation:** https://merwanroudane.github.io/dscc/
📦 **PyPI:** https://pypi.org/project/dscc/
🧑‍💻 **Author:** Dr Merwan Roudane · merwanroudane920@gmail.com · https://github.com/merwanroudane

---

## Table of contents
1. [What this package does](#1-what-this-package-does)
2. [Start from absolutely nothing](#2-start-from-absolutely-nothing)
3. [Installation](#3-installation)
4. [Your first analysis in 60 seconds](#4-your-first-analysis-in-60-seconds)
5. [Complete example with real data](#5-complete-example-with-real-data)
6. [Full syntax reference](#6-full-syntax-reference)
7. [Interpreting the output](#7-interpreting-the-output)
8. [Using your own Excel or CSV file](#8-using-your-own-excel-or-csv-file)
9. [Exporting tables and figures](#9-exporting-tables-and-figures)
10. [The method](#10-the-method)
11. [Fidelity to the paper and to the authors' code](#11-fidelity-to-the-paper-and-to-the-authors-code)
12. [Practical guidance and pitfalls](#12-practical-guidance-and-pitfalls)
13. [Limitations](#13-limitations)
14. [References](#14-references)
15. [Citation](#15-citation)
16. [Author and licence](#16-author-and-licence)

---

## 1. What this package does

Correlation matrices must stay positive definite, and most multivariate volatility models
pay for that by restricting the matrices they can reach. DSCC writes every conditional
variance matrix as

$$\Sigma_t = P^{\delta_t} e^{\Upsilon_t}\,\Lambda_t\,e^{-\Upsilon_t} P^{\delta_t},$$

with an unrestricted skew-symmetric $\Upsilon_t$, positive eigenvalues $\Lambda_t$ and a
binary $\delta_t$. Every parameter is free (up to $k$ simple sign conditions), the
representation is minimal, and it spans the whole space of correlation matrices.

You give the package a $(T\times k)$ table of returns. It gives you back:

- **Estimation** of DSCC-λ<sub>garch</sub> (eq. 2.5) and DSCC-λ<sub>egarch</sub> (eqs 2.6–2.7) by maximum likelihood (2.13), with in-sample $\hat\delta_t$, multi-start optimisation and standard errors.
- **Conditional variances and correlations** $\Sigma_t$, $\Omega_t$ (2.14), standardised residuals, eigenvalue paths.
- **Multistep forecasts** (Section 4), including the "-av" variant, in the paper's form or exactly as in the authors' MATLAB code.
- **The span benchmark** of Section 3: exact volume of the elliptope $E_k$, slice-by-slice convex-hull coverage, DCC simulation, Table 1 and Figures 1–3.
- **A DCC(1,1) baseline** (Engle 2002).
- **Portfolio evaluation** of Section 5: global minimum-variance weights, AV/SD/IR, turnover and leverage, HAC pairwise tests, the v/Δ/d/Δ′/κ diagnostics, and a rolling monthly backtest (Tables 2–6, Figures 4–5).
- **Journal-quality output**: aligned text tables with stars, LaTeX (booktabs) and Markdown export, figures in vector PDF and 300-dpi PNG, MATLAB's Parula as the default colour scale.

## 2. Start from absolutely nothing

*If you have never used Python, do these four steps once. They take about fifteen minutes.*

### Step 1 — install Python
Download **Anaconda** from https://www.anaconda.com/download. It is the recommended option for researchers: one installer gives you Python, Jupyter and Spyder. Alternatively, install plain Python from https://www.python.org/downloads/; on the first screen of that installer, **tick "Add Python to PATH"**.

### Step 2 — open a terminal
- **Windows:** press `Win`, type `Anaconda Prompt` (or `cmd`), press Enter.
- **macOS:** press `Cmd+Space`, type `Terminal`, press Enter.

Check that Python works by typing this and pressing Enter:
```bash
python --version
```
You should see something like `Python 3.11.5`.

### Step 3 — install this package
```bash
pip install "dscc[fast]"
```
(`[fast]` also installs `numba`, which makes the estimation about four times faster. Plain `pip install dscc` works too.)

### Step 4 — run your first command
Type `python`, press Enter, then paste:
```python
import dscc
print(dscc.__version__)
```
Type `exit()` to leave.

### Where do I write my code?
| Tool | Best for | How to open |
|---|---|---|
| **Jupyter Notebook** | step-by-step work with the figures shown inline | type `jupyter notebook` in the terminal |
| **Spyder** | a window like MATLAB or RStudio | type `spyder`, or open it from Anaconda Navigator |
| **A `.py` file** | a reproducible script | save `analysis.py`, then run `python analysis.py` |

## 3. Installation

```bash
pip install dscc                  # core: numpy, scipy, pandas, matplotlib
pip install "dscc[fast]"          # + numba (about 4x faster likelihood)
pip install "dscc[interactive]"   # + plotly
```
Development version:
```bash
pip install git+https://github.com/merwanroudane/dscc.git
```
Requires Python ≥ 3.9.

## 4. Your first analysis in 60 seconds

Two Fama–French daily factors, the last 1,000 days of the paper's sample (the data are downloaded once from the Kenneth R. French Data Library, then cached):

```python
import dscc

ff = dscc.datasets.load_ff5_daily(end="2022-03-31")
data = ff[["Mkt-RF", "HML"]].iloc[-1000:]

res = dscc.DSCC(data, dynamics="egarch").fit(n_starts=3, seed=1)
print(res.summary())

fc = res.forecast(H=21, init="average")          # one trading month ahead
print(dscc.correlation_from_cov(fc.mean_sigma)[0, 1])
print(dscc.gmv_weights(fc.mean_sigma))
```

Output:

```text
==============================================================================
           Dynamic Spectral Conditional Correlation -- ML estimates           
==============================================================================
Model:            DSCC-egarch (2.6)     Log-likelihood:              -2508.149
Variables (k):    2                     AIC:                          5042.297
Observations (T): 1000                  BIC:                          5106.085
Q2t complement:   householder           Converged:                         yes
------------------------------------------------------------------------------
                                coef   std err        t    P>|t|  [0.025  0.975]
------------------------------------------------------------------------------
Eigenvalue dynamics, Lambda_t
  D0[Mkt-RF]                 -0.2204    0.0562   -3.921    0.000  -0.331  -0.110 ***
  D0[HML]                    -0.1563    0.0201   -7.793    0.000  -0.196  -0.117 ***
  D11[Mkt-RF]                 0.9609    0.0179   53.774    0.000   0.926   0.996 ***
  D11[HML]                    0.9786    0.0068  143.392    0.000   0.965   0.992 ***
  D21[Mkt-RF]                 0.2348    0.0512    4.586    0.000   0.134   0.335 ***
  D21[HML]                    0.1853    0.0237    7.813    0.000   0.139   0.232 ***
Skew-symmetric dynamics, Upsilon_t (2.9)
  mu[Mkt-RF,HML]              1.6690    0.0134  124.989    0.000   1.643   1.695 ***
  alpha[Mkt-RF,HML]           0.9293    0.0135   68.658    0.000   0.903   0.956 ***
  beta[Mkt-RF,HML]           -0.0201    0.0027   -7.441    0.000  -0.025  -0.015 ***
Asymmetry function g (2.7)
  a[Mkt-RF]                  -0.2265    0.0959   -2.362    0.018  -0.414  -0.039 **
  a[HML]                      0.2479    0.1172    2.114    0.034   0.018   0.478 **
  c[Mkt-RF]                   0.0562    0.0321    1.751    0.080  -0.007   0.119 *
  c[HML]                      0.2140    0.1275    1.678    0.093  -0.036   0.464 *
------------------------------------------------------------------------------
Permutation component delta_t (Proposition 2.1, eq. 2.13)
  share of t with delta_t = 1:   0.5095      psi = Pr(delta = 0):   0.4905
==============================================================================
Notes: *** p<0.01, ** p<0.05, * p<0.10. Standard errors from the numerical Hessian
of the profile log-likelihood with delta_t held at its MLE; '--' = not available
(parameter on a bound, or non-positive local curvature). Gaussian innovations.

Forecast correlation, average over the next 21 days: -0.471
Minimum-variance weights: [0.508 0.492]
```

## 5. Complete example with real data

`examples/ff5_dscc_example.py` runs the whole paper on public data: the Fama–French five daily factors (Mkt-RF, SMB, HML, RMW, CMA), 2 January 1980 to 31 March 2022, in percent. These are exactly the five factors whose covariance matrix Abadir and Rockinger model with DSCC in Section 5.

```bash
python examples/ff5_dscc_example.py --months 60 --out results
```

The script follows the paper section by section.

**Step 1 — estimate on the first five-year window (1,260 days), as in the paper:**
```python
ff = dscc.datasets.load_ff5_daily(start="1980-01-02", end="2022-03-31")
factors = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
window = factors.iloc[:1260]
res = dscc.DSCC(window, dynamics="egarch").fit()
print(res.summary())
```

**Step 2 — correlations, eigenvalues and forecasts:**
```python
res.plot_correlations(); res.plot_eigenvalues()
fc = res.forecast(21, init="average")             # DSCC-lambda_egarch-av
```

**Step 3 — the span benchmark (Section 3, Table 1):**
```python
from dscc import elliptope as el
el.elliptope_volume(3)                            # pi^2/2
tab = el.coverage_table(T=400_000, innovations="iid", slice_on=0)
```

**Step 4 — the monthly minimum-variance backtest (Section 5):**
```python
fcs = [dscc.EqualWeight(), dscc.SampleCovForecaster(), dscc.DCCForecaster(),
       dscc.DSCCForecaster("garch", init="last"),
       dscc.DSCCForecaster("egarch", init="average")]
bt = dscc.rolling_backtest(factors, fcs, n_months=60)
bt.performance_table()      # Table 2
bt.pairwise()               # Table 3
bt.turnover_table()         # Table 4
bt.covariance_table()       # Tables 5-6
dscc.plot_wealth(bt)        # Figure 4
```

The script prints every table and writes them as `.tex`, `.md` and `.csv` files, together with
the figures (PDF and 300-dpi PNG), into the folder given by `--out`.

## 6. Full syntax reference

### `DSCC(data, dynamics="garch", demean=True, complement="householder", xi=None)`
The model.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `data` | DataFrame or (T, k) array | — | returns; column names are carried into every table |
| `dynamics` | `"garch"`, `"egarch"` | `"garch"` | eigenvalue dynamics: eq. (2.5) or eqs (2.6)–(2.7) |
| `demean` | bool | `True` | subtract the sample mean (Abadir 2023, §4.2) |
| `complement` | `"householder"`, `"eig"` | `"householder"` | how the orthogonal complement $Q_{2t}$ in (2.10) is built; irrelevant for k = 2 |
| `xi` | (T, l) array or None | `None` | precomputed $\xi_t$ of (2.11); pass the authors' MATLAB series to reproduce their likelihood exactly |

Methods: `.fit(...)` → `DSCCResults`; `.loglike(theta)` → float (eq. 2.13); `.filter(theta)` → paths.

### `DSCC.fit(start=None, n_starts=1, seed=0, method="SLSQP", maxiter=500, se=True, polish=True, polish_maxiter=20, verbose=False)`

| Argument | Default | Meaning |
|---|---|---|
| `start` | eq. (5.1) of Abadir (2023) | starting parameter vector |
| `n_starts` | `1` | number of starts; the extra ones are random draws in the admissible region (the paper's §5.5 uses 10) |
| `seed` | `0` | seed of the random starts |
| `method` | `"SLSQP"` | SciPy optimiser; SLSQP enforces $d_{1i}+d_{2i}<1$ |
| `se` | `True` | standard errors from the profile likelihood with $\delta_t$ fixed at its MLE |
| `polish` | `True` | derivative-free Powell polish of the best point found |

Parameter order (k variables, l = k(k−1)/2): `D0 (k), D11 (k), D21 (k), mu (l), alpha (l), beta (l)`, plus `a (k), c (k)` for EGARCH. This is the same order as the authors' code.

### `DSCCResults`
| Attribute / method | Returns | Meaning |
|---|---|---|
| `.params`, `.bse`, `.tvalues`, `.pvalues` | Series | estimates and Wald inference |
| `.loglik`, `.aic`, `.bic` | float | fit statistics |
| `.delta`, `.psi` | array, float | in-sample $\hat\delta_t$; $\hat\psi=\Pr(\delta=0)$ |
| `.lam`, `.ups` | arrays | $\Lambda_t$ and the free elements of $\Upsilon_t$ |
| `.sigma(t=None)`, `.correlation(t=None)` | (k,k) or (T,k,k) | $\Sigma_t$ (2.12), $\Omega_t$ (2.14) |
| `.std_resid()` | (T,k) | $\varepsilon_t=\Sigma_t^{-1/2}x_t$ |
| `.forecast(H=21, mode="paper", init="last")` | `DSCCForecast` | Section 4 forecasts; `.sigma`, `.correlation`, `.mean_sigma` |
| `.summary()`, `.to_frame()`, `.to_latex()`, `.to_markdown()` | str / DataFrame | tables |
| `.plot_correlations()`, `.plot_eigenvalues()` | Figure | figures |

`forecast` options: `mode="paper"` iterates $\Upsilon$ with (2.9) and weights the two δ-branches by $\hat\psi$ at every horizon. `mode="replication"` reproduces the authors' scripts. `init="average"` is the "-av" variant.

### `dscc.elliptope`
| Function | Meaning |
|---|---|
| `elliptope_volume(k)` | exact volume of $E_k$ ($\pi^2/2$ for k = 3, eq. 3.2) |
| `ellipse_slice_area(c)` | $\pi\sqrt{1-c^2}$, the section of $E_3$ at a fixed correlation |
| `slab_projection_area(lb, ub)` | exact maximal area of a slab (a sharper benchmark than the paper's) |
| `simulate_dcc(S, alpha, beta, T, innovations="cdcc")` | DCC paths; `"iid"` is the authors' simulator, `"engle"` Engle (2002) |
| `slice_coverage(points, n_slices=8, slice_on=2, benchmark="paper")` | Table 1 for one design |
| `coverage_table(...)` | the full Table 1 |
| `equicorrelation`, `toeplitz_corr`, `uniform_perturbation` | the designs of Table 1 |

### `dscc.portfolio`, `dscc.backtest`, `dscc.DCC`
| Function | Meaning |
|---|---|
| `gmv_weights(Sigma)` | $w=\Sigma^{-1}\iota/\iota'\Sigma^{-1}\iota$ |
| `performance(daily)` | AV, SD (annualised, percent) and IR |
| `turnover_leverage(W, R_list)` | TO, GL, PL |
| `hac_mean_test(d)`, `pairwise_tests(monthly)` | one-sided Newey–West tests (Table 3) |
| `covariance_summaries(Sigmas)` | $v_h$, $\Delta_h$ (5.2), $d_h$, $\Delta'_h$ (5.3), $\kappa_h$ |
| `rolling_backtest(returns, forecasters, window=1260, horizon=21, holding="buy_and_hold")` | monthly GMV backtest |
| `DSCCForecaster`, `DCCForecaster`, `SampleCovForecaster`, `EqualWeight` | the competing models |
| `DCC(data).fit()` | Engle (2002) DCC(1,1); `.forecast(H)` |

### Data and colours
`datasets.load_ff5_daily(path=None, start=None, end=None)` and `datasets.load_industry_daily(n=10)` download and cache the French library files (they are not redistributed with the package). `parula_colors(n)`, `matlab_jet_colors`, `turbo_colors`, `bluered_colors`, `sinha_colors`, `resolve_colorscale("Parula")` and `get_cmap("Parula")` handle colours.

## 7. Interpreting the output

**The summary table.**
- *D0, D11, D21* are the GARCH (or EGARCH) coefficients of each eigenvalue $\lambda_{it}$. *D11* is the persistence and *D21* the reaction to the newest rotated return $z_{t-1}$.
- *mu, alpha, beta* drive each free element $\upsilon_{ijt}$ of the rotation:
  - *mu* is its long-run level, in radians.
  - *alpha* is its persistence.
  - *beta* is the loading on the data innovation $\xi_{ij,t-1}$.

  A large |alpha| means the eigenvectors, and so the correlation structure, move slowly.
- *a, c* (EGARCH only) make the response asymmetric: $g_i(v)=|v_i-c_i|-a_i(v_i-c_i)$.
- The δ block reports the share of days on which the permuted branch $P\,e^{\Upsilon}$ fits better, and $\hat\psi=\Pr(\delta=0)$, which weights the two branches in forecasts.
- Stars use two-sided Wald p-values. "--" means the standard error is not available: the parameter sits on a bound, or the local Hessian is not positive there, which happens because $\delta_t$ makes the likelihood piecewise smooth.

**Table 1.** Each row is a slice of one correlation. "Max cover" is the largest area positive definiteness allows in that slice, and the other columns are the areas DCC actually reaches. DSCC reaches the maximum by construction.

**Tables 2–6.**
- AV, SD and IR are the annualised mean return, volatility and information ratio of daily out-of-sample returns.
- TO is turnover, GL gross leverage, and PL the share of short positions.
- In Table 3, a small p-value means Model 1 earns significantly more than Model 2.
- $\Delta'$ close to 0 means the forecast correlations are close to zero. A large κ means an ill-conditioned matrix.

## 8. Using your own Excel or CSV file

```python
import pandas as pd
import dscc

df = pd.read_excel("C:/Users/you/Documents/returns.xlsx", index_col=0, parse_dates=True)
# or: df = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
df = df[["Asset1", "Asset2", "Asset3"]].dropna()

res = dscc.DSCC(df, dynamics="egarch").fit(n_starts=5)
print(res.summary())
```
On Windows, write paths with forward slashes (`C:/...`) or doubled backslashes (`C:\\...`). Use returns, not prices, and no missing values. Percent returns (as in the French library) keep the parameters on the scale used in the paper.

## 9. Exporting tables and figures

```python
print(res.to_latex())                       # paste into the manuscript
res.to_frame().to_excel("dscc_estimates.xlsx")
fig = res.plot_correlations()
dscc.save_figure(fig, "figure_correlations")   # writes .pdf (vector) and .png (300 dpi)
```

## 10. The method

For $x_t$ with conditional variance $\Sigma_t$:

1. **Representation** ([A23] Prop. 2.1; [AR26] 2.1): $\Sigma_t=P^{\delta_t}e^{\Upsilon_t}\Lambda_te^{-\Upsilon_t}P^{\delta_t}$, where $P$ swaps the first two variates.
2. **Eigenvalues** (2.4)–(2.5): with $z_t=e^{-\Upsilon_t}P^{\delta_t}x_t$, $\Lambda_t=D_0+D_{11}\Lambda_{t-1}+D_{21}\,\mathrm{dg}(z_{t-1}z_{t-1}')$; or the EGARCH (2.6) with $g$ of (2.7).
3. **Data innovations** (2.10)–(2.11): $x_tx_t'=Q_t\,\mathrm{diag}(x_t'x_t,0)Q_t'$, $|Q_t|=+1$, $\Xi_t=\log Q_t$ (principal value).
4. **Rotation dynamics** (2.9): $\upsilon_{ijt}=(1-\alpha_{ij})\mu_{ij}+\alpha_{ij}\upsilon_{ij,t-1}+\beta_{ij}\xi_{ij,t-1}$.
5. **Likelihood** (2.13): $\sum_t[\log f_\varepsilon(\Sigma_t^{-1/2}x_t)-\tfrac12\sum_i\log\lambda_{it}]$, with $\delta_t\in\{0,1\}$ chosen to maximise each summand.
6. **Forecasts** (Section 4, eq. 4.1): the two δ-branches are iterated forward and weighted by $\hat\psi$.

The full step → equation map is in [docs/methods.md](docs/methods.md).

## 11. Fidelity to the paper and to the authors' code

Every line was checked against the paper and against the authors' replication code (Zenodo, [doi:10.5281/zenodo.20817596](https://doi.org/10.5281/zenodo.20817596)). Where they disagree, the package follows the **paper** by default and offers an option that reproduces the **code**. The main points:

| Point | Paper | Authors' code | `dscc` |
|---|---|---|---|
| Complement $Q_{2t}$, k ≥ 3 | not specified | LAPACK's arbitrary null-space basis | deterministic Householder complement; `xi=` reproduces MATLAB |
| $\xi_t$, k = 2 | arcsin, eq. (4.5) of [A23] | principal log | principal log = atan2 (arcsin fails when $x_1<0$) |
| $\Upsilon$ over the forecast horizon | iterated with (2.9) | frozen at $\Upsilon_T$ | iterated; `mode="replication"` |
| DCC simulation, Table 1 | Aielli (2013) cDCC | iid innovations | cDCC; `innovations="iid"` reproduces Table 1 |
| Holding period | buy-and-hold | weights reset daily | buy-and-hold; `holding="constant"` |

Further checks:
- The general-k filter equals the bivariate closed form of [A23] to $10^{-13}$.
- $V_3=\pi^2/2$ is exact.
- The DCC column of Table 1 (ρ = 0.3) is reproduced to the second decimal.

The complete list, with measured effects, is in [docs/methods.md](docs/methods.md).

## 12. Practical guidance and pitfalls

- **Use several starts.** Because $\delta_t$ switches, the likelihood has many local optima. `n_starts=10` is the paper's choice. On the paper's first window, a single start stopped 10 points below the best value found.
- **Scale.** Use percent returns. The starting values (5.1) of [A23] are calibrated for them.
- **Dimension.** The number of parameters grows as $3k(k+1)/2$ (GARCH): k = 5 gives 45, k = 10 gives 165. A 1,260 × 5 fit takes one to three minutes with numba.
- **k ≥ 3.** Estimates depend on the convention for $Q_{2t}$. Keep the default for reproducible results across machines.
- **Standard errors** treat $\delta_t$ as known. Read them as approximate.

## 13. Limitations

- Only Gaussian innovations are implemented, as in the paper's applications.
- The NL-DCC (nonlinear-shrinkage DCC) block of De Nard, Ledoit and Wolf (2021), used for the 1,000-stock residuals, is not included. Neither is the CRSP stock data (restricted).
- The modified-likelihood estimation of the correlation-only parameters mentioned in Section 2.1 is left for future work, as in the paper.

## 14. References

- Abadir, K. M. and M. Rockinger (2026). Dynamic spectral conditional correlations. *Econometrics Journal*. https://doi.org/10.1093/ectj/utag017
- Abadir, K. M. (2023). Explicit minimal representation of variance matrices, and its implication for dynamic volatility models. *Econometrics Journal* 26, 88–104. https://doi.org/10.1093/ectj/utac023
- Abadir, K. M. and M. Rockinger (2026). Replication package. Zenodo. https://doi.org/10.5281/zenodo.20817596
- Aielli, G. P. (2013). Dynamic conditional correlation: on properties and estimation. *Journal of Business and Economic Statistics* 31, 282–299.
- De Nard, G., O. Ledoit and M. Wolf (2021). Factor models for portfolio selection in large dimensions: the good, the better and the ugly. *Journal of Financial Econometrics* 19, 236–257.
- Engle, R. F. (2002). Dynamic conditional correlation. *Journal of Business and Economic Statistics* 20, 339–350. https://doi.org/10.1198/073500102288618487
- Lewandowski, D., D. Kurowicka and H. Joe (2009). Generating random correlation matrices based on vines and extended onion method. *Journal of Multivariate Analysis* 100, 1989–2001. https://doi.org/10.1016/j.jmva.2009.04.008
- Newey, W. K. and K. D. West (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55, 703–708. https://doi.org/10.2307/1913610

*(Every DOI above was verified on https://doi.org/.)*

## 15. Citation

```bibtex
@software{roudane_dscc,
  author  = {Roudane, Merwan},
  title   = {{dscc}: Dynamic Spectral Conditional Correlations in Python},
  year    = {2026},
  url     = {https://github.com/merwanroudane/dscc},
  version = {0.1.0}
}
```
Please also cite Abadir and Rockinger (2026), doi:10.1093/ectj/utag017, and Abadir (2023), doi:10.1093/ectj/utac023.

## 16. Author and licence

**Dr Merwan Roudane**
📧 merwanroudane920@gmail.com
🔗 https://github.com/merwanroudane

Released under the MIT Licence — see [LICENSE](LICENSE).
