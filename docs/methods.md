# dscc — methods, equation map, and reconciliation with the authors' code

Sources
- **[AR26]** Abadir, K. M. and M. Rockinger (2026). Dynamic spectral conditional correlations. *Econometrics Journal*. doi:10.1093/ectj/utag017
- **[A23]** Abadir, K. M. (2023). Explicit minimal representation of variance matrices, and its implication for dynamic volatility models. *Econometrics Journal* 26, 88–104. doi:10.1093/ectj/utac023
- **[Code]** Authors' replication package, Zenodo doi:10.5281/zenodo.20817596 (CC-BY-4.0), MATLAB. It is read as a numerical reference. The Python code is written from the equations.

Rule when the sources disagree: **the paper's equations first, then the authors' code, then convenience.** Every point where the code departs from the text is exposed as an option (`mode="replication"`, `innovations="iid"`, …), so either version can be run.

## 1. Equation → function map

| Paper | Content | Python |
|---|---|---|
| [A23] Lemma 2.1–2.2, Prop. 2.1; [AR26] (2.1) | Σ = P^δ e^Υ Λ e^{−Υ} P^δ | `compose_sigma`, `decompose_sigma` |
| [AR26] (2.3) | Σ^{1/2} | `DSCCResults.std_resid` (ε_t = Σ_t^{−1/2} x_t) |
| [AR26] (2.4) | z_t = e^{−Υ_t} P^{δ_t} x_t | `_filters.dsh_filter` (`z = expU[t-1].T @ x_prev`) |
| [AR26] (2.5) | GARCH(1,1) for Λ_t | `dsh_filter(dynamics="garch")` |
| [AR26] (2.6)–(2.7) | EGARCH(1,1) for log Λ_t, g_i(v) = \|v_i−c_i\| − a_i(v_i−c_i) | `dsh_filter(dynamics="egarch")`, `g_abs` |
| [AR26] (2.9) | ARMA(1,1) for υ_ijt | `_filters.ups_path` |
| [AR26] (2.10)–(2.11) | Q_t, Ξ_t = log Q_t (principal value, [A23] after 3.6) | `orthogonal_from_data`, `xi_from_data` |
| [AR26] (2.13) / [A23] (3.11) | Gaussian log-likelihood, δ_t maximised per summand | `dsh_filter`, `DSCC.loglike`, `DSCC.fit` |
| [AR26] (2.14) | Ω_t = dg(Σ_t)^{−1/2} Σ_t dg(Σ_t)^{−1/2} | `correlation_from_cov`, `DSCCResults.correlation` |
| [A23] (4.2)–(4.8) | bivariate closed form | `tests/test_core.py::_bivariate_closed_form` (acceptance test) |
| [A23] (5.1) | starting values | `default_start` |
| [AR26] §4, (4.1) | multistep forecast, ψ-weighting | `DSCCResults.forecast` |
| [AR26] (3.1) | boundary of E_3 | `elliptope.elliptope_boundary` |
| [AR26] (3.2) | V_3 = π²/2 (general k: LKJ constant) | `elliptope.elliptope_volume` |
| [AR26] §3, Table 1 | slice-by-slice convex-hull coverage | `elliptope.slice_coverage`, `coverage_table`, `simulate_dcc` |

## 2. Fixed conventions (verified against the code)

1. **Dead start-up** at t = 1: Λ_1 = O (GARCH), log Λ_1 = O (EGARCH), Υ_1 = O, δ_1 = 0; t = 1 does not enter the likelihood ([A23] after 4.8).
2. **δ_t search is sequential and greedy**: at each t, given Λ_t and Υ_t, the t-th summand is maximised over δ_t ∈ {0,1}, and the chosen δ_t then enters z_t and Λ_{t+1} ([A23] §3.3 and §4.1; [Code] `likGARCHDSHmult_parMR.m`).
3. **Skew-symmetric storage**: the free elements fill the upper triangle (i < j). The paper writes them below the diagonal. The two conventions are the reparameterisation υ ↦ −υ, and the upper one is kept so that the authors' `thetaM.mat` / `thetaME.mat` load unchanged.
4. **Parameter order** (k variables, l = k(k−1)/2): `[D0, D11, D21, μ, α, β]` (+ `[a, c]` for EGARCH).
5. **Υ_t does not depend on Λ or δ**, so all e^{Υ_t} are computed in a single batched Hermitian eigendecomposition. This is exact, by Lemma 2.1 of [A23].

## 3. Where the code and the paper differ, and what `dscc` does

| # | Point | Paper | Authors' code | `dscc` default | Option |
|---|---|---|---|---|---|
| 1 | Orthogonal complement Q₂t for k ≥ 3 | "a normalised orthogonal complement" (not unique) | `eig(x x')`: q₁ is the **last** column; the null-space basis is whatever LAPACK returns | Householder complement, Q = (q₁, Q₂), deterministic on every platform | `complement="eig"`; `xi=` to pass the exact MATLAB series |
| 2 | ξ_t for k = 2 | (4.5): ξ = arcsin(x₂/‖x‖) | principal `logm` | principal log = atan2(x₂, x₁) | — (eq. 4.5 fails whenever x₁ < 0) |
| 3 | Υ over the forecast horizon | iterate (2.9) and use e^{Υ_{T+h}} | e^{Υ_T} frozen for all h (the Υ forecast is computed but not used) | iterate (2.9) | `forecast(mode="replication")` |
| 4 | Weights at h = 1 | ψ | last in-sample δ̂_T | ψ | `mode="replication"` |
| 5 | "-av" initialisation | "average volatility over the last month" | also averages e^{Υ_t} and x_t over the month | the code's definition (the only full specification) | `init="average"` |
| 6 | DCC simulation for Table 1 | "as in Aielli (2013)", e_t ∼ N(0, Ω_t) | `simDcc.m`: ε_t ∼ N(0, I), independent of the state | Aielli cDCC | `simulate_dcc(innovations="iid")` reproduces Table 1 |
| 7 | "Uniform" design | ρ_ij = γ U(0,1) | ρ_ij and ρ_ji drawn separately (S not symmetric) | symmetric | `uniform_perturbation(symmetric=False)` |
| 8 | Slicing variable | ω₃ | ω₁ (`Xt(:,1)`) | ω₃ | `slice_on=0` |
| 9 | Max-cover benchmark | maximal span of the slice | π√(1 − lb²), the section at the lower edge | same as the paper | `benchmark="projection"` (exact maximum of the slab) |
| 10 | Estimation of θ | MLE with multiple starts (§5.5) | not included; θ is loaded from `thetaM.mat` | `DSCC.fit` (SLSQP, constraints, multi-start) | `method=`, `n_starts=` |
| 12 | Daily out-of-sample returns | "buy-and-hold strategy" over the month | `RtoutS*m`: weights reset every day | buy-and-hold | `rolling_backtest(holding="constant")` |
| 13 | Monthly returns for Table 3 | "monthly returns" | sum of the daily returns | compounded, ∏(1 + r) − 1 | — |
| 14 | DCC block, first forecast day | Σ_{T+1} | `Hproj(:,:,1) = sigmahat`, i.e. Σ_T is used as the h = 1 matrix | Σ_{T+1} from the recursion | — |
| 15 | HAC p-value | one-sided, H₀: α ≤ 0 | `1 − tcdf(abs(t))` (one-sided in \|t\|) | one-sided in t | — (they coincide when Model 1 has the higher return) |
| 11 | Turnover w_hold | weights drifted by the month's returns | drifts w_{h−1} by the returns of month h, and treats simple returns as log returns (`exp(sum)`); includes TO₁ = 0 in the mean | correct timing, product of (1 + r) | `turnover(..., replication=True)` (planned) |

### Effects measured so far
- **#1**: on the first rolling window (FF5, 1980-01-02, T = 1260), NumPy's and MATLAB R2024a's null-space bases give ξ_t series that differ by up to 5.45 rad. No SciPy LAPACK driver (`ev`, `evd`, `evr`, `evx`) reproduces MATLAB. The authors' ξ_t, and hence their θ, cannot be reproduced outside their MATLAB/LAPACK build unless the ξ series itself is passed in.
- **#6**: with α = 0.3, α+β = 0.998, ρ = 0.3 and 400,000 steps, the iid simulator reproduces Table 1 (2.77, 2.73, 2.64, 2.47, 2.27, 1.99, 1.59, 0.91 vs the paper's 2.77, 2.73, 2.64, 2.47, 2.29, 1.95, 1.55, 0.85). Aielli's cDCC, which is what the paper describes, covers noticeably more: 2.90, 2.91, 2.80, …, 1.78, 1.23. The gap to the maximum remains.
- **#10**: the δ_t search makes the likelihood piecewise smooth. On window 1 (MATLAB ξ), a single SLSQP run from (5.1) stopped 10.15 log-likelihood points below the authors' θ (−1825.97 vs −1815.81). Started at the authors' θ, plain SLSQP stepped away and returned a far worse point. `DSCC.fit` therefore tracks the best admissible point at every evaluation, polishes it with derivative-free Powell, and supports several starts (`n_starts=10` matches §5.5 of [AR26]). On the same window and the same MATLAB ξ, the results were:
  - started at the authors' θ, the fit reached −1794.62, which is +21.20 above their stored value;
  - with 10 starts, it reached **−1771.91, which is +43.91**.

  The stored θ of window 1 is therefore not the maximum of the likelihood (2.13).
- **#9**: the exact slab maximum is larger than the paper's benchmark: 3.26 vs 3.14 for (0, 0.125), and 1.53 vs 1.52 for (0.875, 1). So the paper's comparison is conservative, and the true DCC shortfall is larger.
