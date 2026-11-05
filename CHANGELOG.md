# Changelog

## 0.1.1 — 2026-11-05
- Fix: `datasets.read_french_csv` (and so `load_ff5_daily` / `load_industry_daily`) raised
  `OSError: File name too long` on Linux and macOS when given CSV text.
- Package metadata lists Dr Merwan Roudane as maintainer; developer credit on the docs site and README.
- CI: Ubuntu 24.04, Windows and macOS; Python 3.10-3.13 (3.9 on Windows).

## 0.1.0 — 2026-11-05
- Explicit minimal representation Σ = P^δ e^Υ Λ e^{−Υ} P^δ (Abadir 2023).
- DSCC with GARCH (2.5) and EGARCH (2.6)–(2.7) eigenvalue dynamics; ARMA(1,1) skew-symmetric dynamics (2.9).
- Maximum likelihood with greedy per-period δ_t, multi-start, best-point tracking and a Powell polish; profile-likelihood standard errors.
- Multistep forecasts (Section 4), in the paper's form and in the authors' replication form; the "-av" variant.
- Elliptope analytics: exact vol(E_k), slice coverage, exact slab-projection benchmark, DCC simulation (cDCC / Engle / iid).
- DCC(1,1) baseline; GMV portfolios; AV/SD/IR, TO/GL/PL, HAC pairwise tests, v/Δ/d/Δ′/κ; rolling backtest.
- Parula default colour scale; journal-style tables (text, LaTeX, Markdown) and figures.
- Optional numba kernels, with a pure-NumPy fallback that gives identical results.
