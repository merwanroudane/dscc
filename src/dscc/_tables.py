"""Journal-style text tables for DSCC results.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import numpy as np

__all__ = ["results_summary", "stars", "params_frame", "to_latex", "to_markdown", "frame_to_latex"]


def params_frame(res, decimals: int = 4):
    """Coefficient / (std err) frame with stars -- the single source of the LaTeX and Markdown tables."""
    import pandas as pd
    df = res.to_frame()
    coef = [f"{c:.{decimals}f}{stars(p)}" for c, p in zip(df["coef"], df["P>|t|"])]
    se = [f"({s:.{decimals}f})" if np.isfinite(s) else "--" for s in df["std err"]]
    return pd.DataFrame({"Estimate": coef, "Std. err.": se}, index=df.index)


def frame_to_latex(df, caption: str = "", label: str = "", notes: str = "") -> str:
    """Booktabs LaTeX from a DataFrame, with a notes row."""
    cols = list(df.columns)
    out = ["\\begin{table}[htbp]", "\\centering"]
    if caption:
        out.append(f"\\caption{{{caption}}}")
    if label:
        out.append(f"\\label{{{label}}}")
    out += [f"\\begin{{tabular}}{{l{'r' * len(cols)}}}", "\\toprule",
            " & ".join([""] + [str(c) for c in cols]) + " \\\\", "\\midrule"]
    for idx, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        name = str(idx).replace("_", "\\_").replace("[", "{[}").replace("]", "{]}")
        out.append(" & ".join([name] + cells) + " \\\\")
    out.append("\\bottomrule")
    if notes:
        out.append(f"\\multicolumn{{{len(cols) + 1}}}{{p{{0.9\\linewidth}}}}{{\\footnotesize {notes}}} \\\\")
    out += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


_NOTES = ("*** p<0.01, ** p<0.05, * p<0.10. Standard errors in parentheses, from the "
          "numerical Hessian of the profile log-likelihood with $\\delta_t$ at its MLE.")


def to_latex(res, caption="DSCC maximum-likelihood estimates", label="tab:dscc") -> str:
    return frame_to_latex(params_frame(res), caption, label, _NOTES)


def to_markdown(res) -> str:
    df = params_frame(res)
    lines = ["| Parameter | Estimate | Std. err. |", "|---|---:|---:|"]
    lines += [f"| {i} | {r['Estimate']} | {r['Std. err.']} |" for i, r in df.iterrows()]
    lines.append("")
    lines.append(_NOTES.replace("$\\delta_t$", "delta_t"))
    return "\n".join(lines)


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _num(v: float, width: int, dec: int) -> str:
    return f"{v:{width}.{dec}f}" if np.isfinite(v) else f"{'--':>{width}}"


def results_summary(res) -> str:
    W = 78
    rule, thin = "=" * W, "-" * W
    m = res.model
    lab = {"garch": "DSCC-garch (2.5)", "egarch": "DSCC-egarch (2.6)"}[res.dynamics]
    conv = "yes" if (res.opt is None or res.opt.success) else "NO"
    left = [("Model:", lab), ("Variables (k):", f"{res.k}"), ("Observations (T):", f"{res.T}"),
            ("Q2t complement:", m.complement if m.xi is not None else "user xi")]
    right = [("Log-likelihood:", f"{res.loglik:.3f}"), ("AIC:", f"{res.aic:.3f}"),
             ("BIC:", f"{res.bic:.3f}"), ("Converged:", conv)]
    lines = [rule, "Dynamic Spectral Conditional Correlation -- ML estimates".center(W), rule]
    for (a, b), (c, d) in zip(left, right):
        lines.append(f"{a:<18}{b:<22}{c:<18}{d:>20}")
    lines.append(thin)
    lines.append(f"{'':<26}{'coef':>10}{'std err':>10}{'t':>9}{'P>|t|':>9}{'[0.025':>8}{'0.975]':>8}")
    lines.append(thin)
    df = res.to_frame()
    groups = [("Eigenvalue dynamics, Lambda_t", ("D0[", "D11[", "D21[")),
              ("Skew-symmetric dynamics, Upsilon_t (2.9)", ("mu[", "alpha[", "beta[")),
              ("Asymmetry function g (2.7)", ("a[", "c["))]
    for title, pref in groups:
        rows = [n for n in df.index if n.startswith(pref)]
        if not rows:
            continue
        lines.append(title)
        for n in rows:
            r = df.loc[n]
            nm = n if len(n) <= 24 else n[:23] + "~"
            lines.append(f"  {nm:<24}{r['coef']:10.4f}"
                         + _num(r["std err"], 10, 4) + _num(r["t"], 9, 3) + _num(r["P>|t|"], 9, 3)
                         + _num(r["[0.025"], 8, 3) + _num(r["0.975]"], 8, 3)
                         + f" {stars(r['P>|t|'])}")
    lines.append(thin)
    lines.append("Permutation component delta_t (Proposition 2.1, eq. 2.13)")
    lines.append(f"  share of t with delta_t = 1: {1 - res.psi:8.4f}      psi = Pr(delta = 0): {res.psi:8.4f}")
    lines.append(rule)
    lines.append("Notes: *** p<0.01, ** p<0.05, * p<0.10. Standard errors from the numerical Hessian")
    lines.append("of the profile log-likelihood with delta_t held at its MLE; '--' = not available")
    lines.append("(parameter on a bound, or non-positive local curvature). Gaussian innovations.")
    return "\n".join(lines)
