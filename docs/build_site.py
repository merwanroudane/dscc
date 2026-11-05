"""Build docs/index.html from docs/index.template.html and the outputs of the examples.

Run after ``python examples/ff5_dscc_example.py --out examples/results``:
    python docs/build_site.py
Every number on the page comes from those files, so the site cannot drift from the code.
"""
from __future__ import annotations

import datetime as dt
import html
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS, RES, EX = ROOT / "docs", ROOT / "examples" / "results", ROOT / "examples"


def version() -> str:
    return re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)


def table_html(df: pd.DataFrame, index=True, dec=2) -> str:
    def cell(v):
        if isinstance(v, float):
            return f'<td class="n">{v:.{dec}f}</td>'
        return f"<td>{html.escape(str(v))}</td>"
    cols = ([""] if index else []) + [str(c) for c in df.columns]
    head = "".join(f'<th{" class=n" if i else ""}>{html.escape(c)}</th>' for i, c in enumerate(cols))
    rows = []
    for idx, r in df.iterrows():
        cells = (f"<td><b>{html.escape(str(idx))}</b></td>" if index else "") + "".join(cell(v) for v in r)
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def md_table_html(path: Path) -> str:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    head, body = rows[0], rows[2:]
    th = "".join(f'<th{" class=n" if i else ""}>{html.escape(h)}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=n" if i else ""}>{html.escape(c)}</td>' for i, c in enumerate(r))
                 + "</tr>" for r in body)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def parse_printed_table(block: str, index: bool) -> pd.DataFrame:
    """Parse a whitespace-aligned pandas table printed in the example log.

    Columns that pandas elided as '...' in the printout are dropped, never invented.
    """
    lines = [l for l in block.splitlines()[1:] if l.strip() and not l.startswith("[")]
    head = [h for h in re.split(r"\s{2,}", lines[0].strip()) if h != "..."]   # names may contain one space
    rows = [[t for t in l.split() if t != "..."] for l in lines[1:]]
    if index:
        df = pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows], columns=head)
    else:
        df = pd.DataFrame(rows, columns=head)

    def num(c):
        try:
            return pd.to_numeric(c)
        except (ValueError, TypeError):
            return c
    return df.apply(num)


def log_block(log: str, start: str, stop_markers) -> str:
    i = log.find(start)
    if i < 0:
        return ""
    j = min([k for k in (log.find(m, i + len(start)) for m in stop_markers) if k > 0] or [len(log)])
    return log[i:j].strip()


def fig(name: str, caption: str) -> str:
    src = RES / f"{name}.png"
    if not src.exists():
        return ""
    shutil.copy(src, DOCS / "assets" / f"{name}.png")
    return (f'<figure><a href="assets/{name}.png"><img src="assets/{name}.png" alt="{html.escape(caption)}" '
            f'loading="lazy"></a><figcaption>{caption}</figcaption></figure>')


def pre(code: str, lang="python", out=False) -> str:
    cls = ' class="out"' if out else ""
    return f'<pre{cls}><code class="language-{lang}">{html.escape(code)}</code></pre>'


def example_section() -> str:
    log = (EX / "results_log.txt").read_text(encoding="utf-8") if (EX / "results_log.txt").exists() else ""
    parts = []
    # step 1 -- estimation
    parts.append("<h3>Step 1 — estimate on the paper's first five-year window (1,260 days)</h3>")
    parts.append(pre('ff = dscc.datasets.load_ff5_daily(start="1980-01-02", end="2022-03-31")\n'
                     'factors = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]\n'
                     'res = dscc.DSCC(factors.iloc[:1260], dynamics="egarch").fit()\nprint(res.summary())'))
    summaries = re.findall(r"={78}\n\s+Dynamic Spectral.*?Gaussian innovations\.", log, re.S)
    if summaries:
        parts.append(pre(summaries[-1], "plaintext", out=True))
    for dyn in ("garch", "egarch"):
        p = RES / f"table_params_{dyn}.md"
        if p.exists():
            parts.append(f"<details><summary>DSCC-{dyn} estimates as an HTML table</summary>{md_table_html(p)}</details>")
    # step 2 -- figures
    parts.append("<h3>Step 2 — conditional correlations, eigenvalues and the forecast</h3>")
    parts.append(pre("res.plot_correlations(); res.plot_eigenvalues()\nfc = res.forecast(21, init=\"average\")"))
    parts.append(fig("fig_correlations", "Conditional correlations Ω<sub>t</sub> (2.14), DSCC-λ<sub>egarch</sub>, 1980–1984."))
    parts.append(fig("fig_eigenvalues", "Eigenvalue paths Λ<sub>t</sub> (log scale)."))
    blk = log_block(log, "21-day average forecast", ["Elliptope volume"])
    if blk:
        parts.append(pre(blk, "plaintext", out=True))
    parts.append(fig("fig_corr_heatmap", "Last in-sample correlation matrix (Parula colour scale)."))
    # step 3 -- span
    parts.append("<h3>Step 3 — the span benchmark (Section 3)</h3>")
    parts.append(pre("from dscc import elliptope as el\nel.elliptope_volume(3)           # pi^2/2\n"
                     'el.coverage_table(T=400_000, innovations="iid", slice_on=0)    # Table 1, authors\' simulator\n'
                     'el.coverage_table(T=400_000, innovations="cdcc")               # Aielli (2013), as described'))
    parts.append(fig("fig1_elliptope", "Figure 1 — the elliptope E<sub>3</sub> and its sections at ω<sub>3</sub> = 0, 0.25, 0.75."))
    for innov, cap in (("iid", "authors' simulator (reproduces the paper's Table 1)"),
                       ("cdcc", "Aielli (2013) corrected DCC, the design the paper describes")):
        p = RES / f"table1_{innov}.csv"
        if p.exists():
            t = pd.read_csv(p, header=[0, 1])
            t.columns = [b if a.startswith("Unnamed") or a == "" else f"{a} {b}" for a, b in t.columns]
            parts.append(f"<p><b>Table 1 — DCC convex-hull surface per slice, {cap}.</b></p>")
            parts.append(table_html(t.set_index(t.columns[0]), dec=2))
    parts.append(fig("fig2_coverage", "Figure 2 — DCC coverage (red) against the elliptope (green), slice by slice."))
    # step 4 -- portfolios
    parts.append("<h3>Step 4 — monthly minimum-variance backtest (Section 5)</h3>")
    parts.append(pre("fcs = [dscc.EqualWeight(), dscc.SampleCovForecaster(), dscc.DCCForecaster(),\n"
                     '       dscc.DSCCForecaster("garch", init="last"),\n'
                     '       dscc.DSCCForecaster("egarch", init="average")]\n'
                     "bt = dscc.rolling_backtest(factors, fcs, n_months=60)\n"
                     "bt.performance_table(); bt.pairwise(); bt.turnover_table(); bt.covariance_table()"))
    for start, stop, csv, idx in (("Table 2 --", ["Table 3 --"], "table2_performance.csv", True),
                                  ("Table 3 --", ["Table 4 --"], "table3_pairwise.csv", False),
                                  ("Table 4 --", ["Tables 5-6 --"], "table4_turnover.csv", True),
                                  ("Tables 5-6 --", ["computing time"], "table56_covariance.csv", True)):
        blk = log_block(log, start, stop)
        if not blk:
            continue
        title = blk.splitlines()[0].replace("--", "—")
        parts.append(f"<p><b>{html.escape(title)}</b></p>")
        if (RES / csv).exists():
            df = pd.read_csv(RES / csv, index_col=0 if idx else None)
        else:
            df = parse_printed_table(blk, idx)
        parts.append(table_html(df, index=idx, dec=3))
    t = log_block(log, "computing time", ["\n\n"])
    if t:
        parts.append(f'<p class="note">{html.escape(t.strip())}</p>')
    parts.append(fig("fig4_wealth", "Figure 4 — value of 100 invested at the start of the evaluation period (log scale)."))
    parts.append('<div class="note">The portfolio is formed from the five factor portfolios themselves. The paper\'s '
                 "1,000-stock exercise needs CRSP data (restricted) and the NL-DCC residual block of De Nard, Ledoit "
                 "and Wolf (2021), so it is not reproduced here.</div>")
    return "\n".join(p for p in parts if p)


def main():
    (DOCS / "assets").mkdir(exist_ok=True)
    t = (DOCS / "index.template.html").read_text(encoding="utf-8")
    q = (EX / "quickstart_output.txt").read_text(encoding="utf-8").strip()
    t = (t.replace("__VERSION__", version()).replace("__DATE__", dt.datetime.now().strftime("%Y-%m-%d"))
          .replace("__QUICKSTART__", html.escape(q)).replace("__EXAMPLE__", example_section()))
    (DOCS / "index.html").write_text(t, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    print("wrote", DOCS / "index.html")


if __name__ == "__main__":
    main()
