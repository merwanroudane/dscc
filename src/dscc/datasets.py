"""Real data used in the examples: Fama-French daily factors and industry portfolios.

The Kenneth R. French Data Library carries no explicit redistribution licence,
so the data are NOT shipped inside the package. They are downloaded on first use
from the library's public URL and cached locally, or read from a CSV you already
have (for example ``FF5Factors.CSV`` from the authors' replication package).

Source: French, K. R., Data Library,
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

__all__ = ["load_ff5_daily", "load_industry_daily", "read_french_csv", "cache_dir"]

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "ind5": "5_Industry_Portfolios_daily_CSV.zip",
    "ind10": "10_Industry_Portfolios_daily_CSV.zip",
}


def cache_dir() -> Path:
    """Cache folder: ``$DSCC_DATA`` if set, else ``~/.dscc_data``."""
    p = Path(os.environ.get("DSCC_DATA", Path.home() / ".dscc_data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_french_csv(text_or_path) -> pd.DataFrame:
    """Parse a French-library daily CSV (first table only) into a DataFrame indexed by date.

    Values are kept in percent, as published. Missing values (-99.99 / -999) become NaN.
    """
    if isinstance(text_or_path, (str, Path)) and Path(str(text_or_path)).exists():
        text = Path(text_or_path).read_text(encoding="latin-1")
    else:
        text = str(text_or_path)
    lines = text.splitlines()
    # header row: first field empty (library files) or "date" (authors' CSV)
    start = next(i for i, ln in enumerate(lines)
                 if "," in ln and ln.split(",")[0].strip().lower() in ("", "date")
                 and any(c.isalpha() for c in ln))
    header = [h.strip() for h in lines[start].split(",")]
    header[0] = "date"
    rows = []
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break
            continue
        rows.append(parts[:len(header)])
    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    return df.mask(df <= -99.99)


def _get(key: str, download: bool) -> pd.DataFrame:
    fname = _FILES[key]
    local = cache_dir() / fname
    if not local.exists():
        if not download:
            raise FileNotFoundError(f"{local} not found; call with download=True or pass a CSV path")
        with urllib.request.urlopen(_BASE + fname, timeout=60) as r:   # nosec - fixed public URL
            local.write_bytes(r.read())
    with zipfile.ZipFile(local) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return read_french_csv(z.read(name).decode("latin-1"))


def load_ff5_daily(path=None, download: bool = True, start=None, end=None) -> pd.DataFrame:
    """Fama-French five daily factors (Mkt-RF, SMB, HML, RMW, CMA) plus RF, in percent.

    Parameters
    ----------
    path : optional CSV (e.g. the authors' ``FF5Factors.CSV``); otherwise the
        library file is downloaded once and cached (see :func:`cache_dir`).
    start, end : optional date bounds, e.g. ``"1980-01-02"``, ``"2022-03-31"``.
    """
    if path is not None:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df = df.set_index("date")
    else:
        df = _get("ff5", download)
    return df.loc[start:end]


def load_industry_daily(n: int = 10, download: bool = True, start=None, end=None) -> pd.DataFrame:
    """Value-weighted daily returns (percent) of the 5 or 10 Fama-French industry portfolios."""
    if n not in (5, 10):
        raise ValueError("n must be 5 or 10")
    return _get(f"ind{n}", download).loc[start:end]
