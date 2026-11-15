"""Ken French daily factor files (evaluation only, never features)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd


def _read_daily_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip()[:8].isdigit()) - 1
    body = []
    for ln in lines[start + 1 :]:
        if not ln.strip()[:8].isdigit():
            break
        body.append(ln)
    header = [c.strip() for c in lines[start].split(",")]
    header[0] = "date"
    df = pd.read_csv(io.StringIO("\n".join(body)), header=None, names=header)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce") / 100.0  # percent -> decimal
    df.columns = [c.strip() for c in df.columns]
    return df


def load_french_daily(folder: Path) -> pd.DataFrame:
    """Columns: Mkt-RF, SMB, HML, RF, Mom, ST_Rev (decimal daily returns)."""
    ff3 = _read_daily_zip(folder / "F-F_Research_Data_Factors_daily_CSV.zip")
    mom = _read_daily_zip(folder / "F-F_Momentum_Factor_daily_CSV.zip")
    rev = _read_daily_zip(folder / "F-F_ST_Reversal_Factor_daily_CSV.zip")
    mom.columns = ["Mom"]
    rev.columns = ["ST_Rev"]
    return ff3.join(mom, how="outer").join(rev, how="outer")
