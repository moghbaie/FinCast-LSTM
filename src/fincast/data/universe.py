"""Point-in-time S&P 500 membership from the fja05680/sp500 change file.

The file has one row per change date with the complete constituent list valid from that date.
Interpretation [A]: the list in a row dated d applies to trading days >= d until the next row.
Because only effective dates (not announcement dates) are available, membership on day d counts
as known at the close of day d (CLAUDE.md, Rule 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FJA_HISTORY_FILE = "S&P 500 Historical Components & Changes (Updated).csv"


def load_membership_changes(folder: Path) -> pd.DataFrame:
    """Return (date, members: frozenset[str]) sorted by date; duplicate dates keep the last row."""
    raw = pd.read_csv(folder / FJA_HISTORY_FILE)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["members"] = raw["tickers"].map(
        lambda s: frozenset(t.strip() for t in str(s).split(",") if t.strip())
    )
    out = raw.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out[["date", "members"]]


def members_on(changes: pd.DataFrame, day: pd.Timestamp | str) -> frozenset[str]:
    """Constituents valid on `day` (last change row with date <= day)."""
    ts = pd.Timestamp(day)
    idx = int(np.searchsorted(changes["date"].to_numpy(), np.datetime64(ts), side="right")) - 1
    if idx < 0:
        raise ValueError(
            f"{ts.date()} is before the first membership list ({changes['date'][0].date()})"
        )
    return frozenset(changes["members"].iloc[idx])


def membership_matrix(changes: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean matrix (trading days x tickers): True if the ticker is a constituent that day."""
    tickers = sorted(set().union(*changes["members"]))
    pos = np.searchsorted(changes["date"].to_numpy(), days.to_numpy(), side="right") - 1
    col = {t: i for i, t in enumerate(tickers)}
    mat = np.zeros((len(days), len(tickers)), dtype=bool)
    for row_i, p in enumerate(pos):
        if p >= 0:
            mat[row_i, [col[t] for t in changes["members"].iloc[p]]] = True
    return pd.DataFrame(mat, index=days, columns=tickers)


def membership_intervals(changes: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per contiguous membership spell: ticker, start, end (exclusive; NaT = member)."""
    rows: list[tuple[str, pd.Timestamp, pd.Timestamp | None]] = []
    open_since: dict[str, pd.Timestamp] = {}
    prev: frozenset[str] = frozenset()
    for d, mem in zip(changes["date"], changes["members"], strict=True):
        for t in mem - prev:
            open_since[t] = d
        for t in prev - mem:
            rows.append((t, open_since.pop(t), d))
        prev = mem
    for t, s in open_since.items():
        rows.append((t, s, end))
    df = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    return df.sort_values(["ticker", "start"]).reset_index(drop=True)


def month_end_lists(changes: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Paper-style month-end constituent lists (Fischer & Krauss p. 4)."""
    return pd.DataFrame(
        {"date": month_ends, "members": [members_on(changes, d) for d in month_ends]}
    )
