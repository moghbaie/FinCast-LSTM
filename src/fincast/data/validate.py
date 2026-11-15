"""Price-series validation and universe coverage statistics (M1 data audit)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

EXTREME_RETURN = 0.5  # |daily total return| above this is flagged for review
STALE_RUN = 5  # this many identical consecutive closes is flagged


def nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    import pandas_market_calendars as mcal

    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index.normalize(), name="date")


@dataclass
class SeriesReport:
    ticker: str
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    n_rows: int
    n_duplicate_dates: int
    n_non_session_dates: int
    n_missing_sessions: int  # NYSE sessions between first and last with no row
    n_nonpositive_close: int
    n_nan_adjclose: int
    n_extreme_returns: int
    max_abs_return: float
    n_stale_runs: int
    adj_ratio_decreases: int  # Adj Close / Close should not fall over time except via splits
    passed: bool


def validate_series(ticker: str, df: pd.DataFrame, sessions: pd.DatetimeIndex) -> SeriesReport:
    """df: index = dates, columns include 'Close' and 'Adj Close' (Yahoo layout)."""
    if df.empty:
        return SeriesReport(ticker, None, None, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, False)
    idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df.set_axis(idx)
    dup = int(idx.duplicated().sum())
    df = df[~idx.duplicated()].sort_index()
    in_range = sessions[(sessions >= df.index[0]) & (sessions <= df.index[-1])]
    non_session = int((~df.index.isin(sessions)).sum())
    missing = int((~in_range.isin(df.index)).sum())
    close = df["Close"]
    adj = df["Adj Close"]
    ret = adj.pct_change(fill_method=None)
    extreme = ret.abs() > EXTREME_RETURN
    same = close.diff().eq(0)
    run_id = (~same).cumsum()
    runs = same.groupby(run_id).sum()
    ratio = (adj / close).replace([np.inf, -np.inf], np.nan)
    ratio_drop = int((ratio.pct_change(fill_method=None) < -0.02).sum())
    rep = SeriesReport(
        ticker=ticker,
        first=df.index[0],
        last=df.index[-1],
        n_rows=len(df),
        n_duplicate_dates=dup,
        n_non_session_dates=non_session,
        n_missing_sessions=missing,
        n_nonpositive_close=int(((close <= 0) | (adj <= 0)).sum()),
        n_nan_adjclose=int(adj.isna().sum()),
        n_extreme_returns=int(extreme.sum()),
        max_abs_return=float(ret.abs().max()) if ret.notna().any() else 0.0,
        n_stale_runs=int((runs >= STALE_RUN - 1).sum()),
        adj_ratio_decreases=ratio_drop,
        passed=False,
    )
    rep.passed = (
        dup == 0 and rep.n_nonpositive_close == 0 and rep.n_nan_adjclose == 0 and non_session == 0
    )
    return rep


def reports_frame(reports: list[SeriesReport]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in reports])


def coverage_by_year(member: pd.DataFrame, available: pd.DataFrame) -> pd.DataFrame:
    """Share of constituent-days with a price, per calendar year.

    member, available: boolean frames (sessions x tickers) on identical axes.
    """
    m = member.to_numpy()
    a = available.reindex_like(member).fillna(False).to_numpy(dtype=bool)
    years = pd.DatetimeIndex(member.index).year
    rows = []
    for y in np.unique(years):
        sel = years == y
        md = m[sel]
        covered = (md & a[sel]).sum()
        total = md.sum()
        tick_any = md.any(0)
        tick_cov = (md & a[sel]).any(0)
        rows.append(
            {
                "year": int(y),
                "member_days": int(total),
                "covered_days": int(covered),
                "coverage": covered / total if total else np.nan,
                "members_in_year": int(tick_any.sum()),
                "members_with_any_price": int((tick_any & tick_cov).sum()),
                "avg_members_per_day": total / sel.sum(),
                "avg_covered_per_day": covered / sel.sum(),
            }
        )
    return pd.DataFrame(rows)
