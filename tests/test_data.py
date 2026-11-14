"""Real-data validation (run with `pytest -m data` after `scripts/audit_data.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
AUDIT = ROOT / "results" / "data_audit"

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def membership() -> pd.DataFrame:
    p = PROCESSED / "membership_daily.parquet"
    if not p.exists():
        pytest.skip("run scripts/audit_data.py first")
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def adjclose() -> pd.DataFrame:
    p = PROCESSED / "adjclose_yahoo.parquet"
    if not p.exists():
        pytest.skip("run scripts/audit_data.py first")
    return pd.read_parquet(p)


def test_index_size_is_plausible(membership: pd.DataFrame) -> None:
    n = membership.sum(1)
    assert n.between(480, 510).all(), n[~n.between(480, 510)]


def test_membership_on_nyse_sessions_only(membership: pd.DataFrame) -> None:
    from fincast.data.validate import nyse_sessions

    s = nyse_sessions(str(membership.index[0].date()), str(membership.index[-1].date()))
    assert membership.index.equals(s)


def test_prices_aligned_and_positive(adjclose: pd.DataFrame, membership: pd.DataFrame) -> None:
    assert adjclose.index.equals(membership.index)
    assert adjclose.index.is_monotonic_increasing and not adjclose.index.duplicated().any()
    assert (adjclose.stack() > 0).all()


def test_no_price_before_listing_is_backfilled(adjclose: pd.DataFrame) -> None:
    """Each series is contiguous-ish: no isolated values before a long gap (backfill artefacts)."""
    first = adjclose.apply(pd.Series.first_valid_index)
    for t in first.dropna().index[:200]:
        s = adjclose[t].loc[first[t] :]
        assert s.iloc[:5].notna().all() or len(s) < 5, t


def test_audit_summary_is_current() -> None:
    p = AUDIT / "summary.json"
    if not p.exists():
        pytest.skip("run scripts/audit_data.py first")
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["validation"]["series"] == s["yahoo_tickers_with_data"]
    assert 0.0 < s["coverage_member_days"]["B_2016_2026"] <= 1.0
