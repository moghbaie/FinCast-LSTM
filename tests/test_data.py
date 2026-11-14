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
    p = PROCESSED / "adjclose_yahoo_clean.parquet"
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
    vals = adjclose.to_numpy()
    assert (vals[~pd.isna(vals)] > 0).all()


def test_excluded_series_absent_and_returns_plausible(
    adjclose: pd.DataFrame, membership: pd.DataFrame
) -> None:
    excl = pd.read_csv(AUDIT / "excluded_series.csv")["ticker"]
    assert not set(excl) & set(adjclose.columns)
    r = adjclose.pct_change(fill_method=None)
    mem = membership.reindex(columns=adjclose.columns, fill_value=False)
    extreme = ((r.abs() > 0.5) & mem).sum()
    assert (extreme < 5).all(), extreme[extreme >= 5]


def test_audit_summary_is_current() -> None:
    p = AUDIT / "summary.json"
    if not p.exists():
        pytest.skip("run scripts/audit_data.py first")
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["validation"]["series"] == s["yahoo_tickers_with_data"]
    assert len(s["excluded_series"]) < 20
    assert 0.0 < s["coverage_member_days"]["B_2016_2026"] <= 1.0
