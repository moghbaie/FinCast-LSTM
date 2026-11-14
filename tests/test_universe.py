"""Point-in-time membership logic on a synthetic change file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fincast.data import universe as u


@pytest.fixture()
def changes(tmp_path: Path) -> pd.DataFrame:
    csv = tmp_path / u.FJA_HISTORY_FILE
    csv.write_text(
        'date,tickers\n2000-01-03,"A,B,C"\n2000-01-10,"A,B,D"\n2000-01-20,"B,D,C"\n',
        encoding="utf-8",
    )
    return u.load_membership_changes(tmp_path)


def test_members_on_uses_last_change_at_or_before_day(changes: pd.DataFrame) -> None:
    assert u.members_on(changes, "2000-01-03") == {"A", "B", "C"}
    assert u.members_on(changes, "2000-01-09") == {"A", "B", "C"}
    assert u.members_on(changes, "2000-01-10") == {"A", "B", "D"}
    assert u.members_on(changes, "2030-01-01") == {"B", "C", "D"}
    with pytest.raises(ValueError):
        u.members_on(changes, "1999-12-31")


def test_membership_matrix_never_looks_ahead(changes: pd.DataFrame) -> None:
    days = pd.bdate_range("2000-01-03", "2000-01-25")
    m = u.membership_matrix(changes, days)
    assert not m.loc[:"2000-01-07", "D"].any()  # D only from 2000-01-10
    assert m.loc["2000-01-10":, "D"].all()
    assert m.loc[:"2000-01-07", "C"].all() and not m.loc["2000-01-10":"2000-01-19", "C"].any()
    assert m.loc["2000-01-20":, "C"].all()  # re-entry


def test_membership_intervals_handles_reentry(changes: pd.DataFrame) -> None:
    iv = u.membership_intervals(changes)
    c = iv[iv.ticker == "C"].reset_index(drop=True)
    assert len(c) == 2
    assert c.loc[0, "end"] == pd.Timestamp("2000-01-10")
    assert c.loc[1, "start"] == pd.Timestamp("2000-01-20") and pd.isna(c.loc[1, "end"])
    a = iv[iv.ticker == "A"].iloc[0]
    assert a.start == pd.Timestamp("2000-01-03") and a.end == pd.Timestamp("2000-01-20")
