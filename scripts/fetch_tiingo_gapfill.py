"""M1b step 1: download Tiingo EOD history for membership spells Yahoo cannot cover.

Candidates are the tickers with `fillable_by_tiingo > 0` in results/data_audit/uncovered_spells.csv
(Tiingo metadata shows prices over at least one uncovered member day). Uses one Tiingo symbol per
ticker out of the free tier's 500 unique symbols per month; throttled to 48 requests/hour.
Resumable: re-run with the same --day to continue. Identity checks happen afterwards (M1b step 3).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from fincast.data import acquire

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=date.fromisoformat, default=None, help="snapshot folder date")
    ap.add_argument("--limit", type=int, default=None, help="fetch at most N tickers")
    args = ap.parse_args()

    spells = pd.read_csv(ROOT / "results" / "data_audit" / "uncovered_spells.csv")
    cand = spells[spells["fillable_by_tiingo"] > 0].sort_values(
        "fillable_by_tiingo", ascending=False
    )
    tickers = cand["ticker"].drop_duplicates().tolist()[: args.limit]
    print(f"{len(tickers)} candidate tickers, {int(cand['fillable_by_tiingo'].sum())} member-days")
    folder = acquire.fetch_tiingo_prices(tickers, day=args.day, log=True)
    got = sum((folder / f"{t}.csv").exists() for t in tickers)
    missing = [t for t in tickers if (folder / f"{t}.missing").exists()]
    print(f"done: {got} with data, {len(missing)} without: {missing}")
    print(f"folder: {folder}")


if __name__ == "__main__":
    main()
