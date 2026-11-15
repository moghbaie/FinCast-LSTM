"""M1 data audit: membership, price coverage (survivorship gap), validation, factor data.

Reads the latest raw snapshots in data/raw (download them first with fincast.data.acquire;
`--download` does that) and writes:
  results/data_audit/coverage_by_year.csv     constituent-day coverage per source and year
  results/data_audit/series_validation.csv    per-ticker validation report (Yahoo)
  results/data_audit/uncovered_spells.csv     membership spells without Yahoo prices
  results/data_audit/summary.json             headline numbers used in data_audit.md
Also builds data/processed/membership_daily.parquet and data/processed/adjclose_yahoo.parquet.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fincast.data import acquire, universe, validate
from fincast.data.french import load_french_daily

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data_audit"
PROCESSED = ROOT / "data" / "processed"
US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "NYSE MKT", "NYSE ARCA", "BATS"}


def latest(source: str) -> Path:
    folders = sorted((acquire.RAW_ROOT / source).iterdir())
    if not folders:
        raise FileNotFoundError(f"no snapshot for {source}; run with --download")
    return folders[-1]


def load_yahoo(folder: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(folder.glob("*.parquet")):
        df = pd.read_parquet(p)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        out[p.stem] = df
    return out


def tiingo_meta_availability(sessions: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    z = zipfile.ZipFile(latest("tiingo_meta") / "supported_tickers.zip")
    meta = pd.read_csv(z.open("supported_tickers.csv"))
    meta = meta[meta["exchange"].isin(US_EXCHANGES) & meta["assetType"].isin(["Stock"])]
    meta["ticker"] = meta["ticker"].str.upper().str.replace("-", ".", regex=False)
    meta = meta.dropna(subset=["startDate", "endDate"])
    avail = pd.DataFrame(False, index=sessions, columns=tickers)
    sv = sessions.to_numpy()
    for t, g in meta[meta["ticker"].isin(tickers)].groupby("ticker"):
        col = np.zeros(len(sessions), dtype=bool)
        for s, e in zip(pd.to_datetime(g["startDate"]), pd.to_datetime(g["endDate"]), strict=True):
            col |= (sv >= np.datetime64(s)) & (sv <= np.datetime64(e))
        avail[t] = col
    return avail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="refresh raw snapshots first")
    ap.add_argument(
        "--end", default=None, help="last session to audit (default: latest Yahoo date)"
    )
    args = ap.parse_args()

    if args.download:
        acquire.fetch_fja05680()
        acquire.fetch_french()
        acquire.fetch_tiingo_supported_tickers()

    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fja_dir = latest("fja05680_sp500")
    changes = universe.load_membership_changes(fja_dir)
    tickers = sorted(set().union(*changes["members"]))
    if args.download:
        acquire.fetch_yahoo_prices(tickers)
    ydir = latest("yahoo")
    yahoo = load_yahoo(ydir)
    last_price_day = max(df.index.max() for df in yahoo.values())
    end = pd.Timestamp(args.end) if args.end else last_price_day
    sessions = validate.nyse_sessions(str(changes["date"].iloc[0].date()), str(end.date()))

    # --- membership
    member = universe.membership_matrix(changes, sessions)
    member.to_parquet(PROCESSED / "membership_daily.parquet")
    sizes = member.sum(1)

    # --- Yahoo validation + availability
    first_price_day = min(df.index.min() for df in yahoo.values())
    all_sessions = validate.nyse_sessions(str(first_price_day.date()), str(end.date()))
    reports = [validate.validate_series(t, yahoo[t].loc[:end], all_sessions) for t in yahoo]
    rep = validate.reports_frame(reports)
    rep.to_csv(OUT / "series_validation.csv", index=False)
    adj = pd.DataFrame({t: df["Adj Close"] for t, df in yahoo.items()}).reindex(sessions)
    adj.to_parquet(PROCESSED / "adjclose_yahoo.parquet")
    # Exclusions: hard-check failures (non-positive prices, duplicates, non-session dates) and
    # identity suspects (>= 5 membership days with |r| > 50 %: wrong company / bad adjustment).
    mem_y = member.reindex(columns=adj.columns, fill_value=False)
    extreme = ((adj.pct_change(fill_method=None).abs() > 0.5) & mem_y).sum()
    hard_cols = ["n_duplicate_dates", "n_non_session_dates", "n_nonpositive_close"]
    hard = rep.set_index("ticker")[hard_cols].sum(axis=1) > 0
    excluded = pd.DataFrame({"hard_check_failure": hard, "extreme_member_days": extreme})
    excluded = excluded[excluded["hard_check_failure"] | (excluded["extreme_member_days"] >= 5)]
    excluded.index.name = "ticker"
    excluded.to_csv(OUT / "excluded_series.csv")
    adj = adj.drop(columns=excluded.index)
    adj.to_parquet(PROCESSED / "adjclose_yahoo_clean.parquet")
    avail_y = adj.notna().reindex(columns=member.columns, fill_value=False)
    avail_t = tiingo_meta_availability(sessions, list(member.columns))

    cov_y = validate.coverage_by_year(member, avail_y)
    cov_t = validate.coverage_by_year(member, avail_t)
    cov_u = validate.coverage_by_year(member, avail_y | avail_t)
    cov = cov_y[["year", "avg_members_per_day", "members_in_year"]].copy()
    cov["yahoo"] = cov_y["coverage"]
    cov["tiingo_meta"] = cov_t["coverage"]
    cov["union"] = cov_u["coverage"]
    cov["yahoo_members_with_any_price"] = cov_y["members_with_any_price"]
    cov.to_csv(OUT / "coverage_by_year.csv", index=False)

    # --- uncovered spells (missing constituent-days by ticker), and Tiingo gap-fill candidates
    miss = (member & ~avail_y).sum()
    tiingo_fill = (member & ~avail_y & avail_t).sum()
    spells = universe.membership_intervals(changes)
    unc = (
        pd.DataFrame({"missing_member_days_yahoo": miss, "fillable_by_tiingo": tiingo_fill})
        .query("missing_member_days_yahoo > 0")
        .join(spells.groupby("ticker").agg(first_start=("start", "min"), last_end=("end", "max")))
        .sort_values("missing_member_days_yahoo", ascending=False)
    )
    unc.index.name = "ticker"
    unc.to_csv(OUT / "uncovered_spells.csv")

    # --- partial-history flag: Yahoo series starts after the ticker's first membership day
    first_member = member.idxmax()
    first_price = pd.Series({t: yahoo[t].index.min() for t in adj.columns})
    late_start = (first_price > first_member.reindex(first_price.index)).sum()

    ff = load_french_daily(latest("ken_french"))
    period = {
        "A_1996_2015": (sessions.year >= 1996) & (sessions.year <= 2015),
        "B_2016_2026": sessions.year >= 2016,
    }
    covered = (member & avail_y).to_numpy().sum(1)
    total = member.to_numpy().sum(1)
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "membership_source": str(fja_dir.relative_to(ROOT)),
        "membership_first": str(changes["date"].iloc[0].date()),
        "membership_last_change": str(changes["date"].iloc[-1].date()),
        "unique_tickers": len(tickers),
        "spells": len(spells),
        "daily_size_min_med_max": [int(sizes.min()), int(sizes.median()), int(sizes.max())],
        "yahoo_snapshot": str(ydir.relative_to(ROOT)),
        "yahoo_tickers_with_data": len(yahoo),
        "yahoo_tickers_missing": len(tickers) - len(yahoo),
        "audit_end": str(end.date()),
        "coverage_member_days": {
            k: float(covered[v].sum() / total[v].sum()) for k, v in period.items()
        },
        "yahoo_late_start_tickers": int(late_start),
        "excluded_series": sorted(excluded.index.tolist()),
        "validation": {
            "series": len(rep),
            "passed_hard_checks": int(rep["passed"].sum()),
            "with_extreme_returns": int((rep["n_extreme_returns"] > 0).sum()),
            "with_missing_sessions": int((rep["n_missing_sessions"] > 0).sum()),
            "with_stale_runs": int((rep["n_stale_runs"] > 0).sum()),
            "with_non_session_dates": int((rep["n_non_session_dates"] > 0).sum()),
        },
        "tiingo_fillable_member_days": int(tiingo_fill.sum()),
        "tiingo_fill_candidate_tickers": int((tiingo_fill > 0).sum()),
        "french": {
            "first": str(ff.index.min().date()),
            "last": str(ff.index.max().date()),
            "columns": list(ff.columns),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(cov.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
