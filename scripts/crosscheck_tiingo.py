"""Cross-check Yahoo adjusted returns against Tiingo on a random sample of tickers (M1).

Uses at most --n Tiingo symbols out of the free tier's 500 unique symbols per month.
Writes results/data_audit/crosscheck_tiingo.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fincast.data import acquire

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data_audit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    adj = pd.read_parquet(ROOT / "data" / "processed" / "adjclose_yahoo.parquet")
    pool = sorted(adj.columns[adj.notna().sum() > 1000])
    rng = np.random.default_rng(args.seed)
    sample = sorted(rng.choice(pool, size=min(args.n, len(pool)), replace=False).tolist())
    folder = acquire.fetch_tiingo_prices(sample)

    rows = []
    for t in sample:
        p = folder / f"{t}.csv"
        if not p.exists():
            rows.append({"ticker": t, "tiingo": "missing"})
            continue
        ti = pd.read_csv(p, parse_dates=["date"]).set_index("date")["adjClose"]
        ti.index = pd.DatetimeIndex(ti.index).tz_localize(None).normalize()
        ry = adj[t].pct_change(fill_method=None)
        rt = ti.pct_change(fill_method=None)
        both = pd.concat([ry, rt], axis=1, keys=["y", "t"]).dropna()
        d = (both["y"] - both["t"]).abs()
        rows.append(
            {
                "ticker": t,
                "tiingo": "ok",
                "overlap_days": len(both),
                "share_abs_diff_lt_1bp": float((d < 1e-4).mean()),
                "share_abs_diff_lt_10bp": float((d < 1e-3).mean()),
                "max_abs_diff": float(d.max()),
                "date_of_max_diff": str(d.idxmax().date()) if len(d) else None,
                "corr": float(both["y"].corr(both["t"])),
                "mean_ret_diff_bp": float((both["y"] - both["t"]).mean() * 1e4),
            }
        )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "crosscheck_tiingo.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
