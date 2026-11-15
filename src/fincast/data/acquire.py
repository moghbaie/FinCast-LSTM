"""Download raw data from free sources into immutable, manifest-tracked folders.

Layout: data/raw/<source>/<download_date>/<files> plus a manifest.json in the same folder, with
URL, UTC download time, SHA-256 and byte size for every file. Existing files are never
overwritten: re-running on the same day skips them (resumable downloads).

Sources (see data_audit.md for licences and limitations):
- fja05680/sp500 (MIT): historical S&P 500 membership 1996-, pinned to a git commit.
- Ken French data library: daily FF3 factors, momentum, short-term reversal (evaluation only).
- Tiingo supported_tickers.zip: all tickers with first/last price dates (free file, no API quota).
- Tiingo EOD API: daily OHLCV + adjClose (free: 500 unique symbols/month, 50 req/h, 1000 req/day).
- Yahoo Finance via yfinance: daily OHLCV, Adj Close (unofficial API; current listings only).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from fincast.utils.secrets import get_secret

RAW_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
USER_AGENT = "fincast-research/0.1 (academic replication)"

FJA_REPO = "fja05680/sp500"
FJA_FILES = [
    "S&P 500 Historical Components & Changes (Updated).csv",
    "sp500_ticker_start_end.csv",
    "sp500_changes_since_2019.csv",
    "sp500.csv",
]
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = [
    "F-F_Research_Data_Factors_daily_CSV.zip",
    "F-F_Momentum_Factor_daily_CSV.zip",
    "F-F_ST_Reversal_Factor_daily_CSV.zip",
]
TIINGO_TICKERS_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def snapshot_dir(source: str, day: date | None = None, root: Path = RAW_ROOT) -> Path:
    d = root / source / (day or datetime.now(UTC).date()).isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record(
    folder: Path, name: str, url: str, content: bytes, extra: dict[str, Any] | None = None
) -> None:
    mpath = folder / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"files": {}}
    manifest["files"][name] = {
        "url": url,
        "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": _sha256(content),
        "bytes": len(content),
        **(extra or {}),
    }
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def download_file(url: str, folder: Path, name: str, extra: dict[str, Any] | None = None) -> Path:
    path = folder / name
    if path.exists():
        return path
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)
    _record(folder, name, url, r.content, extra)
    return path


def fja_head_commit() -> str:
    r = requests.get(
        f"https://api.github.com/repos/{FJA_REPO}/commits/master",
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
        timeout=60,
    )
    r.raise_for_status()
    return str(r.json()["sha"])


def fetch_fja05680(commit: str | None = None) -> Path:
    """Download the membership files at a pinned commit (default: current master head)."""
    sha = commit or fja_head_commit()
    folder = snapshot_dir("fja05680_sp500")
    for f in FJA_FILES:
        url = f"https://raw.githubusercontent.com/{FJA_REPO}/{sha}/{quote(f)}"
        download_file(url, folder, f, {"commit": sha})
    return folder


def fetch_french() -> Path:
    folder = snapshot_dir("ken_french")
    for f in FRENCH_FILES:
        download_file(FRENCH_BASE + f, folder, f)
    return folder


def fetch_tiingo_supported_tickers() -> Path:
    folder = snapshot_dir("tiingo_meta")
    download_file(TIINGO_TICKERS_URL, folder, "supported_tickers.zip")
    return folder


class TiingoClient:
    """Throttled Tiingo EOD client that respects the free-tier hourly limit."""

    def __init__(self, max_per_hour: int = 48) -> None:
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {get_secret('TIINGO_API_KEY')}",
        }
        self.min_interval = 3600.0 / max_per_hour
        self._last = 0.0

    def prices(self, ticker: str, start: str = "1990-01-01", end: str | None = None) -> bytes:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"startDate": start, "format": "csv", "resampleFreq": "daily"}
        if end:
            params["endDate"] = end
        for attempt in range(5):
            r = requests.get(
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                params=params,
                headers=self.headers,
                timeout=120,
            )
            self._last = time.monotonic()
            if r.status_code != 429:
                break
            time.sleep(600 * (attempt + 1))  # hourly quota hit: back off, then retry
        if r.status_code == 404:
            return b""
        r.raise_for_status()
        return r.content


def fetch_tiingo_prices(
    tickers: Iterable[str],
    client: TiingoClient | None = None,
    day: date | None = None,
    log: bool = False,
) -> Path:
    """Download full daily history per ticker into one snapshot folder (resumable).

    Pass `day` to keep a long run (it can cross UTC midnight) in a single snapshot folder.
    """
    client = client or TiingoClient()
    folder = snapshot_dir("tiingo_eod", day)
    todo = [
        t
        for t in tickers
        if not (folder / f"{t}.csv").exists() and not (folder / f"{t}.missing").exists()
    ]
    for i, t in enumerate(todo, 1):
        name = f"{t}.csv"
        if log:
            print(f"[{datetime.now(UTC):%H:%M:%S}Z] {i}/{len(todo)} {t}", flush=True)
        content = client.prices(t)
        if not content.strip() or content.strip().startswith(b"[]"):
            (folder / f"{t}.missing").write_text("no data", encoding="utf-8")
            continue
        (folder / name).write_bytes(content)
        _record(folder, name, f"tiingo:daily/{t}/prices", content)
    return folder


def yahoo_symbol(ticker: str) -> str:
    """fja05680 / Wikipedia style (BRK.B) -> Yahoo style (BRK-B)."""
    return ticker.replace(".", "-")


def fetch_yahoo_prices(
    tickers: Iterable[str], start: str = "1990-01-01", batch: int = 50, pause: float = 2.0
) -> Path:
    """Daily OHLCV with both Close and Adj Close per ticker, stored as one parquet per ticker."""
    import yfinance as yf

    folder = snapshot_dir("yahoo")
    todo = [
        t
        for t in tickers
        if not (folder / f"{t}.parquet").exists() and not (folder / f"{t}.missing").exists()
    ]
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        sym = {yahoo_symbol(t): t for t in chunk}
        df = yf.download(
            list(sym),
            start=start,
            auto_adjust=False,
            actions=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        for ys, t in sym.items():
            sub = (
                df[ys]
                if isinstance(df.columns, pd.MultiIndex) and ys in df.columns.get_level_values(0)
                else None
            )
            if sub is None or sub.dropna(how="all").empty:
                (folder / f"{t}.missing").write_text("no data", encoding="utf-8")
                continue
            sub = sub.dropna(how="all")
            path = folder / f"{t}.parquet"
            sub.to_parquet(path)
            _record(folder, path.name, f"yfinance:{ys}", path.read_bytes(), {"yahoo_symbol": ys})
        time.sleep(pause)
    return folder
