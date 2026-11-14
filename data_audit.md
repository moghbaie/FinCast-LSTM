# Data audit (M1) — free sources

Status: **first pass, 2026-11-14.** Numbers come from `scripts/audit_data.py`. Outputs are in
`results/data_audit/` (`summary.json`, `coverage_by_year.csv`, `series_validation.csv`,
`uncovered_spells.csv`, `crosscheck_tiingo.csv`). Raw downloads are in
`data/raw/<source>/<date>/`, each with a `manifest.json` recording the URL, UTC time, SHA-256
and size. `data/` is git-ignored.

**Bottom line.**
- Free data supports **Phase B (2016–2026) and Phase C (2027)** reasonably well: 88 % of
  constituent-days have prices now, and about 95 % should be reachable after Tiingo gap-filling.
- Free data supports **Phase A only as a biased, approximate replication.** Membership starts in
  1996, not 1989. Prices exist for only 35–50 % of constituent-days in 1996–2005. The missing
  firms are mostly those later acquired, merged or bankrupt, so survivorship bias is large exactly
  where the paper found most of its profit (the 1990s).

## 1. Sources

| Source | What | Snapshot | Licence / terms | Used for |
|---|---|---|---|---|
| fja05680/sp500 (GitHub) | Daily S&P 500 membership change file, 1996-01-02 → 2026-08-18; derived from Clenow's *Trading Evolved* list (originally Norgate) plus manual Wikipedia updates | commit pinned in manifest, 2026-11-14 | MIT | point-in-time universe |
| Yahoo Finance via `yfinance` 1.8.0 | Daily OHLCV, `Close` and `Adj Close` (split- and dividend-adjusted), splits and dividends | 2026-11-14, 1,209 symbols requested | Unofficial API; Yahoo ToS restrict redistribution and commercial use; personal research use only, never redistributed | primary prices |
| Tiingo `supported_tickers.zip` | Every ticker with first and last price date (109k rows) | 2026-11-14 | Free file | coverage estimate, gap-fill candidates |
| Tiingo EOD API (free Starter) | OHLCV + `adjClose`, `divCash`, `splitFactor` | on demand | **Internal use only**; 500 unique symbols/month, 50 req/h, 1,000 req/day | cross-check, gap-fill |
| Ken French data library | Daily Mkt-RF, SMB, HML, RF, Mom, ST_Rev, 1926-01-26 → **2026-09-30** | 2026-11-14 | Free for academic use | evaluation only (Sharpe excess, factor alphas, MKT) |
| Stooq, Nasdaq.com | — | not used | — | per-ticker CSV access now restricted; no delisted coverage; dropped |

## 2. Universe (point-in-time membership)

- 2,720 change rows. **1,209 unique tickers** in 1,262 membership spells; 503 current members.
  Daily index size min/median/max = 487 / 497 / 507.
- **1990–1995 is not covered.** The paper's first study periods (training from Jan 1990, trading
  from Dec 1992) cannot be built. The first possible trading period starts about 3 years after
  1996-01-02, i.e. early 1999.
- The file's author notes that **1996–2000 may be missing a few constituents** (size 487–494).
- The last change in the snapshot is **2026-08-18**. Index changes after that date are missing
  and must be added from S&P announcements or Wikipedia before Phase C (M8).
- Tickers are as of each date, so renames appear as one ticker removed and another added on the
  same day. Interpretation: a list dated d applies from session d; membership is treated as
  known at the close of d. Unit tests: `tests/test_universe.py`.
- The paper used month-end lists. We have daily changes; Phase A faithful mode samples them at
  month-end (`universe.month_end_lists`).

## 3. Price coverage (survivorship gap)

Share of constituent-days with a price, by source:

| Year | Members/day | Yahoo | Tiingo (metadata, upper bound) | Union |
|---|---|---|---|---|
| 1996 | 487 | 0.38 | 0.47 | 0.48 |
| 1999 | 491 | 0.45 | 0.55 | 0.56 |
| 2002 | 495 | 0.50 | 0.63 | 0.64 |
| 2005 | 496 | 0.53 | 0.66 | 0.67 |
| 2008 | 497 | 0.61 | 0.80 | 0.81 |
| 2011 | 497 | 0.66 | 0.87 | 0.87 |
| 2014 | 498 | 0.72 | 0.91 | 0.91 |
| 2016 | 505 | 0.77 | 0.93 | 0.93 |
| 2019 | 505 | 0.84 | 0.94 | 0.94 |
| 2022 | 504 | 0.92 | 0.97 | 0.97 |
| 2024 | 503 | 0.96 | 0.99 | 0.99 |
| 2026 | 503 | 0.99 | 1.00 | 1.00 |

All years are in `coverage_by_year.csv`. Over constituent-days, after exclusions: **1996–2015 = 56%, 2016–2026 = 88% (Yahoo only).** Yahoo has data for 759 of 1,209 tickers. The other 450 are delisted,
acquired, or renamed symbols.

Why the gaps exist:

1. **Delisted firms** (bankrupt or acquired). Yahoo keeps only current listings. Tiingo has some
   (YHOO, TWX, CPWR) but not, for example, Lehman, Enron, WorldCom, the pre-2009 GM or Motorola.
   These firms are excluded *because of what happened to them later*, which is survivorship bias.
2. **Renamed tickers.** The membership file switches symbol on the rename date, while Yahoo stores
   the full history only under the new symbol. The largest Yahoo gaps include still-listed firms
   whose ticker changed (BK, MMC, PKI, BLL, GPS, LB, …). These are recoverable with a
   rename map.
3. **Recent take-privates and acquisitions** (K, IPG, WBA, HES, MRO, CMA, JWN, …). Tiingo has
   most of these (`fillable_by_tiingo` in `uncovered_spells.csv`: 191 tickers, 497k
   constituent-days).

## 4. Ticker identity (wrong-company risk)

Reused tickers can map a historical constituent to a different, later company:

- **23 Yahoo series begin inside a membership spell** (≈ 40.6k constituent-days, 1.0 %). Some
  are legitimate truncated histories (e.g. AEE, FE, FOXA). Others are a different entity
  (e.g. `DEC`: Yahoo's series starts 2021-01-04, but the 1990s constituent was Digital
  Equipment).
- **30 series have |daily return| > 50 % on a membership day.** Most are genuine: AIG, C, HIG,
  GNW, FNMA and FMCC in 2008–09, and AAPL in Sept 2000. Series with **repeated** impossible jumps
  are almost certainly the wrong company or bad adjustment data: **EP (27), CPWR (26),
  MCIC (9), PARA (7)**. These are excluded until reviewed by hand.
- **Exclusion rule implemented** (`scripts/audit_data.py` → `results/data_audit/excluded_series.csv`;
  clean panel `data/processed/adjclose_yahoo_clean.parquet`). A series is excluded if it fails a
  hard check (non-positive `Close`/`Adj Close`, duplicate dates, non-session dates) or has at least
  5 membership days with |r| > 50 %. **Currently excluded: CPWR (26), DEC (zero adjusted prices),
  EP (27), GNW (6), MCIC (9), PARA (7).** GNW (Genworth, 2008–09) may be genuine and is pending
  manual review; the others look like a wrong entity or bad adjustments.
- Not yet handled: series that start inside a membership spell but have no extreme returns
  (e.g. AEE, FE, FOXA, AVB). They are kept, and their pre-start membership days count as
  uncovered. Each is checked by name in M1b.

## 5. Series validation (Yahoo, 759 series)

| Check | Result |
|---|---|
| Duplicate dates, non-positive prices, dates that are not NYSE sessions | none |
| NaN `Adj Close` | 4 series with 1 missing value each (AVB, PARA, PSKY, USBC); 755 of 759 pass every hard check |
| Missing NYSE sessions inside a series | 3 series |
| Stale runs (≥ 5 identical closes) | 289 series, mostly 1990s low-priced history. Median 0 runs per series, max 298 (EP, a wrong-entity candidate) |
| Member-day return quantiles | 0.01 %: −28.6 %, 0.1 %: −14.1 %, median +0.05 %, 99.9 %: +14.9 %, 99.99 %: +32.2 % |
| `Adj Close` as total return | Yahoo adjusts multiplicatively for splits and cash dividends, so returns from `Adj Close` match a total-return index up to Yahoo's dividend data quality. Cross-checked against Tiingo in §6 |

## 6. Cross-check against Tiingo

*Pending (running).* 25 randomly chosen tickers (seed 0) with more than 1,000 days. Metrics:
share of days where the two daily returns differ by less than 1 bp and less than 10 bp, the
largest difference, and the correlation.

## 7. Factor data

Ken French daily factors run to **2026-09-30**. The library updates monthly with a lag, so the
end of Phase B is evaluated against factors once they are published. Factors are never model
inputs.

## 8. Consequences for each phase

| Phase | Data verdict | Label |
|---|---|---|
| A (paper period) | Trading can start ≈ 1999 at the earliest (paper: Dec 1992). Coverage 35–70 % of constituent-days in 1999–2015, biased toward survivors. Comparison with the paper is restricted to overlapping years (1999–2015) | **Approximate replication with residual survivorship bias** — never "reproduction" |
| B (2016–2026) | 73–95 % (Yahoo) → ≈ 93–99 % after Tiingo gap-fill and rename linking | approximate (free data), bias small and quantified |
| C (2027) | Current members ≈ 99 % covered. Training window 2024–2026 ≈ 95 %+ | adequate; index changes after 2026-08-18 must be added |

Point-in-time correctness: membership is used as of each date. Prices are adjusted backwards
(split/dividend back-adjustment), so only **returns** are used as features (CLAUDE.md, Rule 2).
Return ratios are unaffected by later adjustments, except where Yahoo revises dividend data.

## 9. Remaining work (M1b, proposed)

1. Tiingo gap-fill for the 191 candidate tickers. They fit in this month's free quota of 500
   symbols; at 50 req/h that is about 4 hours.
2. Rename map: link old to new tickers using company names (Tiingo metadata for the 450 missing
   symbols, Wikipedia changes table). This must stay within the 500-symbol quota, which is tight;
   alternatively spread it over two months.
3. Manual review of the identity-suspect series (EP, CPWR, MCIC, PARA, DEC, and the 23
   inside-spell starts).
4. Add index changes after 2026-08-18 (before M8).
5. Re-run the audit. Acceptance: no unexplained extreme-return series, a documented rename map,
   and coverage reported per study period.
