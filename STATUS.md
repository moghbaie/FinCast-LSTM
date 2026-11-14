# STATUS / handoff (updated 2026-11-14)

## Done
- Plan approved with decisions A1–A18 (`plan.md` §0). Milestone order: M0 → M1 → M1b → M2 → M3a
  baselines → M3b one LSTM → M8a freeze (≤ 2026-12-21) → full A/B, H = 5/20, multi-seed → M7 → M9.
- **M0 done.**
  - Trainer: CUDA-graph step, early stopping, checkpoint/resume (tested).
  - `PaperLSTM` with 2,752 trainable params.
  - Benchmark: GTX 1650 18 s/epoch at batch 512, 165 s at batch 32.
  - Colab notebook `notebooks/00_colab_setup_benchmark.ipynb`, README, `deviations.md`, pinned
    `requirements.txt`.
- **M1 first pass done.** See `data_audit.md`.
  - Membership from fja05680, 1996 → 2026-08-18.
  - Yahoo prices for 759 of 1,209 tickers. Coverage 56 % of constituent-days in 1996–2015 and
    88 % in 2016–2026.
  - 6 series excluded. The clean panel is `data/processed/adjclose_yahoo_clean.parquet`.
  - Ken French factors to 2026-09-30.
  - Tiingo cross-check of 25 tickers in `results/data_audit/crosscheck_tiingo.csv`.
- Checks pass: ruff, mypy, 10 unit tests, 1 GPU test, 5 data tests.

## Waiting on the user
- Run the Colab notebook and paste the benchmark JSON.
- Approve the M1b approach and its Tiingo quota use (≈ 475 of 500 symbols this month).
- Decide whether to accept a free-data Phase A with trading from 1999 and heavy survivorship
  bias, or approve a paid source.

## Next: M1b data repair
1. Tiingo gap-fill for 191 candidate tickers (`uncovered_spells.csv`, `fillable_by_tiingo > 0`).
2. Rename map via company names (Tiingo metadata, Wikipedia changes).
3. Name-check the inside-spell series (AEE, FE, FOXA, AVB, …) and GNW.
4. Add index changes after 2026-08-18 from Wikipedia or S&P.

## Gotchas
- Windows: always pass `encoding="utf-8"` in Python file I/O (cp1252 corrupted a file once).
- The cuDNN eval path uses ≈ 0.5 MB per row, so keep the eval batch ≤ 2048 on the 4 GB GPU.
- Tiingo free tier: 500 unique symbols/month, 50 req/h. `TiingoClient` throttles to 48/h.
