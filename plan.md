# Research plan — FinCast-LSTM

Status: **APPROVED 2026-11-13 with modifications (§0).** M0 and M1 authorised; model training
beyond feasibility benchmarks waits for the M0/M1 gate report.

## 0. Approved decisions (2026-11-13) — these override anything below

| # | Decision |
|---|---|
| A1 | **Compute:** Google Colab is the primary training environment (free GPU first; Pro/A100 only if benchmarks justify the cost). All training is resumable with checkpoints in Google Drive. The same `fincast` package runs locally and on Colab; notebooks are thin drivers. |
| A2 | **Git:** private GitHub repo. Commit code, configs, tests, docs. Never commit raw/licensed data, credentials, or large checkpoints. |
| A3 | **Data:** free sources only (no WRDS/CRSP confirmed, no purchases without approval). Phase A is labelled **approximate replication**, never exact reproduction. |
| A4 | **Paper:** the 2017 FAU discussion paper is the reference; uncertainty from not having the EJOR version is recorded (O-1). |
| A5 | **Batch size:** 512 initially, subject to memory/speed benchmarks; compared with 32 on a small experiment. Larger batches are *not* assumed to preserve the original optimization behaviour; the comparison is reported. |
| A6 | **Validation:** last 20 % of training dates, purged by the label horizon (gap ≥ H days between the last fit-label window and the first validation feature date). |
| A7 | **Reproduction:** `faithful` (paper assumptions) and `strict` (corrected) results are reported separately, never mixed. |
| A8 | **Phase B:** 2016–2026 walk-forward, only data available at each decision point (incl. point-in-time membership). |
| A9 | **Horizons:** H = 1 direction is primary. H = 5/20 are secondary and must not delay the 2027 primary model. |
| A10 | **Costs/execution:** actual turnover at {2, 5, 10, 20} bps per half-turn + explicit short-borrow fee; **next-session (open) execution is the realistic case**; close-to-close kept only as the paper-comparable case. |
| A11 | **Benchmarks:** PyTorch DNN; LOG, RF, XGBoost, GRU, 5-day reversal. **Simple baselines are run before expensive neural experiments.** |
| A12 | **Seeds:** 1 seed for debugging/feasibility; 5 seeds for final H = 1; 3 for H = 5/20 (subject to measured cost). |
| A13 | **Scheduling:** both a Windows Task Scheduler job and a GitHub Actions workflow (preferred for CPU inference if runtime and data-source terms permit). |
| A14 | **Forecast records:** append-only ledger with model version, forecast timestamp (UTC), target date, predicted probability, data cutoff; frozen protocol archived with an independent timestamp (e.g. Zenodo/OSF DOI or OpenTimestamps of the protocol hash). |
| A15 | **Rigor:** no presumption of an LSTM edge; the key test is LSTM vs reversal and simple baselines *after costs*, across regimes. |
| A16 | **Deadline:** primary 2027 model + protocol frozen **before 2027-01-04**; historical reproduction must not delay it. |
| A17 | **Order:** data audit → environment benchmark → baselines → one working LSTM → (validated) full replication and secondary horizons. |
| A18 | **Deviations log:** every departure from the paper (data, architecture, preprocessing, validation, execution) is recorded in `deviations.md` with rationale. |

### Revised milestone order (supersedes the order in §5)

M0 environment + git + benchmark → M1 free-data acquisition + audit → **M1b data repair (Tiingo gap-fill, rename map, identity review, post-2026-08-18 index changes)** → M2 pipeline core + leakage
tests → M3a simple baselines (naive, reversal, momentum, LOG, RF, XGBoost) → M3b one working LSTM
(1 seed) → **M8a freeze primary 2027 model/protocol (target ≤ 2026-12-21)** → M5 full Phase A
(faithful/strict), M6 Phase B, H = 5/20, GRU/DNN, multi-seed → M7 uncertainty → M9 live 2027.

Reproduce Fischer & Krauss (2018) (Phase A), test whether the result holds 2016–2026 (Phase B),
run a pre-registered prospective forecast for 2027 (Phase C), and add an uncertainty-quantification
extension (Phase U). Method details and page references: `paper_analysis.md`.

---

## 1. Scientific assessment (concise)

- **Claim.** A 1-layer, 25-unit LSTM fed 240 standardized daily returns ranks S&P 500 stocks for
  next-day out-/under-performance of the cross-sectional median. Long top 10 / short flop 10 earns
  0.46 %/day gross (Sharpe 5.8), 0.26 %/day after 5 bps/half-turn (Sharpe 2.3), 1992–2015, beating
  RF, a DNN and logistic regression.
- **Strengths.** Survivor-bias-aware universe, non-overlapping out-of-sample trading years, fitted
  statistics restricted to training data, simple and well-specified model, multiple benchmarks,
  DM/PT tests.
- **Weaknesses that matter for us:**
  1. The edge is concentrated in 1993–2000 and is roughly zero after costs from 2010 (p. 18–20).
     The prior for Phase B is "little or no net edge".
  2. Execution at the signal close, full-turnover costs of 5 bps per half-turn, no borrow cost, and
     no impact. Sharpe ratios this high from daily large-cap reversal are a sign of optimistic
     execution assumptions.
  3. The signal is largely short-term reversal (R² ≈ 0.52 against a 5-day reversal rule). Part of
     that is bid–ask bounce, which close-to-close returns overstate.
  4. Batch size, learning rate, and the composition of the validation set are not reported. There
     is a 1-day label overlap at the train/trade boundary. Dropout was tuned on "initial
     experiments" without saying which data was used. The paper reports a single run per period,
     so seed variance is unknown.
  5. The DM test pools 115,000 picks and ignores within-day cross-sectional dependence, so the
     p-values are overstated.
- **Expected reproduction outcome.** With survivor-bias-free data, a qualitatively similar result
  (LSTM ≈ RF > DNN > LOG in the 1990s, decay later) is plausible. Exact figures depend on the data
  vendor and the training details listed in the open items O-1 … O-11.

## 2. Exact reproduction requirements

| Requirement | Paper | Available to us? | Plan |
|---|---|---|---|
| Month-end S&P 500 constituent lists 12/1989–09/2015 | Thomson Reuters | Only with a licence (Datastream, CRSP, Norgate, Compustat). Free lists start ≈1996 and are incomplete | Decision D1 |
| Daily total-return index 1990–10/2015, including delisted stocks | Thomson Reuters RI | CRSP `ret` or Norgate total-return series with a licence. Free sources lack most delisted stocks | Decision D1 |
| NYSE trading calendar | implicit | Yes (`pandas_market_calendars`) | — |
| LSTM: 1×25 units, dropout 0.16 on inputs and recurrent connections, RMSprop, softmax 2, patience 10, ≤1,000 epochs, 80/20 split | stated | Yes | PyTorch, own variational-dropout LSTM cell |
| Batch size, learning rate, validation composition | not stated | — | D3, D4 |
| RF 1,000 trees, depth 20, √p | stated | Yes | scikit-learn |
| DNN 31-31-10-5-2 maxout, dropout 0.5, L1 1e-5 (H2O) | partly stated | Yes | PyTorch re-implementation (D9); remaining settings from Krauss et al. (2017) |
| LOG, L2 grid of 100, 5-fold CV, L-BFGS | stated | Yes | scikit-learn |
| Fama–French market and risk-free rate (MKT, Sharpe) | Ken French library | Yes, free | Evaluation only |
| Python 3.5 / keras / H2O versions | stated loosely | Not needed | Document as a deviation |
| Reported numbers | Table 2, Table 3, Fig. 3, Fig. 5, Table 4 | Yes (discussion paper) | EJOR version: D2 |

## 3. Unavailable or ambiguous information

See `paper_analysis.md` §10 (O-1 … O-11). The ones with material impact:

1. **Point-in-time constituents and delisted total returns** (O-11, data). This decides whether we
   can call Phase A a reproduction or only an approximate replication.
2. **Batch size and learning rate** (O-2). They affect results and compute time a lot.
3. **Validation set composition** (O-3): chronological, or the last stocks in the sorted array.
4. **1-day label overlap at the train/trade boundary.** We run the faithful version and a strict
   version.
5. **DNN details** (O-8). They need Krauss, Do & Huck (2017).
6. **Fig. 3 values** (O-9). They must be digitized from the figure.

## 4. Data acquisition strategy

### 4.1 Candidate sources (to be verified hands-on in M1 and written up in `data_audit.md`)

| Source | Historical constituents | Delisted stocks | Corporate actions / TR | Coverage | Licence | Verdict |
|---|---|---|---|---|---|---|
| Thomson Reuters / LSEG Datastream (original) | Yes | Yes | RI = total return | 1990– | Institutional | Exact reproduction, if you have access |
| CRSP via WRDS | Yes (`dsp500list`) | Yes, with delisting returns | `ret` includes dividends | 1925– | Institutional | Best scientific substitute |
| Norgate Data (Platinum) | Yes (S&P 500 historical membership) | Yes | Total-return adjustment | ≈1990– | Paid personal subscription; Windows updater fits this machine | Best paid option without a university |
| Sharadar (Nasdaq Data Link) | SP500 membership table | Yes | Adjusted closes plus dividends | ≈1998– (to verify) | Paid | Covers Phase B only, not 1990s |
| Tiingo | No | Partly | Adjusted close | 1960s– | Free tier has symbol/request caps; paid tier cheap | Phase B/C price source |
| Yahoo Finance (yfinance) | No | Mostly no | Adjusted close | 1960s– | ToS restricts automated use; unofficial API | Cross-check only |
| Stooq | No | Few | Split-adjusted; dividends inconsistent | 1980s– | Free | Cross-check only |
| Nasdaq.com historical | No | No | Unadjusted | ≈10 years | Free, manual | Not suitable |
| fja05680/sp500 (GitHub), Wikipedia changes list | Yes, from ≈1996, reconstructed | n/a | n/a | 1996– | Open | Free membership fallback; validate against licensed source |
| Ken French data library | n/a | n/a | n/a | 1926– | Free academic | MKT, RF, Carhart + ST-reversal factors (evaluation only) |

### 4.2 Strategy by tier (choose in D1)

- **Tier 1 (recommended if you have university access):** CRSP/WRDS for Phases A and B
  (constituents, `ret`, delisting returns, `permno` mapping). Tiingo or Norgate for daily 2027
  updates. Phase A is labelled "reproduction with a substitute vendor".
- **Tier 2:** Norgate for Phases A–C. Delisted stocks and membership are covered. Labelled
  "approximate replication (vendor substitute)".
- **Tier 3 (free only):** fja05680 membership (1996+) plus Tiingo/Yahoo/Stooq prices. Phase A then
  starts trading in ≈1999, misses delisted stocks without a price, and is labelled
  "approximate replication with residual survivorship bias". The audit quantifies the share of
  historical constituents with prices per year. We never substitute current constituents.

### 4.3 Always

- Raw downloads are immutable under `data/raw/<source>/<download_date>/` with a manifest (URL,
  date, SHA-256, row counts).
- Ticker→permanent-ID map (permno / Norgate asset ID) handles renames and re-used tickers.
- Validation checks (also run as `pytest -m data`): trading-calendar alignment, duplicate dates,
  non-positive prices, |daily return| > 50 % flagged against corporate actions, stale-price runs,
  membership count ≈ 500 each month, every member has prices, and vendor cross-check of returns on
  overlapping stocks (Yahoo vs primary; agreement rate reported).

## 5. Implementation plan and milestones

Every milestone ends with a **gate**: (1) data validation where data is touched, (2) `ruff format`,
`ruff check`, `mypy src`, `pytest -W error` (+ `-m data`), (3) the milestone's evaluation run, and
(4) a short results note. A milestone is not done until its gate passes.

| M | Content | Gate: validation / tests / eval | Acceptance |
|---|---|---|---|
| **M0** Setup | git init, `pyproject.toml`, `requirements.txt` (pinned), seeds and provenance (versions, git hash, config hash per run), LSTM throughput benchmark on the GTX 1650 (custom cell vs cuDNN, batch 32/256/512) | lint/type/tests on the skeleton; benchmark table | Reproducible env from scratch; measured s/epoch |
| **M1** Data & audit | Acquisition for the chosen tier; universe builder; validation suite; `data_audit.md` | Validation report: coverage, missingness, outliers, vendor agreement, membership counts; `pytest -m data` | Every source, date, mapping and fix documented; survivorship coverage quantified |
| **M2** Pipeline core | Study-period splitter (`check_split`), returns, standardization, 240-sequences (strided, no copies), targets (median, H = 1/5/20), 31-feature set | **Leakage tests**: perturb all data after T and require features, scalers, training sets and predictions unchanged. **Alignment tests**: targets at t+H, features ≤ t. **Synthetic known-answer tests**: (a) random-walk data gives accuracy ≈ 50 % and the monkey distribution; (b) injected reversal signal is detected | All tests pass in faithful and strict mode |
| **M3** Models | LSTM (variational dropout), RF, DNN (maxout), LOG; then GRU, XGBoost, momentum, reversal, naive | Unit tests: LSTM 2,752 params, shapes, seed determinism, early-stopping restores best; smoke run on 1 study period | All models train on 1 period end to end |
| **M4** Backtest & evaluation | Portfolios (k ∈ {10,50,100,150,200}), costs (paper full-turnover and actual-turnover), execution modes, metrics, statistical tests, figures | Metric functions tested against statsmodels / hand-computed values; monkey mean ≈ 0 | Reproduces Table 3's structure on synthetic data |
| **M5** Phase A | Full 23-period run (or tier-limited), 5 seeds, all four paper models + simplified reversal rule + patterns (Fig. 7–8) | Compare to Table 2/3, Fig. 3/5, Table 4 | `reproduction_report.md`; pre-declared criteria (§6.1) evaluated honestly |
| **M6** Phase B | 2016-01 → latest 2026 close, all Phase B models, H = 1/5/20 | Same gates; DM/SPA tests with multiple-testing control | `modern_validation.md` |
| **M7** Phase U | Ensembles, bootstrap, conformal | Coverage/width/calibration by regime | `uncertainty_extension.md` (separate from the reproduction) |
| **M8** Phase C freeze | Final protocol, config, code hash, final model trained through the 2026-12-31 close; dry run of the daily job on Dec 2026 | Ledger integrity test; dry-run forecasts reproducible bit-for-bit | `forecast_2027.md` frozen **before 2027-01-04** (first 2027 session) |
| **M9** Phase C live | Daily forecasts during 2027; quarterly interim reports; final evaluation in Jan 2028 | Daily data validation before each forecast | Untouched protocol; all forecasts logged before outcomes |

**Calendar constraint.** M8 must be frozen by about 2026-12-28, which is roughly 6 weeks from now.
If Phase B is not finished by mid-December, Phase C is frozen with the paper's original LSTM
configuration (already a defensible pre-declared choice), and Phase B results are not used to
choose it. This is stated in `forecast_2027.md`.

### 5.1 Phase A design (faithful)

- 23 study periods (750 training / 250 trading days, step 250) from the first trading day of 1990.
  n_i = constituents on the last training day. Membership and delisting rules as on p. 5–6.
- Features, targets, models and costs exactly as in `paper_analysis.md` §4–7. Ambiguities use the
  defaults in §10 of that file and decisions D3/D4.
- **Two variants:** `faithful` (paper boundary, close-to-close) and `strict` (embargo
  t ≤ T − 1 for the labels, no other change). The headline is `faithful` and the gap is reported.
- No tuning in Phase A. Fixed seeds {0,…,4}, results per seed and seed-mean.

### 5.2 Phase B design (modern validation)

- Evaluation (trading) days: first session of 2016 → last available 2026 close (≈ 11 trading
  blocks of 250 days, the last one partial). Training windows reach back into 2013–2015 (D6).
- Models: LSTM (paper configuration, no retuning), GRU (same size and training), LOG, RF, XGBoost
  (fixed sensible configuration plus one nested-validation tuned variant), momentum (12-1 month and
  20-day), short-term reversal (5-day, paper's rule), naive baselines (constant class /
  random ranking = monkey / previous-day-class persistence).
- Same inputs per model family: sequence models get the 240 standardized returns, memory-free
  models get the 31 multi-period returns. A cross-check gives the LSTM the 31 features and the
  tree models flattened sequences, to separate architecture from information set.
- Tasks: H = 1 direction (primary); H = 5 and H = 20 returns (secondary): regression on the
  return in excess of the cross-sectional mean, plus direction. Embargo t ≤ T − H.
- **Decomposition of effects:**
  1. Forecasting: accuracy, AUC, log loss, calibration on all stocks and on the selected top/flop k.
  2. Portfolio construction: same forecasts under k ∈ {10,…,200}, equal vs probability-weighted,
     and a sector-neutral variant.
  3. Execution and costs: gross; paper's full-turnover 5 bps; actual-turnover costs at
     {2, 5, 10, 20} bps per half-turn; borrow cost (D8); close-to-close vs next-open execution.
- Regimes: calendar years, VIX terciles, market drawdown periods (2018Q4, 2020 COVID, 2022 rate
  shock), high vs low cross-sectional dispersion.

### 5.3 Phase C design (prospective 2027; details fixed in `forecast_2027.md` at M8)

- Universe: S&P 500 constituents as of each forecast day, captured daily (point-in-time snapshot
  saved with a timestamp).
- Primary: next-day direction relative to the cross-sectional median for every constituent and the
  k = 10 long-short paper portfolio (paper trading only, no real money). Secondary: 5- and 20-day
  return forecasts.
- Models: the LSTM (paper configuration) plus pre-declared comparators: LOG, RF, the reversal rule,
  and naive baselines.
- **Retraining schedule (proposed, D7):** paper-consistent. One model per seed, trained on the 750
  sessions ending 2026-12-31, used for all of 2027 (≈ 250 sessions). No intra-year retraining. A
  secondary quarterly-retrain arm is optional.
- Each trading day after the close (+ data-availability buffer): ingest, validate, forecast for the
  next session, append to `forecasts_2027/ledger.csv` with UTC timestamp and SHA-256 chain, and
  commit. Outcomes are scored only afterwards. A missing day is logged as missing, never backfilled
  with hindsight.
- Pre-declared metrics and tests (same as §6) and reporting dates (end of each quarter; final
  report in Jan 2028).

### 5.4 Phase U design (uncertainty extension, separate report)

- Point model: single LSTM. Ensembles: 5 and 10 seed ensembles (deep ensembles).
  Bootstrap: B = 10 refits on moving-block bootstrap samples of stock-days (only on a subset of
  periods, for compute). MC dropout as a cheap comparator.
- Conformal methods for time series: split conformal on the last chronological block of training
  (baseline, exchangeability-violating), **adaptive conformal inference (ACI, Gibbs & Candès
  2021)**, and **EnbPI** (Xu & Xie 2021). Applied to (a) H = 5/20 return intervals and
  (b) classification sets / calibrated probabilities for H = 1.
- Evaluation: empirical coverage vs nominal (80/90/95 %), mean width, interval score, ECE and
  reliability diagrams, rolling coverage by regime. **Decision quality:** does filtering or sizing
  positions by uncertainty improve net Sharpe and drawdown? Pre-declared rules: trade only
  confident picks, and inverse-width sizing.

## 6. Evaluation protocol

### 6.1 Reproduction criteria (pre-declared, Phase A)

"Qualitatively reproduced" if all of: (i) LSTM k = 10 gross mean daily return > 0 with a
Newey–West t > 1.96; (ii) the ordering of mean returns LSTM ≥ RAF > DNN, LOG at k = 10, or a
documented exception; (iii) decline in the post-2010 sub-period. "Quantitatively close" if the
paper's 0.46 % lies inside our 95 % block-bootstrap CI (seed-averaged) and the LSTM accuracy is
within ±1 pp of 54.3 %. All numbers are reported regardless of the outcome.

### 6.2 Metrics

- Statistical: accuracy, balanced accuracy, precision, recall, ROC-AUC, log loss, Brier,
  ECE/reliability, on all stocks and on top/flop k. For H = 5/20: MAE, RMSE, rank IC.
- Financial: annualized return and volatility, Sharpe (excess over RF), Sortino, max drawdown,
  VaR/CVaR, turnover, cost drag, gross vs net, long/short leg split, alpha vs Carhart + short-term
  reversal factors, comparison to MKT and to the reversal rule.

### 6.3 Inference

- Uncertainty: stationary block bootstrap 95 % CIs (Politis–Romano) for means, Sharpe and accuracy.
  Seed dispersion (min/median/max over 5 seeds).
- Forecast comparison: DM test on **daily cross-sectionally averaged** loss differentials with HAC
  variance (also the paper's pooled version, for comparability). PT test.
- Strategy comparison: Ledoit–Wolf (2008) HAC Sharpe-difference test, and Newey–West alpha
  t-stats.
- Multiple testing: Holm correction within each pre-declared comparison family. Hansen's SPA and
  Romano–Wolf stepdown across all Phase B model × k × cost variants. Deflated Sharpe ratio with the
  total number of variants tried, which is logged automatically.

## 7. Proposed directory structure (created as a skeleton)

```
FinCast-LSTM/
├── CLAUDE.md  plan.md  paper_analysis.md  README.md  requirements.txt  pyproject.toml
├── data_audit.md  reproduction_report.md  modern_validation.md
├── uncertainty_extension.md  forecast_2027.md              (written at their milestones)
├── config/            phase_a.yaml, phase_b.yaml, phase_c.yaml, models/*.yaml
├── src/fincast/
│   ├── data/          acquire.py, universe.py, calendar.py, validate.py
│   ├── features/      returns.py, sequences.py, targets.py
│   ├── models/        lstm.py, gru.py, dnn.py, sklearn_models.py, xgb.py, rules.py
│   ├── training/      trainer.py, early_stopping.py, seeds.py
│   ├── backtest/      splits.py, portfolio.py, costs.py, execution.py
│   ├── evaluation/    metrics.py, stat_tests.py, uncertainty.py, conformal.py
│   ├── viz/           plots.py
│   └── utils/         provenance.py, io.py
├── scripts/           audit_data.py, benchmark_lstm.py, run_phase_a.py, run_phase_b.py,
│                      run_uncertainty.py, train_phase_c.py, forecast_daily.py, score_2027.py
├── notebooks/         01_data_audit, 02_phase_a_results, 03_phase_b_results, 04_uncertainty
├── tests/             test_no_leakage.py, test_alignment.py, test_splits.py, test_models.py,
│                      test_metrics.py, test_data.py (marker: data)
├── results/           tables/, figures/   (small, versioned)
├── forecasts_2027/    ledger.csv, frozen/  (protocol hash, model weights manifest)
├── data/              raw/, interim/, processed/   (git-ignored)
└── runs/              per-run outputs, logs, checkpoints  (git-ignored)
```

## 8. Computational requirements (to be confirmed by the M0 benchmark)

Machine: i7-9850H (6C/12T), 64 GB RAM, GTX 1650 4 GB, 55 GB free disk.

| Workload | Count | Per unit (estimate) | Total |
|---|---|---|---|
| LSTM Phase A (≈204k train sequences × 240 steps, variational dropout means no fused cuDNN kernel) | 23 periods × 5 seeds = 115 fits | 20–60 min at batch 512; 3–6× longer at batch 32 | **40–115 GPU-h** (batch 512); 200–600 GPU-h at batch 32 |
| RF 1,000 trees, depth 20, ≈255k × 31 | 115 fits | 5–15 min (12 threads) | 10–30 CPU-h |
| DNN, LOG (CV 100 × 5), XGBoost | 115 each | 1–10 min | < 20 CPU-h |
| Phase B: LSTM + GRU, H = 1/5/20 | ≈11 periods × 5 seeds × 2 archs × 3 horizons = 330 fits | 20–60 min | **110–330 GPU-h** (shared heads or fewer seeds for H = 5/20 can cut this ≈3×) |
| Phase U: 10-member ensembles, B = 10 bootstrap on a subset | ≈100 extra fits | 20–60 min | 35–100 GPU-h |
| Storage | data < 2 GB; runs/checkpoints ≈ 5–15 GB | | fits in 55 GB |

Wall-clock on this laptop is about 2–4 weeks of continuous GPU time at batch 512. CPU jobs run in
parallel. Options: cut H = 5/20 to 3 seeds, use a shared multi-horizon head, or rent a cloud GPU
(≈ 5–10× faster) for M5/M6. The laptop GPU runs warm (71 °C idle), so long runs need checkpointing
and resume.

## 9. Decisions as originally proposed (resolved — see §0)

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Data tier: CRSP/WRDS, Datastream, Norgate (paid), or free-only | CRSP/WRDS or Datastream if you have institutional access; otherwise Norgate. Free-only is possible, but Phase A then becomes a 1999+ approximate replication with residual survivorship bias |
| D2 | Can you supply the EJOR published PDF for the cross-check? | Yes if available; otherwise the discussion paper is the reference |
| D3 | LSTM batch size (unspecified) | 512 primary, plus a batch-32 sensitivity check on 3 study periods; lr 1e-3 |
| D4 | Validation split (unspecified) | Chronological last 20 % of training dates; keras-style split as a sensitivity check |
| D5 | Phase A boundary handling | Report both `faithful` (headline) and `strict` |
| D6 | Phase B window | Trading 2016-01 → latest 2026 close, training reaching back to 2013 |
| D7 | Phase C retraining | Annual (paper-consistent): train once on data through 2026-12-31; optional quarterly arm declared as secondary |
| D8 | Costs and execution in Phases B/C | Base 5 bps/half-turn on actual turnover, grid {2,5,10,20}; borrow 50 bps p.a. on shorts; close-to-close and next-open both reported, next-open as "investable" |
| D9 | DNN implementation | PyTorch maxout re-implementation (no Java/H2O), documented deviation |
| D10 | Compute budget | Full 5 seeds for Phases A and B on H = 1; 3 seeds for H = 5/20; local GPU unless you approve cloud |
| D11 | Provenance / timestamping of 2027 forecasts | `git init` now; daily ledger commits with a hash chain; optionally push to a private GitHub repo or OSF for third-party timestamps |
| D12 | Daily 2027 operation | Windows Task Scheduler job after 17:30 ET, or you trigger it manually; missed days logged as missing |

## 10. Risks

- **Data licence/coverage** decides how Phase A can be labelled. Mitigation: tiered labelling and
  coverage statistics in the audit.
- **Compute.** Mitigation: benchmark first (M0), checkpoint and resume, reduce seeds for secondary
  horizons.
- **Phase C deadline** (≈ 6 weeks). Mitigation: freeze with the paper configuration if Phase B is
  late.
- **Free data drift** (Yahoo/Tiingo revisions) in 2027. Mitigation: snapshot the raw download each
  day and validate against a second source.
