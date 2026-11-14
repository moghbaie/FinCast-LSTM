# CLAUDE.md — FinCast-LSTM

Research code: reproduction of Fischer & Krauss (2018, EJOR 270(2)) "Deep learning with LSTM
networks for financial market predictions", a modern (2016–2026) walk-forward validation, a
pre-registered 2027 prospective forecast, and an uncertainty-quantification extension.
Plan: `plan.md`. Paper facts: `paper_analysis.md`. Data facts: `data_audit.md`.

## Non-negotiable rules

### Rule 1: Evaluate with rolling-origin (walk-forward) backtesting, without leakage

- The time unit is the **trading day** of the US equity calendar (NYSE). Lead time L, horizon H,
  origin step S and window length W are all measured in trading days. Never hardcode calendar days
  or 252/250 outside `config/`.
- A day is complete only once its official close (16:00 America/New_York, 13:00 on half days) has
  printed. All timestamps are stored in America/New_York.
- At each origin T (end of trading day T, after the close):
  - Training: use only examples whose full target window ends at or before T. An example with
    feature time t has target window (t + L, t + L + H], so include it only if t ≤ T − L − H.
    (Phase A faithful mode uses L = 0, H = 1; see "Faithful-mode exceptions" below.)
  - Prediction: compute features as of T and predict the window (T + L, T + L + H].
- Data after T must never be used in training, feature fitting, scaling, imputation, universe
  selection, model selection, early stopping, or hyperparameter tuning for that fold. Tuning and
  early stopping use a nested, chronological split inside the training window (validation block at
  the end of the window, with an embargo of L + H days before it).
- Study periods: sliding training window W = 750 days, trading block S = 250 days (paper default;
  configurable). Report metrics per study period and aggregated across periods.

### Rule 2: Build features using only past data (point-in-time correctness)

- Every feature is computed as of T. "Lag k" means k trading days before T.
- Rolling and lag features look backward only (shift before rolling; no centered windows, no
  backfill, no forward-fill across a gap longer than the configured limit).
- Statistics fitted on data (mean/std for standardization, scalers, encoders, imputers) are fitted
  only on the training window of each study period and then applied to its trading window.
- Universe membership (S&P 500 constituents) is point-in-time: a stock is eligible at T only if it
  was a constituent per the membership list known at T. Never use today's constituent list for the
  past. Delisted stocks stay in the data until their last price.
- External data is joined on its availability timestamp, not its event timestamp.

| Field | Event time | Available at |
|---|---|---|
| Close / total-return index of day d | day d | close of day d |
| Open of day d | day d | 09:30 ET day d |
| Index membership change | effective date | effective date (announcement dates not reliably available; conservative) |
| Split / dividend adjustment factors | ex-date | ex-date; only **returns** may be used as features (back-adjusted price *levels* embed future events) |
| Fama–French factors, risk-free rate | day d | used for **evaluation only**, never as features |
| Calendar (weekday, holidays) | target time | always available |

Before changing the pipeline: check it against both rules. If it breaks them, explain where,
update `plan.md`, rewrite only the affected code, and keep the leakage tests passing.

## Faithful-mode exceptions (Phase A only, documented, never used in Phases B/C)

- The paper labels sample t with the return over (t, t+1] and trains on all sequences with t inside
  the 750-day training block, so the last training label uses the first trading-period day. Faithful
  mode reproduces this; `strict` mode (default everywhere else) applies the t ≤ T − L − H embargo.
  Both are reported in `reproduction_report.md`.
- The paper executes at the same close used to form the signal (close-to-close). Phases B/C also
  report next-open execution.

## How the rules are enforced in code

- `fincast.data.universe` builds point-in-time membership; `fincast.data.validate` checks prices.
- `fincast.features` builds as-of-T features; `fincast.backtest.splits` defines study periods and
  `check_split` asserts Rule 1 on every fold at run time.
- `tests/test_no_leakage.py` perturbs all data after T and requires features, training sets,
  fitted scalers and predictions to be unchanged; `tests/test_alignment.py` checks that targets
  are aligned to t+H and features to ≤ t.

## Other conventions

- Phase C: the 2027 protocol (`forecast_2027.md`, `config/phase_c.yaml`, code commit hash) is frozen
  before 2027-01-01 and never edited afterwards; deviations go in a dated addendum.
- Phases are kept separate: reproduction results are never tuned using modern or 2027 data;
  the uncertainty extension is reported separately from the reproduction.
- Raw data, runs and model outputs stay out of git (`data/`, `runs/`). Licensed data is never
  committed or redistributed.
- Fixed seeds; ≥ 5 seeds for every stochastic model; record package versions in every run dir.
- Report gross and net (after costs) returns; never present gross returns as investable.
  Report negative results and discrepancies with the paper without omission.
- Before finishing a milestone: `ruff format`, `ruff check`, `mypy src`, `pytest -W error`
  (plus `-m data` when data is present), then the milestone's validation/eval command.
- Do not commit or push unless the user asks.

## Commands (filled in as milestones land)

```bash
uv venv && uv pip install -r requirements.txt && uv pip install -e .
.venv/Scripts/python -m pytest -W error -m "not data"   # synthetic tests
.venv/Scripts/python -m pytest -m data                  # real-data tests
.venv/Scripts/python scripts/audit_data.py              # M1 data audit / validation
.venv/Scripts/python scripts/run_phase_a.py --config config/phase_a.yaml [--smoke]
```
