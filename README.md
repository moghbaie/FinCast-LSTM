# FinCast-LSTM

An independent PyTorch re-implementation and **approximate replication** of Fischer & Krauss
(2018), *Deep learning with long short-term memory networks for financial market predictions*,
EJOR 270(2). The project has four parts:

- **Phase A:** replicate the paper's 1992–2015 S&P 500 experiment as faithfully as free data allows.
- **Phase B:** walk-forward validation on 2016–2026.
- **Phase C:** a pre-registered prospective forecast for 2027.
- **Uncertainty extension:** ensembles, bootstrap and conformal intervals.

Research code. Nothing here is investment advice. Backtest returns are not investable returns.

## Documents

| File | Content |
|---|---|
| `plan.md` | Research plan, approved decisions (§0), milestones, acceptance criteria |
| `paper_analysis.md` | Method extraction with page references |
| `deviations.md` | Every departure from the paper |
| `data_audit.md` | Data sources, coverage, survivorship gaps, validation |
| `results/benchmarks/m0_benchmark.md` | GPU throughput benchmarks |
| `STATUS.md` | Current progress |
| `CLAUDE.md` | Leakage rules for anyone (human or AI) changing the pipeline |

## Setup

Local (Windows/Linux, CUDA 12.6):

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
uv pip install -e ".[data,models,viz,dev]"
cp env.secrets.example env.secrets   # then add TIINGO_API_KEY (git-ignored)
```

Google Colab: open `notebooks/00_colab_setup_benchmark.ipynb` and follow its first cell. It
needs the `GH_TOKEN` and `TIINGO_API_KEY` secrets. Data and checkpoints go to
`MyDrive/FinCast-LSTM/`.

## Checks

```bash
.venv/Scripts/ruff format && .venv/Scripts/ruff check && .venv/Scripts/mypy src
.venv/Scripts/python -m pytest -W error          # unit + leakage tests (CPU)
.venv/Scripts/python -m pytest -W error -m gpu   # GPU tests
.venv/Scripts/python -m pytest -m data           # real-data validation (after M1 download)
```

## Running

```bash
python scripts/benchmark_lstm.py --label <name>   # M0 GPU benchmark
python scripts/audit_data.py                      # M1 download + validation + coverage report
```

Later phases add `run_phase_a.py`, `run_phase_b.py` and the 2027 forecasting jobs (see `plan.md`).

## Data policy

Free sources only (no licensed data). Raw downloads stay in `data/` (git-ignored) with a manifest
of source, download time and checksums. Credentials stay in `env.secrets` (git-ignored) or Colab
Secrets.
