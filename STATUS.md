# STATUS / handoff (updated 2026-11-13)

## Done
- Read the paper. `paper_analysis.md` is complete (page refs follow the discussion-paper numbering).
- `plan.md` updated with the approved decisions A1–A18 (§0) and the revised milestone order.
- `CLAUDE.md` rewritten for this project (leakage rules plus the faithful-mode exceptions).
- M0 (partial):
  - Directory skeleton, `pyproject.toml`, `.venv` with torch 2.14.1+cu126.
  - `fincast.models.lstm.PaperLSTM`: variational per-gate dropout, keras-style init, 2,752
    trainable params; eval mode uses cuDNN.
  - `fincast.utils.provenance`: seeds and run manifest.
  - `scripts/benchmark_lstm.py`.
  - Benchmark results: `results/benchmarks/m0_benchmark.md`. With CUDA graphs + `unbind`, an
    epoch takes ≈ 25 s at batch 512 on the GTX 1650.

## Next (M0 remainder)
1. Trainer with CUDA-graph step, early stopping (patience 10, restore best), checkpoint/resume
   to a configurable dir (Google Drive on Colab).
2. Tests: param count 2,752; custom loop without masks == cuDNN output; seed determinism.
   Run ruff, mypy and pytest.
3. `requirements.txt` (pinned from `.venv`), `README.md`, `deviations.md`, Colab notebook
   `notebooks/00_colab_setup_benchmark.ipynb` (mount Drive, clone private repo, `pip install -e`,
   run benchmark).
4. Run the benchmark on Colab (needs the user to run the notebook).

## Next (M1: free-data audit)
- Membership: fja05680/sp500 GitHub (≈1996+); Wikipedia current list + changes table.
- Prices: yfinance (primary free source); Stooq cross-check; Tiingo needs the user's API key
  (ask). Measure the share of historical constituents with prices per year (survivorship gap).
- Ken French daily factors (MKT, RF, momentum, short-term reversal) for evaluation only.
- Write `data_audit.md` and `scripts/audit_data.py`; add a validation suite plus
  `pytest -m data`.

## Gate report owed to the user before M2
Actual GPU benchmark (local + Colab), data availability and coverage, and the remaining
reproduction limitations.
