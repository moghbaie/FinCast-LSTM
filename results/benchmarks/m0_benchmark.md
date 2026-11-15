# M0 LSTM throughput benchmark

Date 2026-11-13. Workload: one study period ≈ 204k training sequences × 240 steps × 1 feature
(synthetic data). One epoch is the training pass only; a validation pass with the fused cuDNN
kernel takes ≈ 0.2 s.

## Local: GTX 1650 Max-Q (4 GB), torch 2.14.1+cu126, cuDNN 9.10, Windows (WDDM)

| Implementation | Batch | s/step | s/epoch (extrapolated) |
|---|---|---|---|
| v0: paper variational dropout, TorchScript loop, `xw[:, t]` slicing | 32 | 1.065 | 6,790 |
| v0 | 512 | 0.938 | 374 |
| v0 | 2048 | 3.394 | 339 |
| cuDNN fused, input dropout only (not the paper's dropout) | 32 | 0.005 | 32 |
| cuDNN fused | 512 | 0.007 | 2.9 |
| v1: per-gate variational loop, `unbind` + CUDA graph + capturable RMSprop | 32 | 0.043 | ≈ 277 |
| v1 | 128 | 0.058 | ≈ 92 |
| v1 | 512 | 0.064 | **≈ 25** |
| v1 | 2048 | 0.131 | ≈ 13 |
| v1 with a single shared mask (keras `implementation=2` style) | 512 | 0.055 | ≈ 22 |

Findings:

- v0 was dominated by kernel-launch overhead (Windows WDDM) and by the backward pass of
  `xw[:, t]`. CUDA graphs and `unbind` fix both, about 15× faster at batch 512.
- `results/benchmarks/local_gtx1650.json` holds the v0 numbers. Its `peak_mem_mb` = 4410 is a
  reporting artifact; the measured peak was 220 MB at batch 32 and 460 MB at batch 512.
- Estimate per fit at batch 512, assuming 20–60 epochs to early stopping: **≈ 8–25 min**. Phase A
  has 23 periods; with 1 seed that is ≈ 3–10 h, with 5 seeds ≈ 15–50 h on this laptop.
- Batch 32 costs about 11× more per epoch (≈ 4.6 min/epoch). That is feasible for the small
  batch-32 comparison (A5) on 2–3 periods.

## v1 measured through the production trainer (`scripts/benchmark_lstm.py`, file `local_gtx1650_v1.json`)

| Implementation | Batch | s/step | s/epoch | Validation pass (51k) | Peak GPU memory |
|---|---|---|---|---|---|
| Paper LSTM, CUDA graph | 32 | 0.026 | **165** | 0.26 s | 1.1 GB |
| Paper LSTM, CUDA graph | 512 | 0.045 | **18** | 0.20 s | 1.1 GB |
| cuDNN input dropout only (comparator) | 512 | 0.007 | 2.9 | 0.21 s | 1.1 GB |

- The evaluation batch is now 2048: the cuDNN evaluation path uses about 0.5 MB per row, so a
  batch of 8192 spilled past the 4 GB card. That explains the earlier 4.2–4.4 GB readings.
- Estimate per LSTM fit, assuming 20–60 epochs to early stopping (not yet measured on real
  data): about 6–18 min at batch 512 and about 1–3 h at batch 32. The 23-period replication with
  1 seed would take about 2.5–7 GPU-hours on this laptop.

## Colab free tier: Tesla T4 (15 GB), Python 3.13, run 2026-11-15

Measured with `scripts/benchmark_lstm.py` from `notebooks/00_colab_setup_benchmark.ipynb` (label
`colab_tesla_t4_v1`, JSON kept in the user's Google Drive). Same workload as above.

| Implementation | Batch | s/step | s/epoch | Validation pass (51k) | Peak GPU memory |
|---|---|---|---|---|---|
| Paper LSTM, CUDA graph | 32 | 0.027 | **175** | 0.84 s | 0.37 GB |
| Paper LSTM, CUDA graph | 128 | 0.027 | 42.5 | 0.09 s | 0.38 GB |
| Paper LSTM, CUDA graph | 512 | 0.034 | **13.6** | 0.09 s | 0.40 GB |
| Paper LSTM, CUDA graph | 2048 | 0.078 | 7.8 | 0.10 s | 0.80 GB |
| cuDNN input dropout only (comparator) | 512 | 0.004 | 1.6 | 0.11 s | 0.42 GB |

- At batch 512 the T4 is only about 1.3× faster than the laptop (13.6 vs 18 s/epoch). The
  sequential 240-step loop is bound by kernel-launch latency, not GPU compute, so a bigger GPU
  (A100) would probably help little. Paid Colab is **not** justified by these numbers (A1).
- Estimate per fit at batch 512, assuming 20–60 epochs: about 4.5–14 min. Phase A with 23 periods
  ≈ 1.7–5 GPU-hours per seed; 5 seeds ≈ 9–26 h, which has to be split across free-tier sessions
  (resumable checkpoints).
- Batch 32 costs about the same per epoch on both GPUs (≈ 3 min), so the A5 comparison stays
  limited to 2–3 periods.
- Caveat: this run happened after the install had upgraded pandas to 3.1.1. That does not affect
  the torch benchmark, but the install cell now pins Colab's preinstalled versions.
