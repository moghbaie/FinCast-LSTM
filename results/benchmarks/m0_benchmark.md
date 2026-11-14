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

## Still to do

- [ ] Fold the CUDA-graph training step into `fincast.training.trainer`. v1 above was measured
  with a stand-alone script; the masks must be resampled into static buffers on each replay.
- [ ] Re-run `scripts/benchmark_lstm.py` with v1 locally and on Colab (T4 / L4 / A100) to fill in
  the Colab rows.
