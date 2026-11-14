# Deviations from Fischer & Krauss (2017 discussion paper)

Every departure from the paper, with the reason and which phase it applies to (decision A18).
**Type:** *forced* = the paper's resource is unavailable; *assumption* = the paper is silent or
ambiguous; *correction* = a fix for a leakage or realism problem, used only in `strict` mode and
Phases B and C; *engineering* = a different implementation that should not change results.

| ID | Area | Paper | Ours | Type | Phases | Status |
|---|---|---|---|---|---|---|
| D-01 | Paper version | EJOR 270(2) published version | 2017 FAU discussion paper (EJOR version not accessible). Numbers or details may differ from the published version | forced | all | open (O-1) |
| D-02 | Data vendor | Thomson Reuters constituent lists + total return indices | Free sources only (A3); see `data_audit.md`. **Approximate replication, not exact reproduction** | forced | A, B | M1 |
| D-03 | Delisted stocks | Included until last price | Only where a free source has their prices; coverage quantified in `data_audit.md` | forced | A, B | M1 |
| D-04 | Framework | keras 2 / TensorFlow (LSTM), H2O (DNN), sklearn | PyTorch (LSTM, GRU, DNN), sklearn, xgboost | engineering | all | done |
| D-05 | LSTM dropout | keras `dropout = recurrent_dropout = 0.16` | Own variational per-gate masks (keras `implementation=1` semantics), checked by unit tests | engineering | all | done |
| D-06 | LSTM bias | One bias per gate (keras) | `bias_hh` frozen at 0, so trainable params = 2,752 as in the paper | engineering | all | done |
| D-07 | Batch size | Not stated | 512 primary (A5); batch-32 comparison on a small experiment. Larger batches may change optimisation; reported | assumption | all | M3b |
| D-08 | Learning rate | Not stated | RMSprop lr 1e-3, rho 0.9, eps 1e-7 (keras 2 defaults) | assumption | all | done |
| D-09 | Validation split | 80/20, composition not stated | Last 20 % of training dates, purged by the label horizon (A6) | assumption | all | M2 |
| D-10 | Train/trade boundary | Last training labels use day 751, the first trading day | `faithful` reproduces this; `strict` purges it; reported separately (A7) | correction | A strict, B, C | M2 |
| D-11 | Execution | Close-to-close (trade at the signal close) | Paper-comparable case kept; next-session open execution is the realistic case (A10) | correction | B, C (A as sensitivity) | M4 |
| D-12 | Costs | 5 bps per half-turn, full turnover every day | Actual turnover at {2, 5, 10, 20} bps + short-borrow fee (A10); paper's rule reported in Phase A | correction | B, C | M4 |
| D-13 | DNN | H2O maxout network | PyTorch re-implementation; H2O defaults reproduced where documented | engineering | A, B | M3 |
| D-14 | Seeds | One run per period (implicit) | 1 seed for debugging, 5 for final H = 1, 3 for H = 5/20 (A12) | assumption | all | — |
| D-15 | Eval-mode kernel | keras | Fused cuDNN LSTM in eval mode (equal to our custom loop to 1e-10 in float64) | engineering | all | done |
| D-16 | Training speed-up | — | CUDA-graph capture of the training step (same maths) | engineering | all | done |
