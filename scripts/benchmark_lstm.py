"""M0 throughput benchmark for the paper LSTM (runs locally and on Colab).

Times training steps on synthetic data shaped like one study period (~204k training
sequences x 240 steps x 1 feature, ~51k validation) and extrapolates seconds per epoch.
Writes results/benchmarks/<label>.json.

Usage: python scripts/benchmark_lstm.py --label local_gtx1650 [--batches 32 128 512 2048]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from fincast.models.lstm import CudnnLSTM, PaperLSTM
from fincast.utils.provenance import environment_info, set_seed

N_TRAIN = 204_000
N_VAL = 51_000
SEQ = 240


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def time_train(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int, steps: int, dev: torch.device
) -> float:
    """Return seconds per training step (after warm-up)."""
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1e-7)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    n = x.shape[0]
    warm = 3
    for i in range(warm + steps):
        if i == warm:
            _sync(dev)
            t0 = time.perf_counter()
        idx = torch.randint(0, n, (batch,), device=dev)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
    _sync(dev)
    return (time.perf_counter() - t0) / steps


@torch.no_grad()
def time_eval(model: nn.Module, x: torch.Tensor, dev: torch.device, batch: int = 8192) -> float:
    model.eval()
    _sync(dev)
    t0 = time.perf_counter()
    for i in range(0, x.shape[0], batch):
        model(x[i : i + batch])
    _sync(dev)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 128, 512, 2048])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="results/benchmarks")
    args = ap.parse_args()

    set_seed(0, deterministic=False)
    dev = torch.device(args.device)
    x = torch.randn(N_TRAIN, SEQ, 1, device=dev)
    y = torch.randint(0, 2, (N_TRAIN,), device=dev)
    xv = torch.randn(N_VAL, SEQ, 1, device=dev)

    rows = []
    for name, ctor in [("paper_variational", PaperLSTM), ("cudnn_input_dropout", CudnnLSTM)]:
        for b in args.batches:
            if name == "cudnn_input_dropout" and b not in (32, 512):
                continue
            model = ctor().to(dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sps = time_train(model, x, y, b, args.steps, dev)
            steps_per_epoch = -(-N_TRAIN // b)
            row = {
                "model": name,
                "batch": b,
                "sec_per_step": round(sps, 5),
                "sec_per_epoch_train": round(sps * steps_per_epoch, 1),
                "sec_val_pass": round(time_eval(model, xv, dev), 2),
                "peak_mem_mb": (
                    round(torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else None
                ),
            }
            rows.append(row)
            print(json.dumps(row))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "seq_len": SEQ,
        "device": str(dev),
        "environment": environment_info(),
        "results": rows,
    }
    (out / f"{args.label}.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
