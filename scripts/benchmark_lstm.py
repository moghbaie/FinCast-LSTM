"""M0 throughput benchmark for the paper LSTM, using the production trainer code path.

Times steady-state training steps on synthetic data shaped like one study period
(~204k training sequences x 240 steps x 1 feature, ~51k validation) and extrapolates
seconds per epoch. Runs locally and on Colab. Writes results/benchmarks/<label>.json.

Usage: python scripts/benchmark_lstm.py --label local_gtx1650_v1 [--batches 32 512]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from fincast.models.lstm import CudnnLSTM, PaperLSTM
from fincast.training.trainer import TrainConfig, _make_optimizer, _recapture, evaluate_loss
from fincast.utils.provenance import environment_info, set_seed

N_TRAIN = 204_000
N_VAL = 51_000
SEQ = 240


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def time_steps(model: nn.Module, batch: int, steps: int, dev: torch.device, graph: bool) -> float:
    """Seconds per training step in steady state."""
    cfg = TrainConfig(batch_size=batch)
    opt = _make_optimizer(model, cfg, capturable=graph)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(batch, SEQ, 1, device=dev)
    y = torch.randint(0, 2, (batch,), device=dev)
    model.train()
    if graph:
        snap_m = {k: v.clone() for k, v in model.state_dict().items()}
        step = _recapture(model, opt, loss_fn, (batch, SEQ, 1), dev, snap_m, opt.state_dict())

        def run() -> None:
            step(x, y)
    else:

        def run() -> None:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    for _ in range(3):
        run()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        run()
    _sync(dev)
    return (time.perf_counter() - t0) / steps


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
    graph = dev.type == "cuda"
    xv = torch.randn(N_VAL, SEQ, 1, device=dev)
    yv = torch.randint(0, 2, (N_VAL,), device=dev)

    rows = []
    for name, ctor in [("paper_variational", PaperLSTM), ("cudnn_input_dropout", CudnnLSTM)]:
        for b in args.batches:
            if name == "cudnn_input_dropout" and b != 512:
                continue
            model = ctor().to(dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sps = time_steps(model, b, args.steps, dev, graph=graph and name == "paper_variational")
            _sync(dev)
            t0 = time.perf_counter()
            evaluate_loss(model, xv, yv, 2048)
            _sync(dev)
            row = {
                "model": name,
                "batch": b,
                "cuda_graph": graph and name == "paper_variational",
                "sec_per_step": round(sps, 5),
                "sec_per_epoch_train": round(sps * math.ceil(N_TRAIN / b), 1),
                "sec_val_pass": round(time.perf_counter() - t0, 2),
                "peak_mem_mb": (
                    round(torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else None
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

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
    path = out / f"{args.label}.json"
    path.write_text(json.dumps(payload, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
