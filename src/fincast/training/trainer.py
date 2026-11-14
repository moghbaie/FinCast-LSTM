"""Mini-batch trainer for the sequence models: RMSprop, early stopping, checkpoint/resume.

Paper settings [P] p. 10-11: cross-entropy loss, RMSprop, at most 1,000 epochs, patience 10 on
validation loss, restore the weights with the lowest validation loss.
Assumptions [A]: RMSprop lr 1e-3, rho 0.9, eps 1e-7 (keras 2 defaults); shuffle every epoch;
the final partial batch is used (keras behaviour). Batch size comes from the config (decision A5).

Speed: on CUDA the full training step (forward with fresh dropout masks, backward, optimizer
step) is captured once in a CUDA graph and replayed. RNG ops inside the graph draw new values
on every replay, so dropout masks differ between steps as in eager mode. The final partial
batch of each epoch runs eagerly.

Resumability: after every epoch a checkpoint (model, optimizer, best weights, history, RNG
states) is written atomically to ``checkpoint_dir``. Calling ``fit`` again with the same
directory continues from the last finished epoch. On Colab, point it at Google Drive.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class TrainConfig:
    batch_size: int = 512
    lr: float = 1e-3
    rho: float = 0.9
    eps: float = 1e-7
    max_epochs: int = 1000
    patience: int = 10
    seed: int = 0
    use_cuda_graph: bool = True
    checkpoint_dir: str | None = None
    max_steps_per_epoch: int | None = None  # benchmarking only
    eval_batch_size: int = 2048  # cuDNN eval uses ~0.5 MB/row; 8192 spills a 4 GB GPU
    verbose: bool = True


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    stopped_early: bool
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


def _make_optimizer(model: nn.Module, cfg: TrainConfig, capturable: bool) -> torch.optim.RMSprop:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.RMSprop(params, lr=cfg.lr, alpha=cfg.rho, eps=cfg.eps, capturable=capturable)


@torch.no_grad()
def evaluate_loss(model: nn.Module, x: Tensor, y: Tensor, batch: int) -> tuple[float, float]:
    """Mean cross-entropy and accuracy, eval mode (no dropout)."""
    model.eval()
    dev = next(model.parameters()).device
    tot_loss, tot_correct = 0.0, 0
    for i in range(0, x.shape[0], batch):
        xb = x[i : i + batch].to(dev, non_blocking=True)
        yb = y[i : i + batch].to(dev, non_blocking=True)
        logits = model(xb)
        tot_loss += float(nn.functional.cross_entropy(logits, yb, reduction="sum"))
        tot_correct += int((logits.argmax(1) == yb).sum())
    n = x.shape[0]
    return tot_loss / n, tot_correct / n


@torch.no_grad()
def predict_proba(model: nn.Module, x: Tensor, batch: int = 2048) -> np.ndarray:
    """P(class 1) for every row of x, eval mode."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, x.shape[0], batch):
        logits = model(x[i : i + batch].to(dev, non_blocking=True))
        out.append(torch.softmax(logits, 1)[:, 1].float().cpu())
    return torch.cat(out).numpy()


class _GraphStep:
    """A training step captured in a CUDA graph for a fixed batch size."""

    def __init__(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        x_shape: tuple[int, ...],
        dev: torch.device,
    ) -> None:
        self.x = torch.zeros(x_shape, device=dev)
        self.y = torch.zeros(x_shape[0], dtype=torch.long, device=dev)
        self.model, self.opt, self.loss_fn = model, opt, loss_fn

        def step() -> Tensor:
            opt.zero_grad(set_to_none=False)
            loss = loss_fn(model(self.x), self.y)
            loss.backward()
            opt.step()
            return loss.detach()

        # Warm-up on a side stream (required before capture). This performs real optimizer
        # updates, so the caller snapshots and restores model/optimizer state around it.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = step()

    def __call__(self, xb: Tensor, yb: Tensor) -> Tensor:
        self.x.copy_(xb, non_blocking=True)
        self.y.copy_(yb, non_blocking=True)
        self.graph.replay()
        return self.loss


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def fit(
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    cfg: TrainConfig,
) -> TrainResult:
    """Train with early stopping. Tensors may live on CPU or on the model's device."""
    dev = next(model.parameters()).device
    use_graph = cfg.use_cuda_graph and dev.type == "cuda"
    opt = _make_optimizer(model, cfg, capturable=use_graph)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg.seed)  # CPU generator for shuffling

    ckpt_path = Path(cfg.checkpoint_dir) / "checkpoint.pt" if cfg.checkpoint_dir else None
    start_epoch, best_loss, best_epoch, bad_epochs = 0, math.inf, -1, 0
    best_state: dict[str, Tensor] = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    elapsed_before = 0.0

    if ckpt_path is not None and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        best_state = ck["best_state"]
        start_epoch, best_loss = ck["epoch"] + 1, ck["best_loss"]
        best_epoch, bad_epochs = ck["best_epoch"], ck["bad_epochs"]
        history, elapsed_before = ck["history"], ck.get("seconds", 0.0)
        gen.set_state(ck["shuffle_rng"])
        torch.set_rng_state(ck["torch_rng"])
        if dev.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        if cfg.verbose:
            print(f"[fit] resumed from epoch {start_epoch} (best {best_loss:.5f} @ {best_epoch})")
        if ck.get("stopped_early") or start_epoch >= cfg.max_epochs:
            model.load_state_dict(best_state)
            return TrainResult(
                best_epoch, best_loss, start_epoch, ck["stopped_early"], history, elapsed_before
            )

    n = x_train.shape[0]
    bs = cfg.batch_size
    graph_step: _GraphStep | None = None
    if use_graph and n >= bs:
        snap_m = copy.deepcopy(model.state_dict())
        snap_o = copy.deepcopy(opt.state_dict())
        graph_step = _recapture(model, opt, loss_fn, (bs, *x_train.shape[1:]), dev, snap_m, snap_o)

    def eager_step(xb: Tensor, yb: Tensor) -> Tensor:
        model.train()
        opt.zero_grad(set_to_none=False)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        return loss.detach()

    t0 = time.perf_counter()
    stopped_early = False
    epoch = start_epoch
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        n_steps = math.ceil(n / bs)
        if cfg.max_steps_per_epoch is not None:
            n_steps = min(n_steps, cfg.max_steps_per_epoch)
        loss_sum = torch.zeros((), device=dev)
        seen = 0
        for s in range(n_steps):
            idx = perm[s * bs : (s + 1) * bs]
            xb = x_train[idx].to(dev, non_blocking=True)
            yb = y_train[idx].to(dev, non_blocking=True)
            if graph_step is not None and idx.numel() == bs:
                loss = graph_step(xb, yb)
            else:
                loss = eager_step(xb, yb)
            loss_sum += loss * idx.numel()
            seen += idx.numel()
        train_loss = float(loss_sum) / max(seen, 1)
        val_loss, val_acc = evaluate_loss(model, x_val, y_val, cfg.eval_batch_size)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )
        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if cfg.verbose:
            print(
                f"[fit] epoch {epoch:4d} train {train_loss:.5f} val {val_loss:.5f} "
                f"acc {val_acc:.4f} best@{best_epoch} ({time.perf_counter() - t0:.0f}s)"
            )
        finished = bad_epochs >= cfg.patience or epoch + 1 >= cfg.max_epochs
        stopped_early = bad_epochs >= cfg.patience
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "best_state": best_state,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "history": history,
                    "seconds": elapsed_before + time.perf_counter() - t0,
                    "shuffle_rng": gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                    "finished": finished,
                    "stopped_early": stopped_early,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
            (ckpt_path.parent / "history.json").write_text(json.dumps(history, indent=1))
        if finished:
            break

    with torch.no_grad():
        model.load_state_dict(best_state)
    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        epochs_run=epoch + 1,
        stopped_early=stopped_early,
        history=history,
        seconds=elapsed_before + time.perf_counter() - t0,
    )


def _recapture(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    x_shape: tuple[int, ...],
    dev: torch.device,
    snap_m: dict[str, Tensor],
    snap_o: dict[str, Any],
) -> _GraphStep:
    """Capture the graph, then restore model and optimizer state *in place*.

    Warm-up/capture mutate weights and optimizer buffers. The captured graph refers to the
    current tensor objects, so the snapshot must be copied into those same tensors.
    """
    g = _GraphStep(model, opt, loss_fn, x_shape, dev)
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(snap_m[k])
        saved_states = snap_o["state"]
        params = [p for grp in opt.param_groups for p in grp["params"]]
        for i, p in enumerate(params):
            st = opt.state[p]
            if i in saved_states:
                for key, val in saved_states[i].items():
                    if isinstance(val, Tensor) and key in st:
                        st[key].copy_(val.to(st[key].device))
            else:  # fresh optimizer: zero the buffers created during warm-up
                for val in st.values():
                    if isinstance(val, Tensor):
                        val.zero_()
    return g
