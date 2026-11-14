"""Unit tests for the paper LSTM and the trainer (CPU; GPU tests carry the ``gpu`` marker)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fincast.models.lstm import PaperLSTM
from fincast.training.trainer import TrainConfig, evaluate_loss, fit, predict_proba
from fincast.utils.provenance import set_seed


def _reversal_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic sequences whose label is a (noisy) reversal of the last 5 returns."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 240, 1, generator=g)
    score = -x[:, -5:, 0].sum(1) + 0.5 * torch.randn(n, generator=g)
    y = (score >= score.median()).long()
    return x, y


def test_param_count_matches_paper() -> None:
    assert PaperLSTM().n_trainable() == 2752  # paper p. 11


def test_loop_without_masks_equals_cudnn_path() -> None:
    set_seed(1)
    m = PaperLSTM().double()
    x = torch.randn(7, 240, 1, dtype=torch.float64)
    ones_x = torch.ones(7, 4, 1, dtype=torch.float64)
    ones_h = torch.ones(7, 4, 25, dtype=torch.float64)
    m.eval()
    ref = m(x)
    loop = m.loop_forward(x, ones_x, ones_h)
    torch.testing.assert_close(loop, ref, rtol=1e-10, atol=1e-10)


def test_dropout_mask_is_per_sequence_and_scaled() -> None:
    m = PaperLSTM()
    mx, mh = m._masks(10_000, torch.zeros(1))
    assert mx.shape == (10_000, 4, 1) and mh.shape == (10_000, 4, 25)
    keep = 1 - 0.16
    nonzero = mh[mh > 0]
    torch.testing.assert_close(nonzero, torch.full_like(nonzero, 1 / keep))
    assert abs(float((mh > 0).float().mean()) - keep) < 0.01


def test_forget_bias_init() -> None:
    b = PaperLSTM().lstm.bias_ih_l0.detach()
    assert torch.all(b[25:50] == 1.0) and torch.all(b[:25] == 0) and torch.all(b[50:] == 0)


def _small_cfg(**kw: object) -> TrainConfig:
    base: dict[str, object] = dict(
        batch_size=64, max_epochs=4, patience=10, verbose=False, use_cuda_graph=False
    )
    base.update(kw)
    return TrainConfig(**base)  # type: ignore[arg-type]


def test_training_is_deterministic_on_cpu() -> None:
    x, y = _reversal_data(512, 0)
    xv, yv = _reversal_data(128, 1)
    outs = []
    for _ in range(2):
        set_seed(3)
        m = PaperLSTM()
        fit(m, x, y, xv, yv, _small_cfg(max_epochs=2))
        outs.append(predict_proba(m, xv))
    assert (outs[0] == outs[1]).all()


def test_resume_from_checkpoint_matches_uninterrupted(tmp_path: Path) -> None:
    x, y = _reversal_data(512, 0)
    xv, yv = _reversal_data(128, 1)

    set_seed(5)
    m_full = PaperLSTM()
    r_full = fit(m_full, x, y, xv, yv, _small_cfg(checkpoint_dir=str(tmp_path / "a")))

    set_seed(5)
    m1 = PaperLSTM()
    fit(m1, x, y, xv, yv, _small_cfg(max_epochs=2, checkpoint_dir=str(tmp_path / "b")))
    set_seed(999)  # different global state: resume must restore everything it needs
    m2 = PaperLSTM()
    r_res = fit(m2, x, y, xv, yv, _small_cfg(checkpoint_dir=str(tmp_path / "b")))

    assert [h["val_loss"] for h in r_res.history] == [h["val_loss"] for h in r_full.history]
    for a, b in zip(m_full.state_dict().values(), m2.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_early_stopping_restores_best_weights() -> None:
    x, y = _reversal_data(256, 0)
    xv, yv = _reversal_data(64, 1)
    set_seed(0)
    m = PaperLSTM()
    r = fit(m, x, y, xv, yv, _small_cfg(max_epochs=30, patience=2, lr=5e-2))
    loss, _ = evaluate_loss(m, xv, yv, 1024)
    assert loss == pytest.approx(r.best_val_loss, rel=1e-6)
    assert r.best_val_loss == min(h["val_loss"] for h in r.history)
    assert r.stopped_early


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_training_learns_reversal_signal() -> None:
    dev = torch.device("cuda")
    x, y = _reversal_data(8192, 0)
    xv, yv = _reversal_data(2048, 1)
    set_seed(0)
    m = PaperLSTM().to(dev)
    cfg = TrainConfig(batch_size=512, max_epochs=15, patience=5, verbose=False)
    fit(m, x.to(dev), y.to(dev), xv.to(dev), yv.to(dev), cfg)
    acc = float(((predict_proba(m, xv) >= 0.5) == yv.numpy()).mean())
    assert acc > 0.7
