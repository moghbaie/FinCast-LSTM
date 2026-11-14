"""LSTM classifier following Fischer & Krauss (2017 discussion paper, p. 8-11).

Topology [P]: 240 timesteps x 1 feature -> LSTM(h=25, dropout=0.16 on inputs and recurrent
connections) -> Dense(2, softmax). 2,752 trainable parameters.

Implementation notes [A] (see deviations.md):
- The paper used keras 2 / TensorFlow. Keras' ``dropout``/``recurrent_dropout`` are variational
  (Gal & Ghahramani 2016): one Bernoulli mask per sequence, reused at every timestep, separate
  masks per gate (keras ``implementation=1``). We reproduce that in a custom recurrence.
- Keras uses one bias vector per gate; PyTorch ``nn.LSTM`` has two (``bias_ih``, ``bias_hh``).
  We keep ``bias_hh`` frozen at zero so the trainable count matches the paper (2,752).
- Initialisation follows keras defaults: Glorot-uniform input kernel, orthogonal recurrent
  kernel, zero bias except forget gate = 1 (``unit_forget_bias=True``).
- In eval mode (no dropout) the forward pass uses the fused cuDNN kernel with the same weights;
  this is numerically equivalent to the custom loop without masks (tested).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# PyTorch gate order in weight matrices: input, forget, cell (candidate), output.
_N_GATES = 4


def _variational_lstm_loop(
    xw: Tensor,  # (B, T, 4, H): per-gate input projections incl. bias, input masks applied
    w_hh: Tensor,  # (4, H, H)
    mask_h: Tensor,  # (B, 4, H) recurrent dropout masks (already scaled)
) -> Tensor:
    # Iterate with unbind over a time-major view: slicing xw[:, t] would make every backward
    # step write a full-size (B, T, 4, H) gradient (measured ~10x slower at batch 512).
    b, _, _, h = xw.shape
    h_t = torch.zeros(b, h, dtype=xw.dtype, device=xw.device)
    c_t = torch.zeros(b, h, dtype=xw.dtype, device=xw.device)
    for x_t in xw.transpose(0, 1).unbind(0):
        hm = h_t.unsqueeze(1) * mask_h  # (B, 4, H)
        rec = torch.einsum("bgk,gjk->bgj", hm, w_hh)  # (B, 4, H)
        g = x_t + rec
        i = torch.sigmoid(g[:, 0])
        f = torch.sigmoid(g[:, 1])
        cand = torch.tanh(g[:, 2])
        o = torch.sigmoid(g[:, 3])
        c_t = f * c_t + i * cand
        h_t = o * torch.tanh(c_t)
    return h_t


class PaperLSTM(nn.Module):
    """Single-layer LSTM + 2-way softmax head. Returns logits of shape (B, 2)."""

    def __init__(
        self,
        n_features: int = 1,
        hidden: int = 25,
        dropout: float = 0.16,
        recurrent_dropout: float = 0.16,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self._w("bias_hh_l0").requires_grad_(False)
        self.head = nn.Linear(hidden, n_classes)
        self.reset_parameters()

    def _w(self, name: str) -> Tensor:
        """Typed access to nn.LSTM parameters (weight_ih_l0, weight_hh_l0, bias_ih_l0, ...)."""
        t: Tensor = getattr(self.lstm, name)
        return t

    def reset_parameters(self) -> None:
        h = self.hidden
        with torch.no_grad():
            # Keras kernel shape (in, 4h): glorot over fan_in=in, fan_out=4h.
            limit = math.sqrt(6.0 / (self.n_features + _N_GATES * h))
            self._w("weight_ih_l0").uniform_(-limit, limit)
            # Keras recurrent kernel (h, 4h) orthogonal.
            w = torch.empty(h, _N_GATES * h)
            nn.init.orthogonal_(w)
            self._w("weight_hh_l0").copy_(w.t())
            self._w("bias_ih_l0").zero_()
            self._w("bias_ih_l0")[h : 2 * h] = 1.0  # forget gate
            self._w("bias_hh_l0").zero_()
            lim = math.sqrt(6.0 / (h + self.head.out_features))
            self.head.weight.uniform_(-lim, lim)
            self.head.bias.zero_()

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _masks(self, b: int, ref: Tensor) -> tuple[Tensor, Tensor]:
        def bern(shape: tuple[int, ...], p: float) -> Tensor:
            if p <= 0.0:
                return torch.ones(shape, dtype=ref.dtype, device=ref.device)
            keep = torch.full(shape, 1.0 - p, dtype=ref.dtype, device=ref.device)
            return torch.bernoulli(keep) / (1.0 - p)

        return (
            bern((b, _N_GATES, self.n_features), self.dropout),
            bern((b, _N_GATES, self.hidden), self.recurrent_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, n_features) -> logits (B, n_classes)."""
        use_masks = self.training and (self.dropout > 0 or self.recurrent_dropout > 0)
        if not use_masks:
            out, _ = self.lstm(x)
            return self.head(out[:, -1])
        mask_x, mask_h = self._masks(x.shape[0], x)
        return self.loop_forward(x, mask_x, mask_h)

    def loop_forward(self, x: Tensor, mask_x: Tensor, mask_h: Tensor) -> Tensor:
        """Custom recurrence with given masks (B,4,F) and (B,4,H). Ones = no dropout."""
        f = x.shape[2]
        h = self.hidden
        w_ih = self._w("weight_ih_l0").view(_N_GATES, h, f)
        bias = (self._w("bias_ih_l0") + self._w("bias_hh_l0")).view(_N_GATES, h)
        # per-gate masked inputs: (B,T,4,F) x (4,H,F) -> (B,T,4,H)
        xm = x.unsqueeze(2) * mask_x.unsqueeze(1)
        xw = torch.einsum("btgf,ghf->btgh", xm, w_ih) + bias
        w_hh = self._w("weight_hh_l0").view(_N_GATES, h, h)
        h_last = _variational_lstm_loop(xw, w_hh, mask_h)
        return self.head(h_last)


class CudnnLSTM(nn.Module):
    """Speed comparator: fused cuDNN LSTM with input dropout only (NOT the paper's dropout)."""

    def __init__(self, hidden: int = 25, dropout: float = 0.16) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(self.drop(x))
        return self.head(out[:, -1])
