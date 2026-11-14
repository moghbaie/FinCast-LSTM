import time

import torch
from torch import nn

dev = torch.device("cuda")
H = 25
T = 240


class V(nn.Module):
    def __init__(s, mode):
        super().__init__()
        s.mode = mode
        s.wih = nn.Parameter(torch.randn(4 * H, 1) * 0.1)
        s.whh = nn.Parameter(torch.randn(4 * H, H) * 0.1)
        s.b = nn.Parameter(torch.zeros(4 * H))
        s.head = nn.Linear(H, 2)

    def forward(s, x, mx, mh):
        B = x.shape[0]
        if s.mode == "pergate":
            xw = torch.einsum(
                "btgf,ghf->btgh", x.unsqueeze(2) * mx.unsqueeze(1), s.wih.view(4, H, 1)
            ) + s.b.view(4, H)
            whh = s.whh.view(4, H, H)
            h = x.new_zeros(B, H)
            c = x.new_zeros(B, H)
            for xt in xw.transpose(0, 1).unbind(0):
                g = xt + torch.einsum("bgk,gjk->bgj", h.unsqueeze(1) * mh, whh)
                i, f, cc, o = g.unbind(1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(cc)
                h = torch.sigmoid(o) * torch.tanh(c)
        else:  # single mask, fused matmul
            xw = (x * mx[:, :1, :1].transpose(1, 2)) @ s.wih.t() + s.b  # (B,T,4H)
            h = x.new_zeros(B, H)
            c = x.new_zeros(B, H)
            m = mh[:, 0]
            for xt in xw.transpose(0, 1).unbind(0):
                g = torch.addmm(xt, h * m, s.whh.t())
                i, f, cc, o = g.chunk(4, 1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(cc)
                h = torch.sigmoid(o) * torch.tanh(c)
        return s.head(h)


def bench(mode, B, graph):
    m = V(mode).to(dev)
    opt = torch.optim.RMSprop(m.parameters(), lr=1e-3, capturable=graph)
    x = torch.randn(B, T, 1, device=dev)
    y = torch.randint(0, 2, (B,), device=dev)
    mx = torch.ones(B, 4, 1, device=dev)
    mh = torch.ones(B, 4, H, device=dev)
    lf = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=False)
        loss = lf(m(x, mx, mh), y)
        loss.backward()
        opt.step()
        return loss

    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        run = g.replay
    else:
        for _ in range(3):
            step()
        run = step
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 10


for mode in ("pergate", "single"):
    for graph in (True,):
        for B in (32, 128, 512, 2048):
            try:
                print(
                    mode,
                    "graph" if graph else "eager",
                    B,
                    round(bench(mode, B, graph), 4),
                    flush=True,
                )
            except Exception as e:
                print(mode, graph, B, "ERR", str(e)[:120])
