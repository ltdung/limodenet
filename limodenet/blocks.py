"""
limodenet.blocks
=================
Shared building blocks for LIMODENet: a depthwise conv, a pointwise MLP, and
the two residual-block types the backbone is built from.

Each block reads as one step of a numerical ODE integrator:
  LargeKernelBlock(discret="euler")  ->  forward-Euler step
  LargeKernelBlock(discret="rk2")    ->  two-stage Heun (RK-2) step
  FocalBlock                         ->  forward-Euler step with a
                                          mean-field (McKean-Vlasov) global
                                          coupling term added to the local one

See Sec. 3 of the paper for the derivation and Sec. 4 for what each choice
buys empirically.
"""
import torch.nn as nn


class DWConv(nn.Module):
    """Depthwise convolution."""
    def __init__(self, dim, kernel):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)

    def forward(self, x):
        return self.conv(x)


class PWMLP(nn.Module):
    """Pointwise (1x1-conv) two-layer MLP: fc1 -> GELU -> fc2."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LargeKernelBlock(nn.Module):
    """A local, depthwise-conv residual block, integrated as Euler or Heun.

    x <- x + alpha * f(GN(x)),  f = PWMLP(GELU(DWConv(.)))

    `discret="rk2"` instead applies a two-stage Heun update on the same
    composite field, halving local truncation error at double the cost
    (Thm. 1 in the paper: worth it only at the coarsest stage).
    """
    def __init__(self, dim, mlp_ratio, alpha, kernel=5, discret="euler"):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = DWConv(dim, kernel)
        self.mlp = PWMLP(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.alpha, self.discret = alpha, discret

    def forward(self, x):
        h = self.norm(x)
        if self.discret == "euler":
            h = self.mlp(self.act(self.dw(h)))
            return x + self.alpha * h
        k1 = self.mlp(self.act(self.dw(h)))
        mid = x + (self.alpha / 2) * k1
        k2 = self.mlp(self.act(self.dw(self.norm(mid))))
        return x + self.alpha * k2


class FocalBlock(nn.Module):
    """A global-context residual block: local depthwise branch + pooled
    global branch, added before the pointwise MLP.

    Reads as a discretized mean-field (McKean-Vlasov) ODE whose coupling
    term is the population mean of the local features (Prop. P6): a full
    global receptive field in one layer at O(N) mixing cost, versus O(N^2)
    for self-attention.
    """
    def __init__(self, dim, mlp_ratio, alpha):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = DWConv(dim, 3)
        self.global_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
        self.act = nn.GELU()
        self.mlp = PWMLP(dim, int(dim * mlp_ratio))
        self.alpha = alpha

    def forward(self, x):
        h = self.norm(x)
        loc = self.dw(h)
        glob = self.global_fc(loc)
        y = self.mlp(self.act(loc + glob))
        return x + self.alpha * y
