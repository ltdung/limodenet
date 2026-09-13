"""
limodenet.snn
=============
Spiking-network conversion of the LIMODENet classifier (snntorch, LIF
neurons, direct input coding). Every GELU becomes a leaky-integrate-and-fire
neuron; every convolution, normalization, and residual/skip addition stays
graded ("synaptic") -- the standard deep-SNN formulation, and the mapping
that runs on Akida/Loihi-2-class hardware.

    from limodenet.snn import SpikingLIMODENet, warm_start
    model = SpikingLIMODENet(num_classes=10, T=8)
    warm_start(model, "limodenet-tiny-eurosat-tinft.pth")   # load ANN weights
    logits = model(images)                                    # rate readout

Requires `snntorch` (`pip install snntorch`).
"""
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate, utils

from .blocks import DWConv


def make_lif(beta, spike_grad, thr=1.0):
    return snn.Leaky(beta=beta, threshold=thr, spike_grad=spike_grad,
                      init_hidden=True, learn_beta=True)


class SPWMLP(nn.Module):
    """fc1 -> LIF -> fc2 (names match the ANN PWMLP for warm-starting)."""
    def __init__(self, dim, hidden, beta, sg):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.lif = make_lif(beta, sg)
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        return self.fc2(self.lif(self.fc1(x)))


class SLargeKernelBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, alpha, beta, sg, kernel=5, discret="euler"):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = DWConv(dim, kernel)
        self.mlp = SPWMLP(dim, int(dim * mlp_ratio), beta, sg)
        self.lif = make_lif(beta, sg)              # replaces the block's GELU
        self.alpha, self.discret = alpha, discret

    def forward(self, x):
        h = self.norm(x)
        if self.discret == "euler":
            h = self.mlp(self.lif(self.dw(h)))
            return x + self.alpha * h
        k1 = self.mlp(self.lif(self.dw(h)))
        mid = x + (self.alpha / 2) * k1
        k2 = self.mlp(self.lif(self.dw(self.norm(mid))))
        return x + self.alpha * k2


class SFocalBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, alpha, beta, sg):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = DWConv(dim, 3)
        self.global_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
        self.mlp = SPWMLP(dim, int(dim * mlp_ratio), beta, sg)
        self.lif = make_lif(beta, sg)               # replaces the focal GELU
        self.alpha = alpha

    def forward(self, x):
        h = self.norm(x)
        loc = self.dw(h)
        glob = self.global_fc(loc)
        y = self.mlp(self.lif(loc + glob))
        return x + self.alpha * y


class SpikingLIMODENet(nn.Module):
    """Spiking LIMODENet classifier. `T` is the number of simulated
    timesteps (T=8 is the paper's operating point: 93.55+/-0.54% top-1 on
    EuroSAT, within 4.9pp of the ANN, at 2.44x lower SynOps energy)."""

    def __init__(self, num_classes=10, dim=128, mlp_ratio=3.0,
                 beta=0.9, T=8, slope=25.0):
        super().__init__()
        self.T = T
        sg = surrogate.fast_sigmoid(slope=slope)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1), nn.GroupNorm(1, dim))
        self.stem_lif = make_lif(beta, sg)
        self.stage1 = nn.Sequential(
            SLargeKernelBlock(dim, mlp_ratio, 0.5, beta, sg, 5, "euler"))
        self.down2 = nn.Conv2d(dim, dim, 2, stride=2, groups=dim)
        self.stage2 = nn.Sequential(
            *[SFocalBlock(dim, mlp_ratio, 0.7, beta, sg) for _ in range(4)])
        self.stage3 = nn.Sequential(
            SLargeKernelBlock(dim, mlp_ratio, 0.5, beta, sg, 5, "rk2"))
        self.head_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head_fc1 = nn.Linear(dim, dim)
        self.head_lif = make_lif(beta, sg)
        self.head_fc2 = nn.Linear(dim, num_classes)

    def _step(self, x):
        x = self.stem_lif(self.stem(x))
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head_pool(x)
        x = self.head_lif(self.head_fc1(x))
        return self.head_fc2(x)                    # graded logits this step

    def forward(self, x, T=None):
        T = T or self.T
        utils.reset(self)                          # clear all LIF membranes
        out = 0.0
        for _ in range(T):
            out = out + self._step(x)               # direct coding: same x each step
        return out / T                              # rate readout


def build_key_map():
    """ANN 'head.2.*'/'head.4.*' -> SNN 'head_fc1.*'/'head_fc2.*'; every
    other module (stem, stageN.M.{norm,dw,mlp.fc1/fc2,global_fc}) keeps its
    ANN name, so `load_state_dict(strict=False)` warm-starts everything but
    the LIF neurons directly."""
    return {"head.2.": "head_fc1.", "head.4.": "head_fc2."}


def warm_start(model, ckpt_path, verbose=True):
    """Load an ANN LIMODENet checkpoint's synaptic weights into a
    SpikingLIMODENet (or SpikingLIMODENet trained from scratch, if you
    prefer -- this is optional but matches the paper's `ANN init` row)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    kmap = build_key_map()
    remapped = {}
    for k, v in sd.items():
        nk = k
        for a, b in kmap.items():
            if k.startswith(a):
                nk = b + k[len(a):]
                break
        remapped[nk] = v
    tgt = model.state_dict()
    loaded = 0
    for k, v in remapped.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k] = v
            loaded += 1
    model.load_state_dict(tgt, strict=False)
    if verbose:
        print(f"[warm-start] matched {loaded}/{len(remapped)} tensors from {ckpt_path}")
    return model
