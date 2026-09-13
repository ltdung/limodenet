"""
limodenet.snn_recon
====================
End-to-end spiking conversion of LIMODENet-skip, the reconstruction
encoder-decoder. Converts with ZERO blocked operations (every GELU -> LIF,
every conv/norm/skip-add stays graded synaptic) -- unlike NAFNet-lite
(24 blocked ops: LayerNorm2d, SimpleGate) or Restormer-lite (22: LayerNorm2d,
softmax MDTA, gated feed-forward), which have no 1:1 spiking-legal
substitute for their non-portable operations (Sec. 4).

    from limodenet.snn_recon import SpikingLIMODENetSkipRecon, warm_start
    model = SpikingLIMODENetSkipRecon(T=8)
    warm_start(model, "limodenet-skip-ae-1db100q.pth")   # load the ANN encoder-decoder
    restored = model(degraded_images)                     # graded [0,1] image

Requires `snntorch` (`pip install snntorch`).
"""
import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate, utils

from .blocks import DWConv

DIM = 128


def make_lif(beta, sg, thr=1.0):
    return snn.Leaky(beta=beta, threshold=thr, spike_grad=sg,
                      init_hidden=True, learn_beta=True)


class SPWMLP(nn.Module):
    def __init__(self, dim, hidden, beta, sg):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.lif = make_lif(beta, sg)
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        return self.fc2(self.lif(self.fc1(x)))


class SLargeKernelBlock(nn.Module):
    def __init__(self, dim, mr, alpha, beta, sg, k=5, disc="euler"):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = DWConv(dim, k)
        self.mlp = SPWMLP(dim, int(dim * mr), beta, sg)
        self.lif = make_lif(beta, sg)
        self.alpha, self.disc = alpha, disc

    def forward(self, x):
        h = self.norm(x)
        if self.disc == "euler":
            h = self.mlp(self.lif(self.dw(h)))
            return x + self.alpha * h
        k1 = self.mlp(self.lif(self.dw(h)))
        mid = x + (self.alpha / 2) * k1
        k2 = self.mlp(self.lif(self.dw(self.norm(mid))))
        return x + self.alpha * k2


class SFocalBlock(nn.Module):
    def __init__(self, dim, mr, alpha, beta, sg):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = DWConv(dim, 3)
        self.global_fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
        self.mlp = SPWMLP(dim, int(dim * mr), beta, sg)
        self.lif = make_lif(beta, sg)
        self.alpha = alpha

    def forward(self, x):
        h = self.norm(x)
        loc = self.dw(h)
        glob = self.global_fc(loc)
        y = self.mlp(self.lif(loc + glob))
        return x + self.alpha * y


class SpikingLIMODENetSkipRecon(nn.Module):
    """Spiking LIMODENet-skip, encoder + decoder, fully converted. GELU
    sites inside the decoder upsample blocks -> LIF; skip additions and the
    final Sigmoid head stay graded (same "residual/output stays synaptic"
    rule as the classifier)."""

    def __init__(self, dim=DIM, mr=3.0, alphas=(0.5, 0.7, 0.5),
                 beta=0.9, T=8, slope=25.0):
        super().__init__()
        self.T = T
        sg = surrogate.fast_sigmoid(slope=slope)
        a1, a2, a3 = alphas

        # ---- encoder
        self.stem = nn.Sequential(nn.Conv2d(3, dim, 3, 2, 1), nn.GroupNorm(1, dim))
        self.stem_lif = make_lif(beta, sg)
        self.stage1 = SLargeKernelBlock(dim, mr, a1, beta, sg, 5, "euler")
        self.down2 = nn.Conv2d(dim, dim, 2, stride=2, groups=dim)
        self.stage2 = nn.Sequential(*[SFocalBlock(dim, mr, a2, beta, sg) for _ in range(4)])
        self.stage3 = SLargeKernelBlock(dim, mr, a3, beta, sg, 5, "rk2")

        # ---- decoder
        self.dec_up1_conv = nn.ConvTranspose2d(dim, dim // 2, 3, 2, 1, output_padding=1)
        self.dec_up1_norm = nn.GroupNorm(1, dim // 2)
        self.dec_up1_lif = make_lif(beta, sg)
        self.dec_up2_conv = nn.ConvTranspose2d(dim // 2, dim // 4, 3, 2, 1, output_padding=1)
        self.dec_up2_norm = nn.GroupNorm(1, dim // 4)
        self.dec_up2_lif = make_lif(beta, sg)
        self.dec_out = nn.Sequential(nn.Conv2d(dim // 4, 3, 3, padding=1), nn.Sigmoid())
        self.skip_proj1 = nn.Conv2d(dim, dim // 2, 1)
        self.skip_proj0 = nn.Conv2d(3, dim // 4, 1)

    def _step(self, x):
        x0 = x
        h = self.stem_lif(self.stem(x))
        s1 = self.stage1(h)
        d = self.down2(s1)
        z = self.stage2(d)
        z = self.stage3(z)
        u1 = self.dec_up1_lif(self.dec_up1_norm(self.dec_up1_conv(z))) + self.skip_proj1(s1)
        u2 = self.dec_up2_lif(self.dec_up2_norm(self.dec_up2_conv(u1))) + self.skip_proj0(x0)
        return self.dec_out(u2)                     # graded [0,1] image, this step

    def forward(self, x, T=None):
        T = T or self.T
        utils.reset(self)
        out = 0.0
        for _ in range(T):
            out = out + self._step(x)               # direct coding: same x each step
        return out / T                              # rate/analog readout


def build_key_map():
    """`limodenet.recon.LIMODENetRecon`'s PWMLP names its Sequential `net`
    (net.0=fc1, net.2=fc2); this maps every block's `.mlp.net.{0,2}` to
    `.mlp.{fc1,fc2}`, plus the decoder's numbered Sequential indices to
    their named equivalents here."""
    m = {
        "dec_up1.0.": "dec_up1_conv.",
        "dec_up1.1.": "dec_up1_norm.",
        "dec_up2.0.": "dec_up2_conv.",
        "dec_up2.1.": "dec_up2_norm.",
    }
    for blk in ["stage1", "stage2.0", "stage2.1", "stage2.2", "stage2.3", "stage3"]:
        m[f"{blk}.mlp.net.0."] = f"{blk}.mlp.fc1."
        m[f"{blk}.mlp.net.2."] = f"{blk}.mlp.fc2."
    return m


def warm_start(model, ckpt_path, verbose=True):
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
