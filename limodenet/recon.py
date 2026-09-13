"""
limodenet.recon
================
LIMODENet's reconstruction variant: the classifier's stem/stage1/down2/
stage2/stage3 encoder, feeding a transposed-convolution decoder
(128 -> 64 -> 32 -> 3, Sigmoid) at 128x128 output.

    from limodenet.recon import build_recon
    model = build_recon("limodenet")          # the paper's LIMODENet-AE
    model = build_recon("limodenet_skip")      # + additive skip connections
    model = build_recon("limodenet_depth4")    # + a 4th stage, no skip

`limodenet_skip` is the kill-test cell that closes roughly half the fidelity
gap to NAFNet-lite/Restormer-lite while staying spiking-legal (Sec. 4); it
is also the encoder-decoder with the verified zero-blocked-operation spiking
port (see `limodenet/snn_recon.py`).
"""
import torch.nn as nn

from .blocks import DWConv


class PWMLP(nn.Module):
    """Pointwise MLP. NOTE: named `net` (a plain nn.Sequential), not
    `fc1`/`fc2` as in `limodenet.blocks.PWMLP` -- this matches the exact
    parameter names the released reconstructor checkpoints were saved
    under, so `load_state_dict` works with no key remapping."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, dim, 1))

    def forward(self, x):
        return self.net(x)


class LargeKernelBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, alpha, kernel=5, discret="euler"):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = DWConv(dim, kernel)
        self.act = nn.GELU()
        self.mlp = PWMLP(dim, int(dim * mlp_ratio))
        self.alpha, self.discret = alpha, discret

    def forward(self, x):
        h = self.norm(x)
        if self.discret == "euler":
            return x + self.alpha * self.mlp(self.act(self.dw(h)))
        k1 = self.mlp(self.act(self.dw(h)))
        mid = x + (self.alpha / 2) * k1
        return x + self.alpha * self.mlp(self.act(self.dw(self.norm(mid))))


class FocalBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, alpha, global_branch=True):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.dw = DWConv(dim, 3)
        self.global_fc = (nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
                           if global_branch else None)
        self.act = nn.GELU()
        self.mlp = PWMLP(dim, int(dim * mlp_ratio))
        self.alpha = alpha

    def forward(self, x):
        h = self.norm(x)
        loc = self.dw(h)
        if self.global_fc is not None:
            loc = loc + self.global_fc(loc)
        return x + self.alpha * self.mlp(self.act(loc))


def make_decoder(dim):
    """32x32 -> 128x128x3, the skip-free decoder used by plain LIMODENet-AE."""
    return nn.Sequential(
        nn.ConvTranspose2d(dim, dim // 2, 3, 2, 1, output_padding=1), nn.GroupNorm(1, dim // 2), nn.GELU(),
        nn.ConvTranspose2d(dim // 2, dim // 4, 3, 2, 1, output_padding=1), nn.GroupNorm(1, dim // 4), nn.GELU(),
        nn.Conv2d(dim // 4, 3, 3, padding=1), nn.Sigmoid())


class LIMODENetRecon(nn.Module):
    """The reconstruction encoder-decoder, with the kill-test's two
    composable axes: `skip` (additive, spiking-legal encoder->decoder skip
    connections) and `depth4` (a 4th encoder stage at constant width)."""

    def __init__(self, dim=128, mr=3.0, alphas=(0.5, 0.7, 0.5),
                 disc1="euler", disc3="rk2", global_branch=True, mr_focal=None,
                 skip=False, depth4=False):
        super().__init__()
        a1, a2, a3 = alphas
        mrf = mr if mr_focal is None else mr_focal
        self.stem = nn.Sequential(nn.Conv2d(3, dim, 3, 2, 1), nn.GroupNorm(1, dim), nn.GELU())
        self.stage1 = LargeKernelBlock(dim, mr, a1, 5, disc1)
        self.down2 = nn.Conv2d(dim, dim, 2, stride=2, groups=dim)
        self.stage2 = nn.Sequential(*[
            FocalBlock(dim, mrf, a2, global_branch=global_branch) for _ in range(4)])
        self.stage3 = LargeKernelBlock(dim, mr, a3, 5, disc3)

        self.depth4 = depth4
        if depth4:
            self.down3 = nn.Conv2d(dim, dim, 2, stride=2, groups=dim)               # 32 -> 16
            self.stage4 = LargeKernelBlock(dim, mr, 0.5, 5, "euler")                # extra block @16x16
            self.pre_up = nn.Sequential(nn.ConvTranspose2d(dim, dim, 3, 2, 1, output_padding=1),
                                         nn.GroupNorm(1, dim), nn.GELU())            # 16 -> 32

        self.skip = skip
        if skip:
            self.dec_up1 = nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 3, 2, 1, output_padding=1),
                                          nn.GroupNorm(1, dim // 2), nn.GELU())
            self.dec_up2 = nn.Sequential(nn.ConvTranspose2d(dim // 2, dim // 4, 3, 2, 1, output_padding=1),
                                          nn.GroupNorm(1, dim // 4), nn.GELU())
            self.dec_out = nn.Sequential(nn.Conv2d(dim // 4, 3, 3, padding=1), nn.Sigmoid())
            self.skip_proj1 = nn.Conv2d(dim, dim // 2, 1)
            self.skip_proj0 = nn.Conv2d(3, dim // 4, 1)
        else:
            self.decoder = make_decoder(dim)

    def forward(self, x):
        x0 = x
        h = self.stem(x)
        s1 = self.stage1(h)
        d = self.down2(s1)
        z = self.stage2(d)
        z = self.stage3(z)
        if self.depth4:
            z = self.down3(z)
            z = self.stage4(z)
            z = self.pre_up(z)
        if self.skip:
            u1 = self.dec_up1(z) + self.skip_proj1(s1)
            u2 = self.dec_up2(u1) + self.skip_proj0(x0)
            return self.dec_out(u2)
        return self.decoder(z)


# One entry per released kill-test cell (Table `tab:modern` in the paper).
VARIANTS = {
    "limodenet":            dict(),
    "limodenet_skip":       dict(skip=True),
    "limodenet_depth4":     dict(depth4=True),
    "limodenet_skip_depth4": dict(skip=True, depth4=True),
}


def build_recon(variant="limodenet", **overrides):
    cfg = dict(VARIANTS[variant])
    cfg.update(overrides)
    return LIMODENetRecon(**cfg)
