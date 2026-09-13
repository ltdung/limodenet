"""
limodenet.family
=================
The LIMODENet classifier and its five-scale family (nano/tiny/small/base,
a pure width ladder, plus the off-ladder `big`). See Table 1 of the paper.

    from limodenet.family import build
    model = build("tiny", num_classes=10)          # 694,538 params

Presets are verified against the released checkpoints: `tiny` and `big`
match the published parameter counts exactly.
"""
import argparse

import torch
import torch.nn as nn

from .blocks import LargeKernelBlock, FocalBlock


class LIMODENet(nn.Module):
    """Configurable LIMODENet classifier. Defaults reproduce `tiny` (~0.69M)."""

    def __init__(self, dim=128, s1_blocks=1, s2_blocks=4, s3_blocks=1,
                 mlp_ratio=3.0, lk_kernel=5, alphas=(0.5, 0.7, 0.5),
                 in_ch=3, num_classes=10):
        super().__init__()
        a1, a2, a3 = alphas
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, stride=2, padding=1),
            nn.GroupNorm(1, dim), nn.GELU())
        self.stage1 = nn.Sequential(*[
            LargeKernelBlock(dim, mlp_ratio, a1, lk_kernel, "euler")
            for _ in range(s1_blocks)])
        self.down2 = nn.Conv2d(dim, dim, 2, stride=2, groups=dim)
        self.stage2 = nn.Sequential(*[
            FocalBlock(dim, mlp_ratio, a2) for _ in range(s2_blocks)])
        self.stage3 = nn.Sequential(*[
            LargeKernelBlock(dim, mlp_ratio, a3, lk_kernel, "rk2")
            for _ in range(s3_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


# Nano/tiny/small/base form a pure width ladder: identical shape (1/4/1
# blocks, mlp_ratio 3.0, kernel 5), only `dim` varies. `big` is deliberately
# off-ladder (also changes depth, mlp_ratio, kernel) and kept for continuity
# with the original published checkpoint.
PRESETS = {
    "nano":  dict(dim=96,  s1_blocks=1, s2_blocks=4, s3_blocks=1, mlp_ratio=3.0, lk_kernel=5),  #   394,954
    "tiny":  dict(dim=128, s1_blocks=1, s2_blocks=4, s3_blocks=1, mlp_ratio=3.0, lk_kernel=5),  #   694,538
    "small": dict(dim=192, s1_blocks=1, s2_blocks=4, s3_blocks=1, mlp_ratio=3.0, lk_kernel=5),  # 1,545,610
    "base":  dict(dim=256, s1_blocks=1, s2_blocks=4, s3_blocks=1, mlp_ratio=3.0, lk_kernel=5),  # 2,732,554
    "big":   dict(dim=256, s1_blocks=2, s2_blocks=6, s3_blocks=2, mlp_ratio=4.0, lk_kernel=7),  # 5,799,434 (off-ladder)
}


def build(preset=None, num_classes=10, **overrides):
    cfg = dict(PRESETS[preset]) if preset else {}
    cfg.update(overrides)
    cfg["num_classes"] = num_classes
    return LIMODENet(**cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, size=64):
    """Rough MAC count via a forward hook on Conv2d/Linear (no external deps)."""
    macs = [0]
    hs = []

    def conv_hook(m, i, o):
        macs[0] += m.in_channels // m.groups * m.out_channels \
            * o.shape[-1] * o.shape[-2] * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, i, o):
        macs[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hs:
        h.remove()
    return macs[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    print(f"{'preset':<8}{'params':>12}{'MMac':>10}   config")
    print("-" * 70)
    for name in PRESETS:
        m = build(name).eval()
        p = count_params(m)
        mac = count_macs(m, args.size) / 1e6
        print(f"{name:<8}{p:>12,}{mac:>10.1f}   {PRESETS[name]}")
