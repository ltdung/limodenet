"""
limodenet.baselines
====================
The iso-parameter comparison restorers of Table `tab:modern`, defined exactly
as they were trained (parameter names match the released baseline
checkpoints, so `load_state_dict` needs no key remapping).

    from limodenet.baselines import build_baseline
    model = build_baseline("unet")        # cnn | unet | dncnn | nafnet | restormer

Tiny-band shapes (~0.75-0.77M params, matched to LIMODENet-AE's 0.77M):

    cnn        plain conv encoder + LIMODENet's own decoder (bottleneck peer)
    unet       skip-connection U-Net (non-bottleneck reference)
    dncnn      DnCNN residual denoiser, full resolution, depth 22
    nafnet     NAFNet-lite: LayerNorm2d + SimpleGate + channel attention
    restormer  Restormer-lite: LayerNorm2d + softmax channel attention (MDTA)

NAFNet-lite and Restormer-lite are the unconstrained modern restorers: they
win on fidelity but contain operations with no spiking-legal equivalent (see
`demo/audit.py`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .recon import make_decoder


# ---------------- CNN-AE: plain conv encoder, same decoder as LIMODENet-AE
class ConvBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c = nn.Conv2d(d, d, 3, padding=1)
        self.n = nn.GroupNorm(1, d)
        self.a = nn.GELU()

    def forward(self, x):
        return self.a(self.n(self.c(x)))


class CNNRecon(nn.Module):
    """Plain conv encoder, 128 -> 64 -> 32, then n_blocks at 32x32."""
    def __init__(self, dim=128, n_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, dim, 3, 2, 1), nn.GroupNorm(1, dim), nn.GELU())
        self.down = nn.Conv2d(dim, dim, 2, stride=2)
        self.blocks = nn.Sequential(*[ConvBlock(dim) for _ in range(n_blocks)])
        self.decoder = make_decoder(dim)

    def forward(self, x):
        return self.decoder(self.blocks(self.down(self.stem(x))))


# ---------------- U-Net: skip-connection reference
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(1, co), nn.GELU(),
                                 nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(1, co), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNetRecon(nn.Module):
    def __init__(self, base=20):
        super().__init__()
        C = base
        self.e1 = DoubleConv(3, C)
        self.e2 = DoubleConv(C, 2 * C)
        self.e3 = DoubleConv(2 * C, 4 * C)
        self.bott = DoubleConv(4 * C, 8 * C)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(8 * C, 4 * C, 2, stride=2)
        self.d3 = DoubleConv(8 * C, 4 * C)
        self.up2 = nn.ConvTranspose2d(4 * C, 2 * C, 2, stride=2)
        self.d2 = DoubleConv(4 * C, 2 * C)
        self.up1 = nn.ConvTranspose2d(2 * C, C, 2, stride=2)
        self.d1 = DoubleConv(2 * C, C)
        self.outc = nn.Sequential(nn.Conv2d(C, 3, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        xb = self.bott(self.pool(x3))
        y = self.d3(torch.cat([self.up3(xb), x3], 1))
        y = self.d2(torch.cat([self.up2(y), x2], 1))
        y = self.d1(torch.cat([self.up1(y), x1], 1))
        return self.outc(y)


# ---------------- DnCNN: residual denoiser, no down-sampling
class DnCNNRecon(nn.Module):
    def __init__(self, depth=22, ch=64):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                       nn.GroupNorm(1, ch), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(x - self.net(x), 0.0, 1.0)


# ---------------- NAFNet-lite (Chen et al., ECCV 2022), scaled to the band
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over (N,C,H,W)."""
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.g + self.b


class SimpleGate(nn.Module):
    """Split channels in half and multiply: a product of two dense streams."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dwc = c * dw_expand
        self.conv1 = nn.Conv2d(c, dwc, 1)
        self.conv2 = nn.Conv2d(dwc, dwc, 3, padding=1, groups=dwc)
        self.conv3 = nn.Conv2d(dwc // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dwc // 2, dwc // 2, 1))
        self.sg = SimpleGate()
        fc = c * ffn_expand
        self.conv4 = nn.Conv2d(c, fc, 1)
        self.conv5 = nn.Conv2d(fc // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.sg(self.conv2(self.conv1(x)))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNetLite(nn.Module):
    def __init__(self, width=26, enc_blks=(1, 1, 1), middle_blk=1, dec_blks=(1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        chan = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk)])
        for n in dec_blks:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))

    def forward(self, x):
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = dec(up(x) + skip)
        return torch.clamp(self.ending(x), 0.0, 1.0)


# ---------------- Restormer-lite (Zamir et al., CVPR 2022), scaled to the band
class MDTA(nn.Module):
    """Softmax self-attention over the channel dimension. The softmax and the
    q@k^T product are functional ops inside forward(), not child modules."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        hd = c // self.heads
        q = F.normalize(q.reshape(b, self.heads, hd, h * w), dim=-1)
        k = F.normalize(k.reshape(b, self.heads, hd, h * w), dim=-1)
        v = v.reshape(b, self.heads, hd, h * w)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.proj((attn @ v).reshape(b, c, h, w))


class GDFN(nn.Module):
    """Gated-Dconv feed-forward: GELU(x1) * x2."""
    def __init__(self, dim, expand=2.66):
        super().__init__()
        hid = int(dim * expand)
        self.proj_in = nn.Conv2d(dim, hid * 2, 1)
        self.dw = nn.Conv2d(hid * 2, hid * 2, 3, padding=1, groups=hid * 2)
        self.proj_out = nn.Conv2d(hid, dim, 1)

    def forward(self, x):
        x1, x2 = self.dw(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


class RestormerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_expand=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expand)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class RestormerLite(nn.Module):
    def __init__(self, width=43, heads=(1, 2, 4), ffn_expand=2.66):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        chan = width
        for hh in heads[:-1]:
            self.encoders.append(RestormerBlock(chan, hh, ffn_expand))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle = RestormerBlock(chan, heads[-1], ffn_expand)
        for hh in reversed(heads[:-1]):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(RestormerBlock(chan, hh, ffn_expand))

    def forward(self, x):
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = dec(up(x) + skip)
        return torch.clamp(self.ending(x), 0.0, 1.0)


# Tiny-band (headline) shapes, from the training code's iso-parameter bands.
BASELINES = {
    "cnn":       (CNNRecon, dict(dim=128, n_blocks=4)),
    "unet":      (UNetRecon, dict(base=20)),
    "dncnn":     (DnCNNRecon, dict(depth=22, ch=64)),
    "nafnet":    (NAFNetLite, dict(width=26)),
    "restormer": (RestormerLite, dict(width=43)),
}


def build_baseline(name, **overrides):
    cls, cfg = BASELINES[name]
    return cls(**dict(cfg, **overrides))
