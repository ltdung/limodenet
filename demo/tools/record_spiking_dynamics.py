#!/usr/bin/env python
"""
Record the real spiking dynamics of the trained spiking LIMODENet-skip
reconstructor (T=8) on the demo samples, for `demo/spiking_dynamics.html`.

Every LIF population is hooked. For each sample and each timestep we store:
  * the rate-coded output so far (running mean of the per-step images),
  * the spike rate of every LIF population,
  * a spatial spike map per population (fraction of channels spiking per pixel),
  * spikes and membrane potentials of the neurons at a 3x3 grid of probe pixels.

Nothing is simulated for the page: it replays these recordings.

    python demo/tools/record_spiking_dynamics.py            # weights from ./weights
    LIMODENET_WEIGHTS=weights_staging python demo/tools/record_spiking_dynamics.py

Requires snntorch.
"""
import base64
import glob
import io
import json
import math
import os
import sys

import numpy as np
import torch
import snntorch as snn
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from limodenet.recon import build_recon                      # noqa: E402
from limodenet.snn_recon import SpikingLIMODENetSkipRecon    # noqa: E402

WEIGHTS = os.environ.get("LIMODENET_WEIGHTS", os.path.join(ROOT, "weights"))
SAMPLES = os.path.join(ROOT, "demo", "samples")
OUT = os.path.join(ROOT, "demo", "spiking_dynamics_data.js")
T = 8
PROBE_FRACS = (0.25, 0.5, 0.75)        # 3x3 probe grid, as fractions of H and W
MAX_PROBE_CH = 64                      # channels kept per probe (raster rows)
MAP_LEVELS = 16                        # spike-map quantization levels
MEM_SCALE = 32.0                       # membrane quantization: int8 = round(mem * 32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Call order of the LIF populations inside one timestep (`_step`). Stage 3 is
# a Heun/RK-2 step: its two field evaluations reuse ONE LIF population, so
# that population integrates twice per timestep.
POPULATIONS = [
    ("stem_lif",          "Stem",                 "stem"),
    ("stage1.lif",        "S1 Euler: field",      "s1"),
    ("stage1.mlp.lif",    "S1 Euler: MLP hidden", "s1"),
] + [
    p for i in range(4) for p in (
        (f"stage2.{i}.lif",     f"S2 Focal {i + 1}: field",      "s2"),
        (f"stage2.{i}.mlp.lif", f"S2 Focal {i + 1}: MLP hidden", "s2"))
] + [
    ("stage3.lif",        "S3 Heun k1: field",      "s3"),
    ("stage3.mlp.lif",    "S3 Heun k1: MLP hidden", "s3"),
    ("stage3.lif",        "S3 Heun k2: field",      "s3"),
    ("stage3.mlp.lif",    "S3 Heun k2: MLP hidden", "s3"),
    ("dec_up1_lif",       "Decoder up 1",           "dec"),
    ("dec_up2_lif",       "Decoder up 2",           "dec"),
]

TF = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


def png_b64(arr):
    """uint8 HxW or HxWx3 -> base64 PNG (lossless: restorations are never
    shown through a lossy codec)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def file_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def to_uint8_img(t):
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def psnr(a, b):
    return 10 * math.log10(1.0 / torch.mean((a - b) ** 2).item())


class Recorder:
    """Forward hooks on every snn.Leaky; with init_hidden=True the hook's
    output is the spike tensor and `module.mem` the membrane after update."""
    def __init__(self, model):
        self.calls = []
        for name, mod in model.named_modules():
            if isinstance(mod, snn.Leaky):
                mod.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(mod, inp, out):
            spk = out[0] if isinstance(out, tuple) else out
            self.calls.append((name, spk.detach()[0].float(), mod.mem.detach()[0].float()))
        return fn


@torch.no_grad()
def record_sample(snn_model, ann, stem):
    deg_p = os.path.join(SAMPLES, f"{stem}_degraded.png")
    cln_p = os.path.join(SAMPLES, f"{stem}_clean.jpg")
    x = TF(Image.open(deg_p).convert("RGB")).unsqueeze(0).to(DEVICE)
    clean = TF(Image.open(cln_p).convert("RGB")).to(DEVICE)

    ann_out = ann(x)[0].clamp(0, 1)
    rec = Recorder.instance
    snntorch_reset(snn_model)
    cum = torch.zeros_like(x)
    steps = []
    for t in range(T):
        rec.calls = []
        cum += snn_model._step(x)
        out_t = (cum / (t + 1))[0]
        assert [c[0] for c in rec.calls] == [p[0] for p in POPULATIONS], \
            "LIF call order changed; update POPULATIONS"
        pops = []
        for (name, spk, mem) in rec.calls:
            C, H, W = spk.shape
            # fraction of channels spiking, quantized to MAP_LEVELS steps (PNG
            # then compresses ~2x better; invisible at display size)
            frac = spk.mean(0).cpu().numpy()
            fmap = (np.round(frac * (MAP_LEVELS - 1)) * (255 // (MAP_LEVELS - 1))).astype(np.uint8)
            probes = []
            for fy in PROBE_FRACS:
                for fx in PROBE_FRACS:
                    yy, xx = int(fy * H), int(fx * W)
                    s = spk[:MAX_PROBE_CH, yy, xx].cpu().numpy().astype(np.uint8)
                    m = (mem[:MAX_PROBE_CH, yy, xx].cpu().numpy() * MEM_SCALE).round()
                    m = np.clip(m, -127, 127).astype(np.int8)
                    probes.append(dict(
                        spk=base64.b64encode(np.packbits(s).tobytes()).decode(),
                        mem=base64.b64encode(m.tobytes()).decode()))
            pops.append(dict(rate=round(spk.mean().item(), 5), map=png_b64(fmap),
                             probes=probes))
        steps.append(dict(out=png_b64(to_uint8_img(out_t)),
                          psnr=round(psnr(out_t, clean), 3), pops=pops))

    # the step-by-step replay must equal the model's own forward pass
    full = snn_model(x)[0]
    assert torch.allclose(full, cum[0] / T, atol=1e-5), "replay != forward()"

    return dict(stem=stem, cls=stem.rsplit("_", 1)[0],
                degraded=file_b64(deg_p), clean=file_b64(cln_p),
                ann=png_b64(to_uint8_img(ann_out)),
                ann_psnr=round(psnr(ann_out, clean), 3),
                input_psnr=round(psnr(x[0], clean), 3),
                steps=steps)


def snntorch_reset(model):
    from snntorch import utils
    utils.reset(model)


def main():
    snn_model = SpikingLIMODENetSkipRecon(T=T).to(DEVICE).eval()
    snn_model.load_state_dict(torch.load(
        os.path.join(WEIGHTS, "limodenet-spiking-skip-recon-t8.pth"), map_location=DEVICE))
    ann = build_recon("limodenet_skip").to(DEVICE).eval()
    ann.load_state_dict(torch.load(
        os.path.join(WEIGHTS, "limodenet-skip-ae-1db100q.pth"), map_location=DEVICE))
    Recorder.instance = Recorder(snn_model)

    stems = sorted(os.path.basename(p)[:-len("_degraded.png")]
                   for p in glob.glob(os.path.join(SAMPLES, "*_degraded.png")))
    # one sample per class (the first of each class's seeded draw), for size
    by_cls = {}
    for s in stems:
        by_cls.setdefault(s.rsplit("_", 1)[0], s)
    chosen = list(by_cls.values())

    # population metadata: shapes, learned leak, threshold
    lifs = dict((n, m) for n, m in snn_model.named_modules() if isinstance(m, snn.Leaky))
    Recorder.instance.calls = []
    snntorch_reset(snn_model)
    with torch.no_grad():
        snn_model._step(torch.zeros(1, 3, 128, 128, device=DEVICE))
    shapes = [list(c[1].shape) for c in Recorder.instance.calls]
    meta = []
    for (name, label, stage), shp in zip(POPULATIONS, shapes):
        beta = lifs[name].beta
        beta = float(beta.mean()) if torch.is_tensor(beta) else float(beta)
        meta.append(dict(module=name, label=label, stage=stage, shape=shp,
                         beta=round(beta, 4), threshold=float(lifs[name].threshold)))

    samples = []
    for stem in chosen:
        samples.append(record_sample(snn_model, ann, stem))
        s = samples[-1]
        print(f"{stem:28s} input {s['input_psnr']:6.2f}  spiking t=1 "
              f"{s['steps'][0]['psnr']:6.2f} -> t={T} {s['steps'][-1]['psnr']:6.2f}  "
              f"ANN {s['ann_psnr']:6.2f} dB")

    data = dict(T=T, probe_fracs=PROBE_FRACS, max_probe_ch=MAX_PROBE_CH,
                mem_scale=MEM_SCALE, map_levels=MAP_LEVELS, populations=meta, samples=samples,
                device=str(DEVICE), torch=torch.__version__)
    with open(OUT, "w") as f:
        f.write("// Generated by demo/tools/record_spiking_dynamics.py -- do not edit.\n")
        f.write("window.LIMO_SNN = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
