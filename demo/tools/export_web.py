#!/usr/bin/env python
"""
Build everything the in-browser demo (demo/index.html) needs, and check it.

  1. Export every restorer in demo/registry.py to ONNX (weights_staging/onnx/,
     uploaded to the Hub's onnx/ folder by upload_to_hf.py).
  2. Verify each ONNX model against its PyTorch checkpoint on all shipped
     samples with onnxruntime: the export fails loudly if they disagree.
  3. Write the samples as 128x128 PNGs (demo/web/samples/), resized exactly as
     in the paper's evaluation, so the browser sees the same input tensors.
  4. Write demo/web/demo_data.js: model metadata (parameters and mult-adds
     measured from the PyTorch models), the sample list and the audit result.

    LIMODENET_WEIGHTS=weights_staging python demo/tools/export_web.py

Needs torch, snntorch, onnx and onnxruntime.
"""
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(DEMO, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, DEMO)

import audit as A                            # noqa: E402
from registry import DEFAULT_MODELS, MODELS  # noqa: E402

WEIGHTS = os.environ.get("LIMODENET_WEIGHTS", os.path.join(ROOT, "weights"))
ONNX_DIR = os.path.join(ROOT, "weights_staging", "onnx")
WEB = os.path.join(DEMO, "web")
SAMPLES = os.path.join(DEMO, "samples")
OPSET = 17
ATOL = 1e-3          # max |ONNX - PyTorch| per pixel, on a [0,1] image
# The spiking model cannot meet ATOL against ANY second backend: a membrane
# within ~1e-6 of threshold fires or not depending on float rounding, and the
# flip propagates. Measured 2026-10-18 on the 30 samples: PyTorch CPU vs CUDA
# differ by 0.085 max per pixel (non-spiking control: 1.4e-6); ONNX vs PyTorch
# CPU has 718 of ~180M membranes across threshold at t=1, cascading to ~2% by
# t=8. Image-level quality is what must agree, so it is checked on pooled PSNR.
SPIKING_DPSNR = 0.05  # dB


def heaviside(x):
    """Forward pass of every snntorch surrogate: spike iff U - theta > 0.
    The surrogate only changes gradients, which inference never uses."""
    return (x > 0).float()


def load(name):
    spec = MODELS[name]
    model = spec["build"]().eval()
    sd = torch.load(os.path.join(WEIGHTS, spec["file"]), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model


def exportable(model):
    """Swap snntorch's autograd spike function for the plain step function so
    the graph traces to standard ONNX ops (no-op for non-spiking models)."""
    try:
        import snntorch as snn
    except ImportError:
        return model
    for m in model.modules():
        if isinstance(m, snn.Leaky):
            m.spike_grad = heaviside
    return model


class SpikingStep(torch.nn.Module):
    """One timestep of the spiking network with its membrane potentials as
    explicit inputs/outputs. The browser runs the T-step loop itself, which
    keeps one copy of the weights (an unrolled export duplicates them T times)
    and lets the page show the output after every step."""
    def __init__(self, net):
        super().__init__()
        import snntorch as snn
        self.net = net
        self.lifs = [m for _, m in net.named_modules() if isinstance(m, snn.Leaky)]

    def forward(self, x, *mems):
        for lif, m in zip(self.lifs, mems):
            lif.mem = m
        y = self.net._step(x)
        return (y, *[lif.mem for lif in self.lifs])

    def state_shapes(self):
        from snntorch import utils
        utils.reset(self.net)
        with torch.no_grad():
            self.net._step(torch.zeros(1, 3, 128, 128))
        return [list(l.mem.shape) for l in self.lifs]


def run_spiking_onnx(sess, x, shapes, T):
    """Exactly what the browser does: zero state, T steps, mean of outputs."""
    mems = [np.zeros(s, dtype=np.float32) for s in shapes]
    acc = 0.0
    for _ in range(T):
        outs = sess.run(None, dict({"input": x}, **{f"mem{i}": m for i, m in enumerate(mems)}))
        acc = acc + outs[0]
        mems = outs[1:]
    return acc / T


def load_inputs():
    """Samples resized to 128x128 with PIL bilinear, as torchvision's
    Resize((128,128)) does in the paper's evaluation code."""
    stems = sorted(f[:-len("_degraded.png")] for f in os.listdir(SAMPLES)
                   if f.endswith("_degraded.png"))
    deg, cln = [], []
    for s in stems:
        deg.append(np.array(Image.open(os.path.join(SAMPLES, f"{s}_degraded.png"))
                            .convert("RGB").resize((128, 128), Image.BILINEAR)))
        cln.append(np.array(Image.open(os.path.join(SAMPLES, f"{s}_clean.jpg"))
                            .convert("RGB").resize((128, 128), Image.BILINEAR)))
    return stems, deg, cln


def to_tensor(arrs):
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float() / 255.0


def pooled_psnr(out, ref):
    return 10 * math.log10(1.0 / torch.mean((out - ref) ** 2).item())


def main():
    import onnxruntime as ort
    from torch.utils.flop_counter import FlopCounterMode

    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(os.path.join(WEB, "samples"), exist_ok=True)

    stems, deg, cln = load_inputs()
    for s, d, c in zip(stems, deg, cln):
        Image.fromarray(d).save(os.path.join(WEB, "samples", f"{s}_degraded.png"))
        Image.fromarray(c).save(os.path.join(WEB, "samples", f"{s}_clean.png"))
    X, Y = to_tensor(deg), to_tensor(cln)

    audit = A.run_all()
    models_meta = []
    for name, spec in MODELS.items():
        model = exportable(load(name))                  # instance used only for export
        dummy = torch.rand(1, 3, 128, 128)
        path = os.path.join(ONNX_DIR, spec["onnx"])
        spiking = name.startswith("Spiking")
        state = None
        with torch.no_grad():
            if spiking:
                step = SpikingStep(model)
                state = step.state_shapes()
                mems = tuple(torch.zeros(s) for s in state)
                torch.onnx.export(step, (dummy, *mems), path, opset_version=OPSET, dynamo=False,
                                  input_names=["input"] + [f"mem{i}" for i in range(len(state))],
                                  output_names=["output"] + [f"mem_out{i}" for i in range(len(state))])
            else:
                torch.onnx.export(model, dummy, path, opset_version=OPSET, dynamo=False,
                                  input_names=["input"], output_names=["output"])

        # --- verify ONNX against a FRESH PyTorch instance on every sample
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ref_model = load(name)                          # untouched spike function
        with torch.no_grad():
            ref = torch.cat([ref_model(X[i:i + 1]) for i in range(len(X))]).clamp(0, 1)
        if spiking:
            got = torch.cat([torch.from_numpy(run_spiking_onnx(sess, X[i:i + 1].numpy(), state,
                                                               ref_model.T))
                             for i in range(len(X))]).clamp(0, 1)
        else:
            got = torch.cat([torch.from_numpy(sess.run(None, {"input": X[i:i + 1].numpy()})[0])
                             for i in range(len(X))]).clamp(0, 1)
        diff = (got - ref).abs().max().item()
        p_pt, p_ox = pooled_psnr(ref, Y), pooled_psnr(got, Y)
        ok = (abs(p_pt - p_ox) < SPIKING_DPSNR) if spiking else (diff < ATOL and abs(p_pt - p_ox) < 0.005)
        print(f"{name:30s} max|onnx-torch| {diff:.2e}  "
              f"PSNR torch {p_pt:.3f} onnx {p_ox:.3f}  "
              f"{os.path.getsize(path) / 1e6:.2f} MB  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"ONNX export of {name} does not match PyTorch")

        fc = FlopCounterMode(display=False)
        with fc, torch.no_grad():
            load(name)(dummy)
        key = spec["audit"]
        models_meta.append(dict(
            name=name, role=spec["role"], onnx=spec["onnx"],
            params=sum(p.numel() for p in model.parameters()),
            gmac=round(fc.get_total_flops() / 2e9, 3),
            blocked=None if key is None else len(audit[key]),
            default=name in DEFAULT_MODELS,
            T=ref_model.T if spiking else None, state=state,
            onnx_bytes=os.path.getsize(path)))

    data = dict(
        models=models_meta,
        samples=[dict(stem=s, cls=s.rsplit("_", 1)[0]) for s in stems],
        audit=[dict(model=k, ops=[dict(module=p, cls=c, why=w) for p, c, w in v])
               for k, v in audit.items()],
        rule=A.__doc__.strip().split("\n\n")[1].replace("\n", " "),
        torch=torch.__version__, opset=OPSET)
    with open(os.path.join(WEB, "demo_data.js"), "w", encoding="utf-8") as f:
        f.write("// Generated by demo/tools/export_web.py -- do not edit.\n")
        f.write("window.LIMO_DEMO = ")
        json.dump(data, f, indent=1)
        f.write(";\n")
    print(f"wrote {len(stems)} sample pairs and demo_data.js to {WEB}")


if __name__ == "__main__":
    main()
