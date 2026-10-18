#!/usr/bin/env python
"""
LIMODENet interactive demo.

    pip install -e ".[demo,hub,spiking]"
    python demo/app.py

Tabs:
  1. Restore: pick a real 1 dB DVB-S2X EuroSAT test image (or upload one) and
     compare every restorer from the paper's Table 2 side by side, with PSNR
     and SSIM against the clean reference. All restorers are tiny-band
     (~0.75-0.78M parameters), trained under one protocol (60 epochs,
     val-selected, seed 0).
  2. Spiking audit: the static op audit behind the "zero blocked operations"
     claim, run live on the model definitions.
  3. Budget: parameters, mult-adds and latency, measured on your machine.

The spiking dynamics (spikes, membranes, output build-up over T=8) are in
demo/spiking_dynamics.html; open it in any browser, no server needed.

Weights are read from $LIMODENET_WEIGHTS (default ./weights) and downloaded
from the HuggingFace Hub if missing.
"""
import glob
import math
import os
import statistics
import sys
import time

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DEMO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(DEMO, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, DEMO)

import audit as A                                  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = os.environ.get("LIMODENET_WEIGHTS", os.path.join(ROOT, "weights"))
SAMPLES = os.path.join(DEMO, "samples")
REPO_ID = "ltdung/limodenet"
DYN_HTML = os.path.join(DEMO, "spiking_dynamics.html")
DYN_DATA = os.path.join(DEMO, "spiking_dynamics_data.js")
# serve exactly these two files (not the whole folder) so tab 4 can embed the page
gr.set_static_paths([DYN_HTML, DYN_DATA])


from registry import DEFAULT_MODELS, MODELS        # noqa: E402

_LOADED, _ERRORS = {}, {}


def _weight_path(filename):
    path = os.path.join(WEIGHTS, filename)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        os.makedirs(WEIGHTS, exist_ok=True)
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=WEIGHTS)
    return path


def get_model(name):
    """Build and load a restorer once; remember why if it is unavailable."""
    if name in _LOADED:
        return _LOADED[name]
    if name in _ERRORS:
        return None
    spec = MODELS[name]
    try:
        model = spec["build"]().to(DEVICE).eval()
        sd = torch.load(_weight_path(spec["file"]), map_location=DEVICE)
        model.load_state_dict(sd, strict=True)
    except Exception as e:                        # missing snntorch, offline, ...
        _ERRORS[name] = f"{type(e).__name__}: {e}"
        return None
    _LOADED[name] = model
    return model


# ---------------- metrics: identical to the paper's evaluation code
TF = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


def _gauss(ws, sigma, device):
    c = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim(x, y, ws=11, sigma=1.5):
    """Standard SSIM on [0,1] images, averaged over channels and batch."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch = x.shape[1]
    g = _gauss(ws, sigma, x.device)
    k = (g[:, None] @ g[None, :]).expand(ch, 1, ws, ws).contiguous()
    p = ws // 2
    mu_x = F.conv2d(x, k, padding=p, groups=ch)
    mu_y = F.conv2d(y, k, padding=p, groups=ch)
    mx2, my2, mxy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = F.conv2d(x * x, k, padding=p, groups=ch) - mx2
    sy = F.conv2d(y * y, k, padding=p, groups=ch) - my2
    sxy = F.conv2d(x * y, k, padding=p, groups=ch) - mxy
    m = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) * (sx + sy + C2))
    return m.mean().item()


def psnr(x, y):
    return 10 * math.log10(1.0 / F.mse_loss(x, y).item())


def to_pil(t):
    return transforms.ToPILImage()(t.clamp(0, 1)[0].cpu())


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------- samples
STEMS = sorted(os.path.basename(p)[:-len("_degraded.png")]
               for p in glob.glob(os.path.join(SAMPLES, "*_degraded.png")))


def load_pair(stem):
    deg = Image.open(os.path.join(SAMPLES, f"{stem}_degraded.png")).convert("RGB")
    cln = Image.open(os.path.join(SAMPLES, f"{stem}_clean.jpg")).convert("RGB")
    return deg, cln


def audit_status():
    res = A.run_all()
    return {k: len(v) for k, v in res.items()}


AUDIT = audit_status()


def spiking_label(name):
    key = MODELS[name].get("audit")
    if key is None:
        return "n/a"
    n = AUDIT[key]
    return "yes (0)" if n == 0 else f"no ({n})"


# ---------------- tab 1: restore
@torch.no_grad()
def restore(stem, upload, chosen, err_maps):
    if not chosen:
        raise gr.Error("Select at least one restorer.")
    if upload is not None:
        deg, cln = upload.convert("RGB"), None
        title = "Uploaded image (no clean reference, so no PSNR/SSIM)"
    else:
        deg, cln = load_pair(stem)
        title = f"{stem.replace('_', ' #')} · real 1 dB/q100 DVB-S2X test image"
    x = TF(deg).unsqueeze(0).to(DEVICE)
    y = TF(cln).unsqueeze(0).to(DEVICE) if cln is not None else None

    def show(out):
        """The image itself, or its error map |out - clean| x 5 (bright = wrong)."""
        if err_maps and y is not None:
            return to_pil(((out - y).abs() * 5).clamp(0, 1))
        return to_pil(out)

    def cap(name, out):
        if y is None:
            return name
        return f"{name} · {psnr(out, y):.2f} dB · SSIM {ssim(out, y):.4f}"

    gallery = [(show(x), cap("Degraded input", x))]
    rows = []
    for name in MODELS:
        if name not in chosen:
            continue
        model = get_model(name)
        if model is None:
            rows.append([name, None, None, None, spiking_label(name),
                         "unavailable: " + _ERRORS[name][:80]])
            continue
        t0 = time.perf_counter()
        out = model(x).clamp(0, 1)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        gallery.append((show(out), cap(name, out)))
        rows.append([name, f"{n_params(model) / 1e6:.3f}M",
                     round(psnr(out, y), 2) if y is not None else None,
                     round(ssim(out, y), 4) if y is not None else None,
                     spiking_label(name), f"{ms:.0f} ms"])
    if y is not None and not err_maps:
        gallery.append((to_pil(y), "Clean reference"))
    return f"### {title}", gallery, rows


@torch.no_grad()
def score_all(chosen, progress=gr.Progress()):
    """PSNR pooled over all pixels of all shipped samples (as in the paper),
    SSIM averaged over images."""
    if not chosen:
        raise gr.Error("Select at least one restorer.")
    pairs = [load_pair(s) for s in STEMS]
    X = torch.stack([TF(d) for d, _ in pairs]).to(DEVICE)
    Y = torch.stack([TF(c) for _, c in pairs]).to(DEVICE)
    rows = [["Degraded input (no restoration)", "", "", round(psnr(X, Y), 2),
             round(ssim(X, Y), 4), ""]]
    names = [n for n in MODELS if n in chosen]
    for i, name in enumerate(names):
        progress((i, len(names)), desc=name)
        model = get_model(name)
        if model is None:
            rows.append([name, MODELS[name]["role"], "", None, None, "unavailable"])
            continue
        out = torch.cat([model(X[j:j + 10]).clamp(0, 1) for j in range(0, len(X), 10)])
        rows.append([name, MODELS[name]["role"], f"{n_params(model) / 1e6:.3f}M",
                     round(psnr(out, Y), 2), round(ssim(out, Y), 4), spiking_label(name)])
    return rows


# ---------------- tab 2: audit
def run_audit():
    res = A.run_all()
    summary = [[k, len(v), ", ".join(sorted({c for _, c, _ in v})) or "-"] for k, v in res.items()]
    detail = [[k, path, cls, why] for k, v in res.items() for path, cls, why in v]
    return summary, detail


# ---------------- tab 3: budget
@torch.no_grad()
def budget(progress=gr.Progress()):
    from torch.utils.flop_counter import FlopCounterMode
    rows = []
    x = torch.rand(1, 3, 128, 128, device=DEVICE)
    for i, name in enumerate(MODELS):
        progress((i, len(MODELS)), desc=name)
        model = get_model(name)
        if model is None:
            rows.append([name, "", "", "", "unavailable"])
            continue
        fc = FlopCounterMode(display=False)
        with fc:
            model(x)
        gmac = fc.get_total_flops() / 2e9
        for _ in range(3):
            model(x)
        times = []
        for _ in range(15):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        rows.append([name, f"{n_params(model) / 1e6:.3f}M", round(gmac, 3),
                     round(statistics.median(times), 1), spiking_label(name)])
    return rows


# ---------------- UI
INTRO = """
# LIMODENet: restoring channel-degraded satellite imagery under a spiking constraint
Satellite images sent over a low-SNR DVB-S2X link at **1 dB** arrive badly degraded. Every
restorer here has ~0.75–0.78M parameters and was trained under the same protocol on the same
data. Use this demo to check two claims from the paper:

1. **LIMODENet restores better than the CNN-AE and the U-Net at matched parameter count.**
   NAFNet-lite and Restormer-lite restore better still. They are included so you can see that.
2. **Only LIMODENet converts to a spiking network with zero blocked operations.** NAFNet-lite
   and Restormer-lite each have more than 20 blocked operations.

The samples are real 1 dB/q100 test images, three per class, drawn with a fixed seed (not hand-picked).
Numbers here come from one seed (seed 0) and, under *Score all*, from 30 images. The paper's table uses three
seeds on the full test split, so expect small differences.
"""

with gr.Blocks(title="LIMODENet demo") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("1 · Restore"):
        with gr.Row():
            with gr.Column(scale=1):
                stem = gr.Dropdown(STEMS, value=STEMS[0] if STEMS else None,
                                   label="Real 1 dB DVB-S2X test image")
                thumb = gr.Image(label="Selected degraded input", interactive=False, height=180)
                upload = gr.Image(type="pil", label="…or upload your own degraded image (optional)",
                                  height=180)
                gr.Markdown("<small>The models were trained only on 1 dB/q100 DVB-S2X damage. Other kinds of "
                            "degradation are out of distribution, so read those results as qualitative.</small>")
                chosen = gr.CheckboxGroup(list(MODELS), value=DEFAULT_MODELS, label="Restorers")
                err_maps = gr.Checkbox(False, label="Show error maps |output − clean| × 5 (bright = wrong)")
                go = gr.Button("Restore", variant="primary")
            with gr.Column(scale=3):
                title = gr.Markdown()
                gallery = gr.Gallery(label="Outputs (128×128)", columns=4, height="auto",
                                     object_fit="contain")
                table = gr.Dataframe(headers=["Restorer", "Params", "PSNR (dB)", "SSIM",
                                              "Spiking-legal (blocked ops)", "Time"], interactive=False, wrap=True)
        with gr.Accordion("Score every shipped sample (30 images)", open=False):
            gr.Markdown("PSNR is pooled over all pixels, as in the paper. SSIM is averaged over images.")
            go_all = gr.Button("Score all samples")
            table_all = gr.Dataframe(headers=["Restorer", "Role", "Params", "PSNR (dB)", "SSIM",
                                              "Spiking-legal (blocked ops)"], interactive=False, wrap=True)

        stem.change(lambda s: load_pair(s)[0].resize((128, 128), Image.NEAREST), stem, thumb)
        go.click(restore, [stem, upload, chosen, err_maps], [title, gallery, table])
        err_maps.change(restore, [stem, upload, chosen, err_maps], [title, gallery, table])
        go_all.click(score_all, [chosen], [table_all])
        demo.load(lambda s: load_pair(s)[0].resize((128, 128), Image.NEAREST), stem, thumb)

    with gr.Tab("2 · Spiking audit"):
        gr.Markdown(
            "Conversion rule (the one that converted LIMODENet end-to-end): every **GELU → LIF neuron**; "
            "conv, transposed conv, GroupNorm, average pooling and the output sigmoid **stay graded**. "
            "Any other operation is **blocked**: converting it would need a redesign, not a substitution. "
            "This is a static audit of the model code (`demo/audit.py`). It needs no weights or data, "
            "so you can read the rule and check every line it reports.")
        go_audit = gr.Button("Run the audit", variant="primary")
        audit_sum = gr.Dataframe(headers=["Restorer", "Blocked ops", "Kinds"], interactive=False)
        audit_det = gr.Dataframe(headers=["Restorer", "Module", "Class", "Why it is blocked"],
                                 interactive=False, wrap=True)
        go_audit.click(run_audit, None, [audit_sum, audit_det])

    with gr.Tab("3 · Budget"):
        gr.Markdown(
            f"Parameters, mult-adds (`torch.utils.flop_counter`, attention matmuls included) and median "
            f"batch-1 latency at 128×128, measured **on this machine** ({DEVICE.type}). "
            "The spiking model's latency is T=8 dense passes on conventional hardware. "
            "This is not a neuromorphic-hardware measurement.")
        go_budget = gr.Button("Measure", variant="primary")
        budget_tab = gr.Dataframe(headers=["Restorer", "Params", "GMac", "Latency (ms)", "Spiking-legal (blocked ops)"],
                                  interactive=False)
        go_budget.click(budget, None, [budget_tab])

    with gr.Tab("4 · Spiking dynamics"):
        # the page loads its data with a relative <script src>, which resolves to
        # the second static file under the same /file= route
        dyn_url = "/gradio_api/file=" + DYN_HTML.replace(os.sep, "/")
        gr.Markdown(
            "The trained spiking LIMODENet-skip at work: spikes per layer, membrane potentials, and the "
            "restoration building up over T=8 timesteps, replayed from activity recorded on real 1 dB test "
            f"images. Press **Play**. [Open it full-screen in a new tab]({dyn_url}).")
        gr.HTML(f'<iframe src="{dyn_url}" title="Spiking LIMODENet dynamics" '
                'style="width:100%;height:1500px;border:0;border-radius:8px"></iframe>')

if __name__ == "__main__":
    demo.launch()
