#!/usr/bin/env python
"""
LIMODENet demo: degrade an image, watch LIMODENet-skip restore it, see the
downstream classifier's prediction before and after.

    pip install -e ".[demo,hub]"
    python demo/app.py

NOTE: the degradation applied here (JPEG re-encoding + additive noise) is a
public-tool APPROXIMATION of the paper's DVB-S2X channel emulator, which is
not redistributed with this release (see the main README's Data section).
Absolute PSNR/dB numbers here will not exactly match the paper's; the
qualitative restore-then-classify behavior is representative.
"""
import io
import os

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from limodenet.family import build
from limodenet.recon import build_recon

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = os.environ.get("LIMODENET_WEIGHTS", "weights")
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _ensure_weights():
    """Download the two checkpoints this demo needs if they're not local."""
    needed = {
        "limodenet-skip-ae-1db100q.pth": os.path.join(WEIGHTS_DIR, "limodenet-skip-ae-1db100q.pth"),
        "limodenet-tiny-eurosat-tinft.pth": os.path.join(WEIGHTS_DIR, "limodenet-tiny-eurosat-tinft.pth"),
    }
    missing = {k: v for k, v in needed.items() if not os.path.exists(v)}
    if missing:
        from huggingface_hub import hf_hub_download
        from scripts.download_weights import REPO_ID
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        for filename in missing:
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=WEIGHTS_DIR)
    return needed


def _load_models():
    paths = _ensure_weights()

    recon = build_recon("limodenet_skip").to(DEVICE).eval()
    recon.load_state_dict(torch.load(paths["limodenet-skip-ae-1db100q.pth"], map_location=DEVICE))

    clf = build("tiny", num_classes=10).to(DEVICE).eval()
    clf.load_state_dict(torch.load(paths["limodenet-tiny-eurosat-tinft.pth"], map_location=DEVICE))

    return recon, clf


RECON, CLF = None, None


def degrade(image, jpeg_quality, noise_std):
    """JPEG re-encode + additive Gaussian noise -- a public-tool stand-in
    for the paper's DVB-S2X channel emulator."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(jpeg_quality))
    buf.seek(0)
    degraded = Image.open(buf).convert("RGB")
    arr = np.array(degraded).astype(np.float32) / 255.0
    arr = arr + np.random.normal(0, noise_std, arr.shape)
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


@torch.no_grad()
def classify(recon_or_degraded_pil):
    tf = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tf(recon_or_degraded_pil).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(CLF(x), dim=1)[0]
    top = torch.argmax(probs).item()
    return f"{EUROSAT_CLASSES[top]}  ({probs[top]*100:.1f}%)"


@torch.no_grad()
def run(image, jpeg_quality, noise_std):
    global RECON, CLF
    if RECON is None:
        RECON, CLF = _load_models()

    image = image.convert("RGB").resize((128, 128))
    degraded = degrade(image, jpeg_quality, noise_std)

    x = transforms.ToTensor()(degraded).unsqueeze(0).to(DEVICE)
    restored = RECON(x).clamp(0, 1)[0].cpu()
    restored_pil = transforms.ToPILImage()(restored)

    label_degraded = classify(degraded)
    label_restored = classify(restored_pil)

    return degraded, restored_pil, label_degraded, label_restored


with gr.Blocks(title="LIMODENet demo") as demo:
    gr.Markdown(
        "# LIMODENet: restore, then classify\n"
        "Upload a satellite image (EuroSAT-like scenes work best), degrade it, "
        "and watch LIMODENet-skip restore it before classification.\n\n"
        "**Note:** the degradation here (JPEG + noise) approximates but does not "
        "reproduce the paper's DVB-S2X channel emulator, which is not "
        "redistributed with this release."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Clean input")
        with gr.Column():
            jpeg_q = gr.Slider(1, 100, value=10, step=1, label="JPEG quality")
            noise = gr.Slider(0.0, 0.3, value=0.1, step=0.01, label="Noise std")
            btn = gr.Button("Degrade & restore", variant="primary")
    with gr.Row():
        out_degraded = gr.Image(type="pil", label="Degraded")
        out_restored = gr.Image(type="pil", label="LIMODENet-skip restored")
    with gr.Row():
        lbl_degraded = gr.Textbox(label="Classified from degraded input")
        lbl_restored = gr.Textbox(label="Classified from restored image")

    btn.click(run, [inp, jpeg_q, noise], [out_degraded, out_restored, lbl_degraded, lbl_restored])

if __name__ == "__main__":
    demo.launch()
