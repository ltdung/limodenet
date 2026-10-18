"""
The restorers shown in the demos, in the paper's order: constrained peers,
LIMODENet, then the unconstrained modern restorers. Shared by the Gradio app
(demo/app.py) and the in-browser app's exporter (demo/tools/export_web.py).

`file` is the PyTorch checkpoint on the Hub, `onnx` its in-browser export,
`audit` the row of demo/audit.py that covers the architecture (None: the
paper does not audit it).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from limodenet.baselines import build_baseline     # noqa: E402
from limodenet.recon import build_recon             # noqa: E402


def _spiking():
    from limodenet.snn_recon import SpikingLIMODENetSkipRecon
    return SpikingLIMODENetSkipRecon(T=8)


MODELS = {
    "CNN-AE": dict(
        file="baseline-cnn-ae-1db100q.pth", build=lambda: build_baseline("cnn"),
        role="bottleneck peer", audit=None),
    "U-Net": dict(
        file="baseline-unet-1db100q.pth", build=lambda: build_baseline("unet"),
        role="skip-connection reference", audit=None),
    "DnCNN": dict(
        file="baseline-dncnn-1db100q.pth", build=lambda: build_baseline("dncnn"),
        role="residual denoiser", audit=None),
    "LIMODENet-AE": dict(
        file="limodenet-ae-1db100q-60ep.pth", build=lambda: build_recon("limodenet"),
        role="ours", audit="LIMODENet-AE"),
    "LIMODENet-skip": dict(
        file="limodenet-skip-ae-1db100q.pth", build=lambda: build_recon("limodenet_skip"),
        role="ours + additive skips", audit="LIMODENet-skip"),
    "Spiking LIMODENet-skip (T=8)": dict(
        file="limodenet-spiking-skip-recon-t8.pth", build=_spiking,
        role="ours, end-to-end spiking", audit="LIMODENet-skip"),
    "NAFNet-lite": dict(
        file="baseline-nafnet-lite-1db100q.pth", build=lambda: build_baseline("nafnet"),
        role="modern, unconstrained", audit="NAFNet-lite"),
    "Restormer-lite": dict(
        file="baseline-restormer-lite-1db100q.pth", build=lambda: build_baseline("restormer"),
        role="modern, unconstrained", audit="Restormer-lite"),
}
for _spec in MODELS.values():
    _spec["onnx"] = _spec["file"].replace(".pth", ".onnx")

DEFAULT_MODELS = ["CNN-AE", "U-Net", "LIMODENet-AE", "LIMODENet-skip",
                  "Spiking LIMODENet-skip (T=8)", "NAFNet-lite"]
