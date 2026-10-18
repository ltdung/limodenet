---
license: cc-by-nc-4.0
library_name: pytorch
tags:
  - image-restoration
  - image-classification
  - remote-sensing
  - satellite-imagery
  - spiking-neural-network
  - neuromorphic
  - earth-observation
pipeline_tag: image-to-image
---

<p align="center">
  <img src="assets/logo.png" alt="LIMODENet" width="480">
</p>

<h3 align="center">Attention-Free Compact Encoders for Information-Preserving<br>Onboard Satellite Image Restoration</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2609.14690">Paper (arXiv)</a> ·
  <a href="https://github.com/ltdung/limodenet">Code (GitHub)</a> ·
  <a href="https://github.com/ltdung/SnT-SENTRY">SENTRY project</a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/ltdung/limodenet-demo"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg.svg" alt="Open the interactive demo in HuggingFace Spaces"></a>
  &nbsp;
  <a href="https://ltdung.github.io/limodenet/demo/spiking_dynamics.html"><img src="https://img.shields.io/badge/%E2%96%B6%20Watch%20the%20spiking%20network-live%20page-c2410c?style=for-the-badge" alt="Watch the spiking network: live page"></a>
</p>

<p align="center">
  <img src="assets/graphical_abstract.png" alt="LIMODENet at a glance" width="100%">
</p>

LIMODENet is a compact (0.69M-parameter), softmax-/QKV-free convolutional
backbone for restoring channel-degraded Earth-observation imagery, designed
under the constraint that it must also run as a spiking network on
neuromorphic accelerators (BrainChip Akida, Intel Loihi-2). We do not claim
it is the best restorer available — two modern CNN restorers (NAFNet,
Restormer) beat it on raw fidelity. What we show is that it is the best
restorer that is *verifiably deployable* where the power budget actually is:
it converts end-to-end to a spiking network with **zero blocked operations**,
while the fidelity-winning competitors have 22–24 each.

Code: `pip install git+https://github.com/ltdung/limodenet`

```python
import torch
from limodenet.recon import build_recon
from huggingface_hub import hf_hub_download

path = hf_hub_download("ltdung/limodenet", "limodenet-skip-ae-1db100q.pth")
model = build_recon("limodenet_skip")
model.load_state_dict(torch.load(path))
restored = model(degraded_images)   # (N, 3, 128, 128), values in [0, 1]
```


## ▶ Try it

- **[Interactive demo (Space)](https://huggingface.co/spaces/ltdung/limodenet-demo)**: compare eight restorers on real 1 dB DVB-S2X images, run the spiking-conversion audit, measure the budget.
- **[Watch the spiking network](https://ltdung.github.io/limodenet/demo/spiking_dynamics.html)**: spikes, membrane potentials and the restoration building up over 8 timesteps, recorded from `limodenet-spiking-skip-recon-t8.pth`.

<p align="center"><a href="https://ltdung.github.io/limodenet/demo/spiking_dynamics.html"><img src="https://huggingface.co/ltdung/limodenet/resolve/main/assets/spiking_dynamics.gif" width="88%" alt="Spiking LIMODENet restoring a 1 dB image over 8 timesteps"></a></p>

## Checkpoints

| File | Task | Params | Headline number |
|---|---|---|---|
| `limodenet-nano-eurosat.pth` | classification | 0.39M | 95.65 ± 0.23% top-1 (EuroSAT, scratch, 3-seed) |
| `limodenet-small-eurosat.pth` | classification | 1.55M | 96.93 ± 0.16% |
| `limodenet-base-eurosat.pth` | classification | 2.73M | 97.26 ± 0.17% |
| `limodenet-tiny-eurosat-tinft.pth` | classification | 0.69M | 98.26% (Tiny-ImageNet pretrained + fine-tuned) |
| `limodenet-big-eurosat-tinft.pth` | classification | 5.80M | off-ladder legacy config, Tiny-ImageNet FT |
| `limodenet-ae-1db100q-60ep.pth` | restoration | 0.77M | 37.39 ± 0.14 dB PSNR, 1dB/100q DVB-S2X (this seed: 37.22) |
| `limodenet-ae-1db100q.pth` | restoration | 0.77M | earlier 25-epoch run (this seed: 35.68 dB); kept for compatibility |
| `limodenet-skip-ae-1db100q.pth` | restoration | 0.78M | 38.07 ± 0.26 dB PSNR, same condition |
| `limodenet-spiking-classifier-t8.pth` | classification, spiking | 0.69M | 93.55 ± 0.54% top-1, T=8, ANN-init |
| `limodenet-spiking-skip-recon-t8.pth` | restoration, spiking | 0.78M | 35.95 ± 0.05 dB PSNR, T=8, zero blocked ops |
| `baseline-cnn-ae-1db100q.pth` | restoration, comparison | 0.75M | CNN-AE bottleneck peer (this seed: 35.84 dB) |
| `baseline-unet-1db100q.pth` | restoration, comparison | 0.75M | U-Net skip reference (this seed: 36.70 dB) |
| `baseline-dncnn-1db100q.pth` | restoration, comparison | 0.74M | DnCNN (this seed: 35.49 dB) |
| `baseline-nafnet-lite-1db100q.pth` | restoration, comparison | 0.75M | NAFNet-lite, unconstrained (this seed: 38.34 dB) |
| `baseline-restormer-lite-1db100q.pth` | restoration, comparison | 0.77M | Restormer-lite, unconstrained (this seed: 39.49 dB) |

All classification numbers are on EuroSAT (10-class RGB land-use). All
restoration numbers are on 1 dB $E_s/N_0$, JPEG quality 100 DVB-S2X-degraded
EuroSAT, three-seed mean ± std unless noted. `seed 0` of each 3-seed run is
released; see the paper for the full 3-seed statistics.

The `onnx/` folder holds ONNX exports of the eight restorers, used by the
[in-browser demo](https://huggingface.co/spaces/ltdung/limodenet-demo). They
are checked against these PyTorch checkpoints by `demo/tools/export_web.py`
(identical PSNR/SSIM on the demo's 30 samples; the spiking model, whose spikes
are sensitive to float rounding, agrees within 0.02 dB).

## Loading

```python
# Classifier
from limodenet.family import build
model = build("tiny", num_classes=10)
model.load_state_dict(torch.load("limodenet-tiny-eurosat-tinft.pth"))

# Reconstructor
from limodenet.recon import build_recon
model = build_recon("limodenet_skip")   # or "limodenet" + limodenet-ae-1db100q-60ep.pth
model.load_state_dict(torch.load("limodenet-skip-ae-1db100q.pth"))

# Comparison restorers (Table tab:modern)
from limodenet.baselines import build_baseline
model = build_baseline("unet")          # cnn | unet | dncnn | nafnet | restormer
model.load_state_dict(torch.load("baseline-unet-1db100q.pth"))

# Spiking classifier (needs `pip install snntorch`)
from limodenet.snn import SpikingLIMODENet
model = SpikingLIMODENet(num_classes=10, T=8)
model.load_state_dict(torch.load("limodenet-spiking-classifier-t8.pth"))

# Spiking reconstructor
from limodenet.snn_recon import SpikingLIMODENetSkipRecon
model = SpikingLIMODENetSkipRecon(T=8)
model.load_state_dict(torch.load("limodenet-spiking-skip-recon-t8.pth"))
```

## License

Weights: **CC-BY-NC-4.0** (non-commercial). Code: MIT. See the GitHub repo's
`WEIGHTS_LICENSE.md` for details, or contact the authors for a commercial
license.

## Part of the SENTRY project

LIMODENet is the foundational task of **SENTRY**, a project of the SIGCOM group
at SnT, University of Luxembourg, funded by the Luxembourg National Research
Fund (FNR), combining satellite communication networks with machine learning for
Earth observation. Other tasks:
[github.com/ltdung/SnT-SENTRY](https://github.com/ltdung/SnT-SENTRY)

## Citation

```bibtex
@article{le2026limodenet,
  title   = {LIMODENet: Attention-Free Compact Encoders for Information-Preserving
             Onboard Satellite Image Restoration},
  author  = {Le, Thanh-Dung and Ha, Vu Nguyen and
             Nguyen, Ti Ti and Chatzinotas, Symeon},
  journal = {arXiv preprint arXiv:2609.14690},
  year    = {2026}
}
```
