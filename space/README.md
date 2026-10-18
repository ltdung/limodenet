---
title: LIMODENet Demo
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: static
app_file: index.html
pinned: true
license: mit
models:
  - ltdung/limodenet
short_description: Restore 1 dB satellite images in your browser
---

# LIMODENet interactive demo

Runs entirely in your browser (ONNX Runtime Web). Check the paper's claims on
real 1 dB DVB-S2X-degraded EuroSAT test images:

1. **Restore**: eight restorers at matched parameter count, side by side, with PSNR, SSIM and error maps.
2. **Spiking audit**: which operations of each restorer block conversion to a spiking network (0 / 24 / 22).
3. **Budget**: parameters, mult-adds, and latency measured on your device.
4. **Spiking dynamics**: the trained spiking LIMODENet at work, timestep by timestep.

Paper: [arXiv:2609.14690](https://arxiv.org/abs/2609.14690) ·
Code: [github.com/ltdung/limodenet](https://github.com/ltdung/limodenet) ·
Weights: [ltdung/limodenet](https://huggingface.co/ltdung/limodenet)
