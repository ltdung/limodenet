#!/usr/bin/env python
"""
Spiking-conversion audit: which operations of each restorer have no
spiking-legal equivalent?

The rule is the one that converted LIMODENet end-to-end (limodenet/snn.py,
limodenet/snn_recon.py): every GELU becomes a LIF neuron 1:1, and graded
synaptic primitives (conv, transposed conv, linear, GroupNorm, average
pooling, the output sigmoid) stay as they are. Any other operation is
blocked: it needs a redesign, not a substitution.

This is a static audit of the model definitions. It needs no weights, no
data and no GPU:

    python demo/audit.py
"""
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from limodenet.baselines import build_baseline     # noqa: E402
from limodenet.recon import build_recon             # noqa: E402

GRADED_OK = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.GroupNorm,
             nn.AdaptiveAvgPool2d, nn.Sigmoid, nn.Flatten, nn.Identity)
CONVERTS_1TO1 = (nn.GELU,)

# Modules whose blocking operation is FUNCTIONAL (written inline in forward(),
# e.g. softmax attention or an elementwise product), so a scan of leaf modules
# alone would miss it: MDTA's children are all plain Conv2d. They are matched
# by class first, before the generic leaf check.
REASONS = {
    "LayerNorm2d": "normalises over all channels of a pixel at one instant: global, "
                   "synchronous cross-channel arithmetic with no event-driven form",
    "SimpleGate": "multiplies two dense activation streams (x1 * x2); LIF neurons "
                  "sum weighted spikes, they do not multiply streams",
    "MDTA": "softmax self-attention (q @ k^T, softmax, @ v) written as functional ops "
            "inside forward(); dense all-to-all products plus softmax",
    "GDFN": "gated feed-forward GELU(x1) * x2: the same product of dense streams, with "
            "the GELU inside the gate",
    "PixelShuffle": "channel-to-space rearrangement; not a primitive the rule converts, "
                    "so it would need its own argument (unverified rather than impossible)",
}

# The four restorers audited in the paper (Sec. 4, tab:modern).
MODELS = {
    "LIMODENet-AE": lambda: build_recon("limodenet"),
    "LIMODENet-skip": lambda: build_recon("limodenet_skip"),
    "NAFNet-lite": lambda: build_baseline("nafnet"),
    "Restormer-lite": lambda: build_baseline("restormer"),
}


def audit(model):
    """Return [(module path, class name, reason)] for every blocked op."""
    blocked = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls in REASONS:
            blocked.append((name, cls, REASONS[cls]))
            continue
        if isinstance(mod, GRADED_OK + CONVERTS_1TO1):
            continue
        if len(list(mod.children())) > 0:
            continue                    # container: its children are audited
        blocked.append((name, cls, "no 1:1 spiking-legal equivalent under this rule"))
    # a blocked container's children are not separate findings
    roots = [b for b in blocked
             if not any(b[0].startswith(o[0] + ".") for o in blocked if o is not b)]
    return roots


def run_all():
    """{model name: list of blocked ops}, for the CLI and the Gradio app."""
    return {name: audit(make()) for name, make in MODELS.items()}


def main():
    print(__doc__.strip().split("\n\n")[1])
    results = run_all()
    for name, blocked in results.items():
        print(f"\n=== {name} ===")
        if not blocked:
            print("  0 blocked: every op is graded-synaptic or GELU -> LIF.")
            continue
        print(f"  {len(blocked)} blocked op(s):")
        for path, cls, why in blocked:
            print(f"    {path:<28} {cls:<12} {why}")
    print("\n=== summary ===")
    for name, blocked in results.items():
        kinds = sorted({c for _, c, _ in blocked})
        print(f"  {name:<16} {len(blocked):>3} blocked  {', '.join(kinds)}")


if __name__ == "__main__":
    main()
