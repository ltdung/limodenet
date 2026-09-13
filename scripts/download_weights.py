#!/usr/bin/env python
"""
Download released LIMODENet checkpoints from the HuggingFace Hub.

    python scripts/download_weights.py                     # everything
    python scripts/download_weights.py --model tiny         # one classifier
    python scripts/download_weights.py --group spiking       # one group
    python scripts/download_weights.py --list

Checkpoints are licensed CC-BY-NC-4.0 (see WEIGHTS_LICENSE.md), separately
from the MIT-licensed code.
"""
import argparse
import os

REPO_ID = "ltdung/limodenet"

# name -> (HF filename, group, one-line description)
CHECKPOINTS = {
    "nano":        ("limodenet-nano-eurosat.pth", "classifier",
                     "nano classifier, EuroSAT, from scratch (95.65% top-1, seed 0)"),
    "small":       ("limodenet-small-eurosat.pth", "classifier",
                     "small classifier, EuroSAT, from scratch (96.93% top-1, seed 0)"),
    "base":        ("limodenet-base-eurosat.pth", "classifier",
                     "base classifier, EuroSAT, from scratch (97.26% top-1, seed 0)"),
    "tiny":        ("limodenet-tiny-eurosat-tinft.pth", "classifier",
                     "tiny classifier, EuroSAT, Tiny-ImageNet-pretrained + fine-tuned "
                     "(the paper's headline classifier, seed 0)"),
    "big":         ("limodenet-big-eurosat-tinft.pth", "classifier",
                     "big classifier, EuroSAT, Tiny-ImageNet-pretrained + fine-tuned (seed 0)"),
    "ae":          ("limodenet-ae-1db100q.pth", "reconstructor",
                     "LIMODENet-AE, 1dB/100q DVB-S2X restoration encoder-decoder (seed 0)"),
    "skip-ae":     ("limodenet-skip-ae-1db100q.pth", "reconstructor",
                     "LIMODENet-skip, same task, +additive skip connections (seed 0)"),
    "snn-clf":     ("limodenet-spiking-classifier-t8.pth", "spiking",
                     "spiking classifier, T=8, ANN-init (93.55% top-1 3-seed mean)"),
    "snn-skip-ae": ("limodenet-spiking-skip-recon-t8.pth", "spiking",
                     "spiking LIMODENet-skip reconstructor, T=8, end-to-end, "
                     "zero blocked operations"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(CHECKPOINTS), default=None,
                     help="download only this checkpoint")
    ap.add_argument("--group", choices=["classifier", "reconstructor", "spiking"],
                     default=None, help="download only this group")
    ap.add_argument("--out-dir", default="weights", help="destination directory")
    ap.add_argument("--list", action="store_true", help="list available checkpoints and exit")
    args = ap.parse_args()

    if args.list:
        for name, (fn, group, desc) in CHECKPOINTS.items():
            print(f"  {name:<12} [{group:<13}] {desc}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Install the Hub client first: pip install huggingface_hub")

    targets = CHECKPOINTS.items()
    if args.model:
        targets = [(args.model, CHECKPOINTS[args.model])]
    elif args.group:
        targets = [(k, v) for k, v in CHECKPOINTS.items() if v[1] == args.group]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, (filename, group, desc) in targets:
        print(f"[{name}] downloading {filename} ...")
        path = hf_hub_download(repo_id=REPO_ID, filename=filename,
                                local_dir=args.out_dir)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
