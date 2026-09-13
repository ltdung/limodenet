#!/usr/bin/env python
"""
Convert an ANN LIMODENet checkpoint to a spiking network and check the
converted model's accuracy against a held-out set.

Classifier:
    python scripts/convert_to_spiking.py classifier \
        --ckpt weights/limodenet-tiny-eurosat-tinft.pth \
        --data /path/to/EuroSAT_RGB/test --num-classes 10 --T 8

Reconstructor (LIMODENet-skip only -- the cell verified to convert with
zero blocked operations):
    python scripts/convert_to_spiking.py reconstructor \
        --ckpt weights/limodenet-skip-ae-1db100q.pth \
        --data /path/to/paired_test --T 8

Requires `snntorch` (`pip install snntorch`).
"""
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def cmd_classifier(args):
    from limodenet.snn import SpikingLIMODENet, warm_start

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingLIMODENet(num_classes=args.num_classes, T=args.T).to(device)
    warm_start(model, args.ckpt)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(args.data, tf)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    correct = total = 0
    with torch.no_grad():
        for x, y in ld:
            x, y = x.to(device), y.to(device)
            pred = model(x, T=args.T).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    print(f"spiking top-1 (T={args.T}): {100 * correct / total:.2f}%  ({correct}/{total})")


def cmd_reconstructor(args):
    from limodenet.snn_recon import SpikingLIMODENetSkipRecon, warm_start
    from scripts.evaluate import PairedFolder, psnr  # reuse the paired-data loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingLIMODENetSkipRecon(T=args.T).to(device)
    warm_start(model, args.ckpt)
    model.eval()

    ds = PairedFolder(args.data)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    total_psnr = n = 0.0
    with torch.no_grad():
        for degraded, clean in ld:
            degraded, clean = degraded.to(device), clean.to(device)
            restored = model(degraded, T=args.T)
            total_psnr += psnr(restored, clean).sum().item()
            n += degraded.size(0)
    print(f"spiking mean PSNR (T={args.T}): {total_psnr / n:.2f} dB  (n={int(n)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("classifier")
    p1.add_argument("--ckpt", required=True, help="ANN checkpoint to warm-start from")
    p1.add_argument("--data", required=True)
    p1.add_argument("--num-classes", type=int, default=10)
    p1.add_argument("--T", type=int, default=8)
    p1.add_argument("--batch", type=int, default=64)
    p1.set_defaults(func=cmd_classifier)

    p2 = sub.add_parser("reconstructor")
    p2.add_argument("--ckpt", required=True, help="ANN LIMODENet-skip checkpoint")
    p2.add_argument("--data", required=True)
    p2.add_argument("--T", type=int, default=8)
    p2.add_argument("--batch", type=int, default=16)
    p2.set_defaults(func=cmd_reconstructor)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
