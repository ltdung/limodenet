#!/usr/bin/env python
"""
Evaluate a released LIMODENet checkpoint.

Classifier, top-1 accuracy on an ImageFolder-style test split:
    python scripts/evaluate.py classify --ckpt weights/limodenet-tiny-eurosat-tinft.pth \
        --preset tiny --data /path/to/EuroSAT_RGB/test --num-classes 10

Reconstructor, PSNR/SSIM on paired (degraded, clean) images. `--data` should
contain `degraded/` and `clean/` subfolders with matching filenames:
    python scripts/evaluate.py reconstruct --ckpt weights/limodenet-ae-1db100q.pth \
        --variant limodenet --data /path/to/paired_test
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_state(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd


def cmd_classify(args):
    from limodenet.family import build

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(args.preset, num_classes=args.num_classes)
    model.load_state_dict(load_state(args.ckpt))
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(args.data, tf)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    correct = total = 0
    with torch.no_grad():
        for x, y in ld:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    print(f"top-1: {100 * correct / total:.2f}%  ({correct}/{total})")


class PairedFolder(Dataset):
    """Expects `root/degraded/*` and `root/clean/*` with matching filenames."""

    def __init__(self, root, size=128):
        self.degraded_dir = os.path.join(root, "degraded")
        self.clean_dir = os.path.join(root, "clean")
        self.names = sorted(os.listdir(self.degraded_dir))
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        degraded = self.tf(Image.open(os.path.join(self.degraded_dir, name)).convert("RGB"))
        clean = self.tf(Image.open(os.path.join(self.clean_dir, name)).convert("RGB"))
        return degraded, clean


def psnr(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return -10 * torch.log10(mse + eps)


def cmd_reconstruct(args):
    from limodenet.recon import build_recon

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_recon(args.variant)
    model.load_state_dict(load_state(args.ckpt))
    model.to(device).eval()

    ds = PairedFolder(args.data)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)

    total_psnr = n = 0.0
    with torch.no_grad():
        for degraded, clean in ld:
            degraded, clean = degraded.to(device), clean.to(device)
            restored = model(degraded)
            total_psnr += psnr(restored, clean).sum().item()
            n += degraded.size(0)
    print(f"mean PSNR: {total_psnr / n:.2f} dB  (n={int(n)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("classify")
    p1.add_argument("--ckpt", required=True)
    p1.add_argument("--preset", default="tiny", choices=["nano", "tiny", "small", "base", "big"])
    p1.add_argument("--data", required=True, help="ImageFolder root (class subdirs)")
    p1.add_argument("--num-classes", type=int, default=10)
    p1.add_argument("--batch", type=int, default=64)
    p1.set_defaults(func=cmd_classify)

    p2 = sub.add_parser("reconstruct")
    p2.add_argument("--ckpt", required=True)
    p2.add_argument("--variant", default="limodenet",
                     choices=["limodenet", "limodenet_skip", "limodenet_depth4", "limodenet_skip_depth4"])
    p2.add_argument("--data", required=True, help="root with degraded/ and clean/ subfolders")
    p2.add_argument("--batch", type=int, default=32)
    p2.set_defaults(func=cmd_reconstruct)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
