#!/usr/bin/env python
"""
Train a LIMODENet reconstructor on paired (degraded, clean) images.

    python scripts/train_reconstructor.py --variant limodenet_skip \
        --data /path/to/paired_train --epochs 60

`--data` should contain `degraded/` and `clean/` subfolders with matching
filenames (see scripts/evaluate.py's PairedFolder for the same convention).
Reproduces the paper's protocol: MSE(sum) loss, AdamW, cosine LR, 60 epochs,
epoch selected on a held-out 10% validation split.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image


class PairedFolder(Dataset):
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
    return (-10 * torch.log10(mse + eps)).mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for degraded, clean in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        restored = model(degraded)
        total += psnr(restored, clean) * degraded.size(0)
        n += degraded.size(0)
    return total / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="limodenet",
                     choices=["limodenet", "limodenet_skip", "limodenet_depth4", "limodenet_skip_depth4"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="recon_checkpoint.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from limodenet.recon import build_recon
    model = build_recon(args.variant).to(device)

    full = PairedFolder(args.data)
    n_val = int(len(full) * args.val_frac)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(args.split_seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)

    train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_ld = DataLoader(val_ds, args.batch, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.MSELoss(reduction="sum")

    best_val, best_state = -1e9, None
    for epoch in range(args.epochs):
        model.train()
        for degraded, clean in train_ld:
            degraded, clean = degraded.to(device), clean.to(device)
            opt.zero_grad()
            loss = criterion(model(degraded), clean) / degraded.size(0)
            loss.backward()
            opt.step()
        sched.step()

        val_psnr = evaluate(model, val_ld, device)
        print(f"epoch {epoch+1}/{args.epochs}  val_psnr={val_psnr:.2f} dB")
        if val_psnr > best_val:
            best_val, best_state = val_psnr, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, args.out)
    print(f"best val_psnr={best_val:.2f} dB, saved -> {args.out}")


if __name__ == "__main__":
    main()
