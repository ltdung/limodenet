#!/usr/bin/env python
"""
Train a LIMODENet classifier from scratch on an ImageFolder-style dataset.

    python scripts/train_classifier.py --preset tiny \
        --data /path/to/EuroSAT_RGB --num-classes 10 --epochs 30

Reproduces the paper's protocol: AdamW (weight decay 5e-2), cosine LR,
64x64 inputs, RandomResizedCrop(0.6-1.0) + horizontal flip, epoch selected
on a held-out 10% validation split (never on test).
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_loaders(root, batch, val_frac=0.1, split_seed=1234, workers=2):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(64), transforms.CenterCrop(64),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    full_train = datasets.ImageFolder(os.path.join(root, "train"), train_tf)
    n_val = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(split_seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)
    val_ds.dataset = datasets.ImageFolder(os.path.join(root, "train"), eval_tf)  # eval transform on val

    test_ds = datasets.ImageFolder(os.path.join(root, "test"), eval_tf)

    return (
        DataLoader(train_ds, batch, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(val_ds, batch, shuffle=False, num_workers=workers, pin_memory=True),
        DataLoader(test_ds, batch, shuffle=False, num_workers=workers, pin_memory=True),
        len(full_train.classes),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100 * correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="tiny", choices=["nano", "tiny", "small", "base", "big"])
    ap.add_argument("--data", required=True, help="dataset root with train/ and test/ subfolders")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoint.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from limodenet.family import build, count_params
    model = build(args.preset, num_classes=args.num_classes).to(device)
    print(f"{args.preset}: {count_params(model):,} params")

    train_ld, val_ld, test_ld, n_classes = get_loaders(args.data, args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        val_acc = evaluate(model, val_ld, device)
        print(f"epoch {epoch+1}/{args.epochs}  val_acc={val_acc:.2f}%")
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_ld, device)
    print(f"final (val-selected epoch): test_acc={test_acc:.2f}%")
    torch.save(best_state, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
