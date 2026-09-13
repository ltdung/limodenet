#!/usr/bin/env bash
# Reproduces LIMODENet-skip's headline restoration number (Table `tab:modern`),
# 1 dB / JPEG quality 100 DVB-S2X operating point. Run once per seed and average.
python scripts/train_reconstructor.py \
    --variant limodenet_skip \
    --data "$1" \
    --epochs 60 \
    --batch 32 \
    --lr 3e-4 \
    --weight-decay 5e-2 \
    --val-frac 0.1 \
    --split-seed 1234 \
    --seed "${2:-0}" \
    --out "limodenet-skip-1db100q-s${2:-0}.pth"
