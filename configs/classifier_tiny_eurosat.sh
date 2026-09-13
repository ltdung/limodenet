#!/usr/bin/env bash
# Reproduces the tiny classifier's from-scratch EuroSAT number (Table 1).
# Run once per seed (0, 1, 2) and average.
python scripts/train_classifier.py \
    --preset tiny \
    --data "$1" \
    --num-classes 10 \
    --epochs 30 \
    --batch 64 \
    --lr 3e-4 \
    --weight-decay 5e-2 \
    --seed "${2:-0}" \
    --out "limodenet-tiny-eurosat-scratch-s${2:-0}.pth"
