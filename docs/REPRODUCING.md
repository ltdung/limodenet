# Reproducing the paper's numbers

This maps each headline table/figure to the script and configuration that
produced it. All protocols follow the paper's standing rules: 3 seeds
minimum, epoch selected on a held-out 10% validation split (never on test),
and reconstruct→classify accuracy always scored under a neutral judge
(architecturally unrelated to the encoders under test).

## Classification family (Table 1 / `tab:rs-datasets`)

```bash
python scripts/train_classifier.py --preset {nano,tiny,small,base,big} \
    --data /path/to/EuroSAT_RGB --num-classes 10 --epochs 30 --seed {0,1,2}
```
Repeat for each of the five RS benchmarks (NWPU-RESISC45, UCMerced,
PatternNet, RSICB128) by pointing `--data`/`--num-classes` at the
corresponding ImageFolder-formatted dataset.

## Restoration comparison (Table `tab:modern`)

```bash
python scripts/train_reconstructor.py --variant limodenet \
    --data /path/to/paired_1db_100q --epochs 60 --seed {0,1,2}
python scripts/train_reconstructor.py --variant limodenet_skip \
    --data /path/to/paired_1db_100q --epochs 60 --seed {0,1,2}
```
`--variant limodenet_depth4` and `limodenet_skip_depth4` give the other two
kill-test cells. The CNN-AE/U-Net baselines and NAFNet-lite/Restormer-lite
are not included in this release (see the paper's supplementary for their
architectures); a plain CNN-AE and skip U-Net are simple enough to
reimplement directly if you want the full comparison.

Evaluate any trained checkpoint:
```bash
python scripts/evaluate.py reconstruct --ckpt recon_checkpoint.pth \
    --variant limodenet_skip --data /path/to/paired_test
```

## Spiking conversion (Table `tab:spiking-recon`, spiking classifier tables)

```bash
python scripts/convert_to_spiking.py classifier \
    --ckpt weights/limodenet-tiny-eurosat-tinft.pth \
    --data /path/to/EuroSAT_RGB/test --T 8

python scripts/convert_to_spiking.py reconstructor \
    --ckpt weights/limodenet-skip-ae-1db100q.pth \
    --data /path/to/paired_test --T 8
```
The reconstructor conversion is verified to have **zero blocked
operations** — every `GELU` becomes a leaky-integrate-and-fire neuron
(`snntorch.Leaky`), every `Conv2d`/`GroupNorm`/skip-addition stays graded
synaptic. This is the same substitution rule applied to both models; see
`limodenet/snn.py` and `limodenet/snn_recon.py`'s module-level docstrings.

## Data format expected by `train_reconstructor.py` / `evaluate.py reconstruct`

```
paired_root/
├── degraded/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── clean/
    ├── 0001.png     # same filename, spatially aligned
    ├── 0002.png
    └── ...
```
Both are resized to 128×128 on load. Produce this pairing with whatever
channel/compression simulator you use; the paper's own DVB-S2X emulator is
not part of this release (see the main README's Data section).

## Second corpus (ImageNet-C corruptions on PatternNet, Sec. `sec:second-corpus`)

This one *is* fully reproducible with public tools:
```python
from torchvision.datasets import ImageFolder
# apply imagecorruptions.corrupt(image, corruption_name="gaussian_noise", severity=5)
# or corruption_name="pixelate" to each PatternNet image, save as `degraded/`,
# keep the originals as `clean/`, then use train_reconstructor.py as above.
```
`pip install imagecorruptions` provides the exact corruption functions used
in the paper.

## Theory / information-preservation checks

The Lipschitz-constant estimation, block-inversion, and probing scripts
referenced in Sec. 4 (`Verifying the theory`) are research-analysis tools
rather than a trained-model release and are not included here. They operate
directly on `limodenet.family.LIMODENet`'s intermediate activations via
forward hooks — straightforward to reimplement against the architecture in
`limodenet/blocks.py` if you want to reproduce Fig. `fig:injectivity`.
