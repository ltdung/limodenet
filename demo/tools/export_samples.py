#!/usr/bin/env python
"""
Export the demo's sample pairs from the paper's private DVB-S2X test split.

Selection is a seeded draw (3 images per class, random.Random(0)) from the
sorted test list, so the set is reproducible and not hand-picked. Run once by
the authors; readers use the exported files in demo/samples/.

    python demo/tools/export_samples.py --clean <EuroSAT_RGB>/test \
        --degraded <DVBS2X_EuroSAT>/Sentry_Data_EsN0db_SNR_1db100qual/test
"""
import argparse
import glob
import os
import random
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--degraded", required=True)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "samples"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)
    for cls in sorted(os.listdir(a.clean)):
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(a.clean, cls, "*.jpg")))
        for stem in rng.sample(stems, a.per_class):
            shutil.copy(os.path.join(a.clean, cls, stem + ".jpg"),
                        os.path.join(a.out, f"{stem}_clean.jpg"))
            shutil.copy(os.path.join(a.degraded, cls, stem + ".png"),
                        os.path.join(a.out, f"{stem}_degraded.png"))
            print(stem)


if __name__ == "__main__":
    main()
