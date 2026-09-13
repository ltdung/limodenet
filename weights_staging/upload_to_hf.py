#!/usr/bin/env python
"""
One-time upload of the curated LIMODENet checkpoints (this directory) to a
HuggingFace Hub model repo, along with the model card.

Setup (once):
    pip install huggingface_hub
    huggingface-cli login          # paste a token with write access

Usage:
    python upload_to_hf.py --repo-id ltdung/limodenet

This creates the repo if it doesn't exist (public, model type) and uploads
every *.pth in this directory, the assets/ images, and HF_MODEL_CARD.md
as that repo's README.
Re-running is safe -- it overwrites files with the same names.
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="ltdung/limodenet", help="target HF model repo")
    ap.add_argument("--private", action="store_true", help="create/keep the repo private")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    ckpts = sorted(glob.glob(os.path.join(here, "*.pth")))
    print(f"Found {len(ckpts)} checkpoints to upload:")
    for c in ckpts:
        print(f"  {os.path.basename(c)}")

    for c in ckpts:
        print(f"uploading {os.path.basename(c)} ...")
        api.upload_file(
            path_or_fileobj=c,
            path_in_repo=os.path.basename(c),
            repo_id=args.repo_id,
            repo_type="model",
        )

    assets = sorted(glob.glob(os.path.join(here, "assets", "*")))
    for a in assets:
        print(f"uploading assets/{os.path.basename(a)} ...")
        api.upload_file(
            path_or_fileobj=a,
            path_in_repo=f"assets/{os.path.basename(a)}",
            repo_id=args.repo_id,
            repo_type="model",
        )

    card = os.path.join(here, "HF_MODEL_CARD.md")
    if os.path.exists(card):
        print("uploading model card as README.md ...")
        api.upload_file(
            path_or_fileobj=card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
