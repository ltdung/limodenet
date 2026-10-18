#!/usr/bin/env python
"""
Upload the curated LIMODENet checkpoints (this directory) to a HuggingFace
Hub model repo, along with the model card.

Setup (once):
    pip install huggingface_hub
    hf auth login                  # paste a token with write access

Usage:
    python upload_to_hf.py --repo-id ltdung/limodenet

This creates the repo if it doesn't exist (public, model type) and uploads,
in one commit: every *.pth in this directory, the ONNX exports in onnx/
(used by the in-browser demo, built by demo/tools/export_web.py), the
assets/ images, and HF_MODEL_CARD.md as that repo's README.
Re-running is safe: files with the same names are overwritten, and files
whose content has not changed are not sent again.
"""
import argparse
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
    patterns = ["*.pth", "onnx/*.onnx", "assets/*"]
    print(f"uploading {', '.join(patterns)} from {here} ...")
    api.upload_folder(folder_path=here, repo_id=args.repo_id, repo_type="model",
                      allow_patterns=patterns, commit_message="Update checkpoints and assets")

    card = os.path.join(here, "HF_MODEL_CARD.md")
    if os.path.exists(card):
        print("uploading model card as README.md ...")
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="model",
                        commit_message="Update model card")

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
