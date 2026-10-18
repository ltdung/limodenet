#!/usr/bin/env python
"""
Publish the in-browser demo as a static HuggingFace Space.

    python space/deploy_space.py        # -> huggingface.co/spaces/ltdung/limodenet-demo

The Space is assembled from this repository, not from a copy: the static page
demo/index.html with its data (demo/web/) and the spiking-dynamics page go to
the Space root, plus the Space card in this folder. The models are not
uploaded here: the page downloads the ONNX exports from the model repo.
Static Spaces are free. Re-run after any change to the demo.

Needs `huggingface_hub` and a write token (`hf auth login`).
"""
import argparse
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FROM_DEMO = ["index.html", "web/demo_data.js", "web/samples/*",
             "spiking_dynamics.html", "spiking_dynamics_data.js"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space-id", default="ltdung/limodenet-demo")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.space_id, repo_type="space", space_sdk="static", exist_ok=True)

    print("uploading the demo pages and data ...")
    api.upload_folder(folder_path=os.path.join(ROOT, "demo"), repo_id=args.space_id,
                      repo_type="space", allow_patterns=FROM_DEMO,
                      commit_message="Update demo from github.com/ltdung/limodenet")
    print("uploading the Space card ...")
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "space", "README.md"),
                    path_in_repo="README.md", repo_id=args.space_id, repo_type="space",
                    commit_message="Update Space card")
    print(f"Done: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
