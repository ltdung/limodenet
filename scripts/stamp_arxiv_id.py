"""Replace the ARXIV_ID placeholder everywhere once the preprint is live.

Usage:
    python scripts/stamp_arxiv_id.py 2609.01234
"""
import io
import os
import sys

TARGETS = [
    "README.md",
    "weights_staging/HF_MODEL_CARD.md",
]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/stamp_arxiv_id.py <arxiv-id>")
    arxiv_id = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in TARGETS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"skip (missing): {rel}")
            continue
        text = io.open(path, encoding="utf-8").read()
        if "ARXIV_ID" not in text:
            print(f"skip (already stamped): {rel}")
            continue
        io.open(path, "w", encoding="utf-8", newline="\n").write(
            text.replace("ARXIV_ID", arxiv_id)
        )
        print(f"stamped: {rel}")


if __name__ == "__main__":
    main()
