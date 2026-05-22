#!/usr/bin/env python3
"""
Download DeepLoc FASTA splits and generate ProtT5 embeddings.

The three FASTA files (train / val / test) come from the LightAttention
repository (Stark et al., 2021). Once downloaded, per-residue ProtT5
embeddings are generated with bio_embeddings and saved as H5 files.

Usage (download + embed in one step):
    python scripts/prepare_deeploc.py

Only download (skip embedding):
    python scripts/prepare_deeploc.py --download-only

Only embed (FASTAs already present):
    python scripts/prepare_deeploc.py --embed-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_files"

# FASTA files as published in the LightAttention repository
FASTA_URLS = {
    "deeploc_our_train_set.fasta": (
        "https://github.com/HannesStark/protein-localization"
        "/raw/master/data_files/deeploc_our_train_set.fasta"
    ),
    "deeploc_our_val_set.fasta": (
        "https://github.com/HannesStark/protein-localization"
        "/raw/master/data_files/deeploc_our_val_set.fasta"
    ),
    "deeploc_test_set.fasta": (
        "https://github.com/HannesStark/protein-localization"
        "/raw/master/data_files/deeploc_test_set.fasta"
    ),
}


def download_fastas(data_dir: Path, force: bool = False) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in FASTA_URLS.items():
        dest = data_dir / filename
        if dest.exists() and not force:
            print(f"  already exists, skipping: {dest.name}")
        else:
            print(f"  downloading {filename} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  → {dest}")
        paths.append(dest)
    return paths


def embed_fastas(fasta_paths: list[Path], embed_args: list[str]) -> None:
    embed_script = PROJECT_ROOT / "scripts" / "embed_bio_embeddings_h5.py"
    cmd = [sys.executable, str(embed_script)] + [str(p) for p in fasta_paths] + embed_args
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DeepLoc data: download + embed.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                   help="Directory to store FASTA and H5 files.")
    p.add_argument("--download-only", action="store_true",
                   help="Download FASTAs but skip embedding.")
    p.add_argument("--embed-only", action="store_true",
                   help="Skip download, embed existing FASTAs.")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download even if FASTA files already exist.")
    # Pass-through arguments for embed_bio_embeddings_h5.py
    p.add_argument("--device", default="mps",
                   help="Device for embedding (mps / cuda / cpu).")
    p.add_argument("--model-directory", default="embedder_model",
                   help="Path to ProtT5 model weights.")
    p.add_argument("--max-length", type=int, default=6000)
    p.add_argument("--max-amino-acids", type=int, default=12000)
    p.add_argument("--max-sequences", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Download ──────────────────────────────────────────────────────────────
    if not args.embed_only:
        print("Downloading DeepLoc FASTA files...")
        fasta_paths = download_fastas(args.data_dir, force=args.force_download)
    else:
        fasta_paths = [args.data_dir / name for name in FASTA_URLS]
        missing = [p for p in fasta_paths if not p.exists()]
        if missing:
            sys.exit(f"ERROR: FASTA files missing (run without --embed-only first):\n  " +
                     "\n  ".join(str(p) for p in missing))

    if args.download_only:
        print("Download complete. Run without --download-only to generate embeddings.")
        return

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("\nGenerating ProtT5 embeddings...")
    embed_args = [
        "--device", args.device,
        "--model-directory", args.model_directory,
        "--max-length", str(args.max_length),
        "--max-amino-acids", str(args.max_amino_acids),
        "--max-sequences", str(args.max_sequences),
    ]
    embed_fastas(fasta_paths, embed_args)
    print("\nDeepLoc data preparation complete.")
    print(f"H5 files written to: {args.data_dir}")


if __name__ == "__main__":
    main()
