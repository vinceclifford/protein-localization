#!/usr/bin/env python3
"""
Download the FLIP Meltome splits, extract human_cell.csv, split into FASTAs,
and generate ProtT5 embeddings.

Usage (download + embed in one step):
    python scripts/prepare_meltome.py

Only download (skip embedding):
    python scripts/prepare_meltome.py --download-only

Only embed (FASTAs already present):
    python scripts/prepare_meltome.py --embed-only
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_files" / "flip_meltome" / "prepared" / "human_cell"

SPLITS_ZIP_URL = (
    "https://github.com/J-SNACKKB/FLIP"
    "/raw/main/splits/meltome/splits.zip"
)
CSV_NAME = "human_cell.csv"
SPLIT = "human_cell"

FASTA_NAMES = [
    f"{SPLIT}_train.fasta",
    f"{SPLIT}_val.fasta",
    f"{SPLIT}_test.fasta",
]


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def csv_to_fastas(csv_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            split = row["set"].strip()
            validation = parse_bool(row["validation"])
            identifier = f"Sequence{row_index}"
            record = (
                f">{identifier} TARGET={row['target'].strip()} "
                f"SET={split} VALIDATION={validation}\n"
                f"{row['sequence'].strip()}\n"
            )
            bucket = "val" if split == "train" and validation else split
            buckets[bucket].append(record)

    paths = []
    for bucket, records in buckets.items():
        path = output_dir / f"{SPLIT}_{bucket}.fasta"
        path.write_text("".join(records))
        print(f"  wrote {path} ({len(records)} records)")
        paths.append(path)
    return [output_dir / name for name in FASTA_NAMES]


def download_fastas(data_dir: Path, force: bool = False) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / CSV_NAME

    if csv_path.exists() and not force:
        print(f"  already exists, skipping: {csv_path.name}")
    else:
        print("  downloading splits.zip ...")
        response = urllib.request.urlopen(SPLITS_ZIP_URL)
        zip_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # find human_cell.csv anywhere in the zip
            matches = [n for n in zf.namelist() if n.endswith(CSV_NAME)]
            if not matches:
                sys.exit(f"ERROR: {CSV_NAME} not found in splits.zip")
            csv_path.write_bytes(zf.read(matches[0]))
        print(f"  → {csv_path}")

    print("  splitting into train/val/test FASTAs...")
    return csv_to_fastas(csv_path, data_dir)


def embed_fastas(fasta_paths: list[Path], embed_args: list[str]) -> None:
    embed_script = PROJECT_ROOT / "scripts" / "embed_bio_embeddings_h5.py"
    cmd = [sys.executable, str(embed_script)] + [str(p) for p in fasta_paths] + embed_args
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare FLIP Meltome data: download + embed.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                   help="Directory to store FASTA and H5 files.")
    p.add_argument("--download-only", action="store_true",
                   help="Download and split into FASTAs but skip embedding.")
    p.add_argument("--embed-only", action="store_true",
                   help="Skip download, embed existing FASTAs.")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download even if CSV already exists.")
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
        print("Downloading FLIP Meltome splits and splitting into FASTAs...")
        fasta_paths = download_fastas(args.data_dir, force=args.force_download)
    else:
        fasta_paths = [args.data_dir / name for name in FASTA_NAMES]
        missing = [p for p in fasta_paths if not p.exists()]
        if missing:
            sys.exit("ERROR: FASTA files missing (run without --embed-only first):\n  " +
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
    print("\nMeltome data preparation complete.")
    print(f"H5 files written to: {args.data_dir}")


if __name__ == "__main__":
    main()
