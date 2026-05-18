#!/usr/bin/env python3
"""Prepare FLIP Meltome FASTA files for embedding generation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def fasta_record(identifier: str, sequence: str, target: str, split: str, validation: bool) -> str:
    return (
        f">{identifier} TARGET={target} SET={split} VALIDATION={validation}\n"
        f"{sequence}\n"
    )


def write_split_fastas(csv_path: Path, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": [], "all": []}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence", "target", "set", "validation"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for row_index, row in enumerate(reader):
            split = row["set"].strip()
            validation = parse_bool(row["validation"])
            identifier = f"Sequence{row_index}"
            record = fasta_record(
                identifier=identifier,
                sequence=row["sequence"].strip(),
                target=row["target"].strip(),
                split=split,
                validation=validation,
            )
            bucket = "val" if split == "train" and validation else split
            if bucket not in ("train", "test", "val"):
                raise ValueError(f"Unexpected set={split!r} in {csv_path}")
            buckets[bucket].append(record)
            buckets["all"].append(record)

    for bucket, records in buckets.items():
        path = output_dir / f"{prefix}_{bucket}.fasta"
        path.write_text("".join(records))
        print(f"wrote {path} ({len(records)} records)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a FLIP Meltome CSV into train/val/test FASTA files."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data_files/flip_meltome/splits/mixed_split.csv"),
        help="FLIP Meltome split CSV. Defaults to the mixed split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_files/flip_meltome/prepared/mixed_split"),
        help="Directory for prepared FASTA files.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the CSV stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_split_fastas(args.csv, args.output_dir, args.prefix or args.csv.stem)


if __name__ == "__main__":
    main()
