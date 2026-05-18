#!/usr/bin/env python3
"""Generate ProtT5 HDF5 embeddings with bio_embeddings.

The LightAttention dataset loader expects the HDF5 key to match the FASTA
record id when `key_format: hash` is used. This script writes remapped FASTA
files with stable numeric ids and stores each embedding under that id.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message="You may be importing Biopython from inside the source tree.*",
)

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

from bio_embeddings.embed import name_to_embedder


DEFAULT_MODEL_CACHE = Path("embedder_model")
DEFAULT_REPO_ID_CACHE = "models--Rostlab--prot_t5_xl_uniref50"


def resolve_model_directory(model_directory: str | Path) -> Path:
    """Return a Hugging Face snapshot directory containing config/spiece files."""
    path = Path(model_directory)
    if (path / "config.json").exists() and (path / "spiece.model").exists():
        return path

    cache_root = path / DEFAULT_REPO_ID_CACHE if (path / DEFAULT_REPO_ID_CACHE).exists() else path
    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        snapshot = cache_root / "snapshots" / refs_main.read_text().strip()
        if (snapshot / "config.json").exists() and (snapshot / "spiece.model").exists():
            return snapshot

    snapshots = cache_root / "snapshots"
    if snapshots.exists():
        for snapshot in sorted(snapshots.iterdir()):
            if (snapshot / "config.json").exists() and (snapshot / "spiece.model").exists():
                return snapshot

    raise FileNotFoundError(
        f"Could not find a ProtT5 Hugging Face snapshot under {model_directory}"
    )


def default_fastas() -> list[Path]:
    return sorted(
        path
        for path in Path("data_files").glob("*.fasta")
        if not path.name.endswith("_remapped.fasta")
    )


def remapped_path_for(fasta: Path) -> Path:
    return fasta.with_name(f"{fasta.stem}_remapped.fasta")


def h5_path_for(fasta: Path) -> Path:
    return fasta.with_suffix(".h5")


def sorted_records(fasta: Path) -> list[SeqRecord]:
    records = list(SeqIO.parse(str(fasta), "fasta"))
    return sorted(records, key=lambda record: -len(record.seq))


def write_remapped_fasta(records: Sequence[SeqRecord], output: Path) -> dict[str, SeqRecord]:
    output.parent.mkdir(parents=True, exist_ok=True)
    remapped_records: list[SeqRecord] = []
    id_to_record: dict[str, SeqRecord] = {}
    for index, record in enumerate(records):
        remapped_id = str(index)
        remapped = SeqRecord(
            record.seq,
            id=remapped_id,
            name=remapped_id,
            description=f"{remapped_id} {record.description}",
        )
        remapped_records.append(remapped)
        id_to_record[remapped_id] = record
    SeqIO.write(remapped_records, str(output), "fasta")
    return id_to_record


def iter_batches(
    items: Sequence[tuple[str, str, int]],
    max_amino_acids: int,
    max_sequences: int,
) -> Iterator[list[tuple[str, str, int]]]:
    batch: list[tuple[str, str, int]] = []
    total = 0
    for item in items:
        length = item[2]
        if batch and (total + length > max_amino_acids or len(batch) >= max_sequences):
            yield batch
            batch = []
            total = 0
        batch.append(item)
        total += length
    if batch:
        yield batch


def embed_batch(embedder, batch: Sequence[tuple[str, str, int]]) -> list[np.ndarray]:
    sequences = [sequence for _, sequence, _ in batch]
    try:
        if hasattr(embedder, "_embed_batch_impl") and hasattr(embedder, "_model"):
            return list(embedder._embed_batch_impl(sequences, embedder._model))
        return list(embedder.embed_batch(sequences))
    except RuntimeError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(batch) == 1:
            raise
        mid = max(1, len(batch) // 2)
        return embed_batch(embedder, batch[:mid]) + embed_batch(embedder, batch[mid:])


def generate_for_fasta(args: argparse.Namespace, fasta: Path, embedder) -> None:
    records = sorted_records(fasta)
    remapped_output = args.remapped_output or remapped_path_for(fasta)
    output = args.output or h5_path_for(fasta)
    id_to_record = write_remapped_fasta(records, remapped_output)

    max_length = None if args.max_length <= 0 else args.max_length
    candidates: list[tuple[str, str, int]] = []
    skipped_length = 0
    for remapped_id, record in id_to_record.items():
        sequence = str(record.seq)
        if max_length is not None and len(sequence) > max_length:
            skipped_length += 1
            continue
        candidates.append((remapped_id, sequence, len(sequence)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "a") as h5_file:
        todo = [item for item in candidates if item[0] not in h5_file]
        print(
            f"{fasta}: {len(records)} records, {len(candidates)} within max length, "
            f"{len(todo)} to embed, {skipped_length} skipped"
        )
        if not todo:
            return

        progress = tqdm(total=len(todo), desc=fasta.stem, unit="seq")
        for batch in iter_batches(todo, args.max_amino_acids, args.max_sequences):
            embeddings = embed_batch(embedder, batch)
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"bio_embeddings returned {len(embeddings)} embeddings for a batch of {len(batch)}"
                )
            for (remapped_id, _, length), embedding in zip(batch, embeddings):
                if embedding.shape[0] != length:
                    raise RuntimeError(
                        f"Length mismatch for {remapped_id}: {embedding.shape[0]} vs {length}"
                    )
                data = embedding.astype(np.float16 if args.store_half else np.float32, copy=False)
                h5_file.create_dataset(remapped_id, data=data, compression=args.compression)
                progress.update(1)
            h5_file.flush()
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LightAttention-compatible .h5 files using bio_embeddings ProtT5."
    )
    parser.add_argument("fasta", nargs="*", type=Path, help="FASTA files. Defaults to data_files/*.fasta.")
    parser.add_argument("--output", type=Path, help="Output .h5 path. Only valid with one FASTA.")
    parser.add_argument("--remapped-output", type=Path, help="Output remapped FASTA path. Only valid with one FASTA.")
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--protocol", default="prottrans_t5_xl_u50")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=6000)
    parser.add_argument("--max-amino-acids", type=int, default=12000)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--store-float32", dest="store_half", action="store_false")
    parser.add_argument("--full-precision-model", dest="half_precision_model", action="store_false")
    parser.add_argument("--compression", default=None)
    parser.set_defaults(store_half=True, half_precision_model=True)
    args = parser.parse_args()

    if (args.output or args.remapped_output) and len(args.fasta) != 1:
        parser.error("--output and --remapped-output require exactly one FASTA")
    if args.protocol not in name_to_embedder:
        parser.error(f"Unknown bio_embeddings protocol: {args.protocol}")
    if args.max_amino_acids <= 0:
        parser.error("--max-amino-acids must be positive")
    if args.max_sequences <= 0:
        parser.error("--max-sequences must be positive")
    if not args.fasta:
        args.fasta = default_fastas()
    return args


def main() -> None:
    args = parse_args()
    model_directory = resolve_model_directory(args.model_directory)
    print(f"Using model directory: {model_directory}")
    print(
        f"Batch limits: max_amino_acids={args.max_amino_acids}, "
        f"max_sequences={args.max_sequences}, max_length={args.max_length}"
    )

    embedder_class = name_to_embedder[args.protocol]
    embedder = embedder_class(
        model_directory=str(model_directory),
        device=args.device,
        half_precision_model=args.half_precision_model,
    )

    for fasta in args.fasta:
        generate_for_fasta(args, fasta, embedder)

    if torch.cuda.is_available():
        print(f"Peak CUDA allocation: {torch.cuda.max_memory_allocated() // 1024**2} MiB")


if __name__ == "__main__":
    main()
