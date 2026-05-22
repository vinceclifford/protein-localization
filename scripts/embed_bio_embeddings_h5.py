#!/usr/bin/env python3
"""Generate final-layer ProtT5-XL HDF5 embeddings using HuggingFace transformers.

Extracts the last encoder hidden state (layer 24) of ProtT5-XL and stores
per-residue embeddings as float16 in H5 files compatible with the dataset loaders
(key_format='hash').

Output per input FASTA (e.g. my_seqs.fasta):
    my_seqs_remapped.fasta   — remapped FASTA with numeric IDs
    my_seqs.h5               — per-residue embeddings, keys = numeric IDs

Usage:
    python scripts/embed_bio_embeddings_h5.py data_files/deeploc_our_train_set.fasta
    python scripts/embed_bio_embeddings_h5.py a.fasta b.fasta c.fasta --device mps
"""

from __future__ import annotations

import argparse
import re
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
from transformers import T5EncoderModel, T5Tokenizer


DEFAULT_MODEL_CACHE = Path("embedder_model")
HF_MODEL_ID = "Rostlab/prot_t5_xl_uniref50"
FINAL_LAYER = 24

_NONSTANDARD = re.compile(r"[UOBZJ]")


def _clean_sequence(seq: str) -> str:
    return _NONSTANDARD.sub("X", seq.upper())


def resolve_model_directory(model_directory: str | Path) -> str:
    """Return a local snapshot path or the HuggingFace model ID for `from_pretrained`."""
    path = Path(model_directory)
    if (path / "config.json").exists() and (path / "spiece.model").exists():
        return str(path)

    repo_cache = path / "models--Rostlab--prot_t5_xl_uniref50"
    cache_root = repo_cache if repo_cache.exists() else path
    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        snapshot = cache_root / "snapshots" / refs_main.read_text().strip()
        if (snapshot / "config.json").exists():
            return str(snapshot)

    snapshots = cache_root / "snapshots"
    if snapshots.exists():
        for snapshot in sorted(snapshots.iterdir()):
            if (snapshot / "config.json").exists():
                return str(snapshot)

    # No local snapshot — fall back to HuggingFace hub (will download on first use).
    print(f"  No local model at {path}; downloading {HF_MODEL_ID} from HuggingFace hub.")
    return HF_MODEL_ID


def sorted_records(fasta: Path) -> list[SeqRecord]:
    records = list(SeqIO.parse(str(fasta), "fasta"))
    return sorted(records, key=lambda r: -len(r.seq))


def write_remapped_fasta(records: Sequence[SeqRecord], output: Path) -> list[tuple[str, str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    remapped: list[SeqRecord] = []
    for index, record in enumerate(records):
        rid = str(index)
        cleaned = _clean_sequence(str(record.seq))
        remapped.append(SeqRecord(
            record.seq, id=rid, name=rid,
            description=f"{rid} {record.description}",
        ))
        items.append((rid, cleaned))
    SeqIO.write(remapped, str(output), "fasta")
    return items


def iter_batches(
    items: list[tuple[str, str]],
    max_amino_acids: int,
    max_sequences: int,
) -> Iterator[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    total = 0
    for rid, seq in items:
        length = len(seq)
        if batch and (total + length > max_amino_acids or len(batch) >= max_sequences):
            yield batch
            batch = []
            total = 0
        batch.append((rid, seq))
        total += length
    if batch:
        yield batch


@torch.no_grad()
def embed_batch(
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    batch: list[tuple[str, str]],
    device: torch.device,
    store_half: bool,
) -> list[tuple[str, np.ndarray]]:
    ids = [rid for rid, _ in batch]
    seqs = [seq for _, seq in batch]
    lengths = [len(s) for s in seqs]

    spaced = [" ".join(list(s)) for s in seqs]
    enc = tokenizer(spaced, return_tensors="pt", padding=True, add_special_tokens=True)
    out = model(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
    )
    hidden = out.last_hidden_state  # [B, token_len, d] — final encoder layer
    dtype = np.float16 if store_half else np.float32
    return [
        (rid, hidden[i, :length, :].cpu().float().numpy().astype(dtype))
        for i, (rid, length) in enumerate(zip(ids, lengths))
    ]


def generate_for_fasta(
    args: argparse.Namespace,
    fasta: Path,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
) -> None:
    records = sorted_records(fasta)
    remapped_out = args.remapped_output or fasta.with_name(f"{fasta.stem}_remapped.fasta")
    h5_out = args.output or fasta.with_suffix(".h5")

    items = write_remapped_fasta(records, remapped_out)

    max_length = None if args.max_length <= 0 else args.max_length
    candidates = [(rid, seq) for rid, seq in items
                  if max_length is None or len(seq) <= max_length]
    skipped = len(items) - len(candidates)
    print(
        f"{fasta.name}: {len(records)} records, {len(candidates)} within max_length, "
        f"{skipped} skipped"
    )

    with h5py.File(h5_out, "a") as h5:
        todo = [(rid, seq) for rid, seq in candidates if rid not in h5]
        print(f"  {len(todo)} sequences to embed")
        if not todo:
            print("  All sequences already embedded — skipping.")
            return

        progress = tqdm(total=len(todo), desc=fasta.stem, unit="seq")
        for batch in iter_batches(todo, args.max_amino_acids, args.max_sequences):
            for rid, emb in embed_batch(tokenizer, model, batch, device, args.store_half):
                h5.create_dataset(rid, data=emb, compression=args.compression)
            h5.flush()
            progress.update(len(batch))
        progress.close()

    print(f"  → {h5_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final-layer ProtT5-XL embeddings as H5 files."
    )
    parser.add_argument("fasta", nargs="*", type=Path,
                        help="FASTA files to embed.")
    parser.add_argument("--output", type=Path,
                        help="Output H5 path (single-FASTA mode only).")
    parser.add_argument("--remapped-output", type=Path,
                        help="Remapped FASTA output path (single-FASTA mode only).")
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", default="mps",
                        help="Torch device: mps, cuda, or cpu.")
    parser.add_argument("--max-length", type=int, default=6000)
    parser.add_argument("--max-amino-acids", type=int, default=12000)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--store-float32", dest="store_half", action="store_false")
    parser.add_argument("--compression", default=None)
    parser.set_defaults(store_half=True)
    args = parser.parse_args()

    if (args.output or args.remapped_output) and len(args.fasta) != 1:
        parser.error("--output and --remapped-output require exactly one FASTA")
    return args


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_directory(args.model_directory)
    print(f"Model directory: {model_dir}")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    print("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_dir).to(device).eval()

    for fasta in args.fasta:
        print(f"\n=== {fasta} ===")
        generate_for_fasta(args, fasta, tokenizer, model, device)

    if device.type == "cuda":
        print(f"\nPeak CUDA memory: {torch.cuda.max_memory_allocated() // 1024**2} MiB")


if __name__ == "__main__":
    main()
