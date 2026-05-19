#!/usr/bin/env python3
"""Generate per-layer ProtT5-XL HDF5 embeddings using HuggingFace transformers.

Extracts hidden states from 4 transformer layers of ProtT5-XL (24 encoder blocks),
writing one H5 file per layer. Default layers: 6, 12, 18, 24 (25%, 50%, 75%, 100%
of model depth). Layer 24 matches the final-layer output from embed_bio_embeddings_h5.py.

Each H5 file uses the same numeric key scheme as embed_bio_embeddings_h5.py so the
existing dataset loaders (key_format='hash') work without modification.

Output per input FASTA (e.g. my_seqs.fasta):
    my_seqs_remapped.fasta      — shared remapping (one file for all layers)
    my_seqs_layer06.h5
    my_seqs_layer12.h5
    my_seqs_layer18.h5
    my_seqs_layer24.h5

Usage:
    python scripts/embed_layers_h5.py --fasta data_files/flip_meltome/prepared/human_cell/human_cell_train.fasta
    python scripts/embed_layers_h5.py --fasta a.fasta b.fasta --layers 12 24
    python scripts/embed_layers_h5.py --fasta a.fasta --device cpu
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
DEFAULT_LAYERS = [6, 12, 18, 24]

_NONSTANDARD = re.compile(r"[UOBZJ]")


def _clean_sequence(seq: str) -> str:
    """Replace non-standard amino acids with X, as ProtT5 expects."""
    return _NONSTANDARD.sub("X", seq.upper())


def resolve_model_directory(model_directory: str | Path) -> Path:
    """Return a Hugging Face snapshot directory containing config/spiece files."""
    path = Path(model_directory)
    if (path / "config.json").exists() and (path / "spiece.model").exists():
        return path

    repo_cache = path / "models--Rostlab--prot_t5_xl_uniref50"
    cache_root = repo_cache if repo_cache.exists() else path
    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        snapshot = cache_root / "snapshots" / refs_main.read_text().strip()
        if (snapshot / "config.json").exists():
            return snapshot

    snapshots = cache_root / "snapshots"
    if snapshots.exists():
        for snapshot in sorted(snapshots.iterdir()):
            if (snapshot / "config.json").exists():
                return snapshot

    raise FileNotFoundError(
        f"Could not find a ProtT5 snapshot under {model_directory}"
    )


def sorted_records(fasta: Path) -> list[SeqRecord]:
    records = list(SeqIO.parse(str(fasta), "fasta"))
    return sorted(records, key=lambda r: -len(r.seq))


def write_remapped_fasta(records: Sequence[SeqRecord], output: Path) -> list[tuple[str, str]]:
    """Write remapped FASTA with numeric IDs. Returns [(remapped_id, cleaned_seq), ...]."""
    output.parent.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    remapped: list[SeqRecord] = []
    for index, record in enumerate(records):
        rid = str(index)
        cleaned = _clean_sequence(str(record.seq))
        remapped.append(SeqRecord(
            record.seq,
            id=rid,
            name=rid,
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
def embed_batch_multilayer(
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    batch: list[tuple[str, str]],
    layers: list[int],
    device: torch.device,
    store_half: bool,
) -> dict[int, list[tuple[str, np.ndarray]]]:
    """Returns {layer: [(rid, embedding_array), ...]} for all requested layers."""
    ids = [rid for rid, _ in batch]
    seqs = [seq for _, seq in batch]
    lengths = [len(s) for s in seqs]

    spaced = [" ".join(list(s)) for s in seqs]
    enc = tokenizer(spaced, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    dtype = np.float16 if store_half else np.float32
    result: dict[int, list[tuple[str, np.ndarray]]] = {layer: [] for layer in layers}
    for layer in layers:
        hidden = out.hidden_states[layer]   # [B, token_len, d]
        for i, (rid, length) in enumerate(zip(ids, lengths)):
            emb = hidden[i, :length, :].cpu().float().numpy().astype(dtype)
            result[layer].append((rid, emb))
    return result


def generate_for_fasta(
    args: argparse.Namespace,
    fasta: Path,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
) -> None:
    records = sorted_records(fasta)
    remapped_out = args.remapped_output or fasta.with_name(f"{fasta.stem}_remapped.fasta")
    items = write_remapped_fasta(records, remapped_out)
    print(f"Wrote remapped FASTA: {remapped_out}")

    h5_paths = {
        layer: fasta.with_name(f"{fasta.stem}_layer{layer:02d}.h5")
        for layer in args.layers
    }

    max_length = None if args.max_length <= 0 else args.max_length
    candidates = [(rid, seq) for rid, seq in items
                  if max_length is None or len(seq) <= max_length]
    skipped = len(items) - len(candidates)
    print(
        f"{fasta.name}: {len(records)} records, {len(candidates)} within max_length, "
        f"{skipped} skipped"
    )

    # Open all H5 files and find which sequences still need embedding
    h5_handles = {layer: h5py.File(h5_paths[layer], "a") for layer in args.layers}
    try:
        todo = [item for item in candidates
                if any(item[0] not in h5_handles[layer] for layer in args.layers)]
        print(f"  {len(todo)} sequences to embed across layers {args.layers}")
        if not todo:
            print("  All sequences already embedded — skipping.")
            return

        progress = tqdm(total=len(todo), desc=fasta.stem, unit="seq")
        for batch in iter_batches(todo, args.max_amino_acids, args.max_sequences):
            layer_results = embed_batch_multilayer(
                tokenizer, model, batch, args.layers, device, args.store_half
            )
            for layer, layer_items in layer_results.items():
                h5 = h5_handles[layer]
                for rid, emb in layer_items:
                    if rid not in h5:
                        h5.create_dataset(rid, data=emb, compression=args.compression)
            for h5 in h5_handles.values():
                h5.flush()
            progress.update(len(batch))
        progress.close()
    finally:
        for h5 in h5_handles.values():
            h5.close()

    for layer, path in h5_paths.items():
        print(f"  Layer {layer:2d}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-layer ProtT5-XL embeddings into separate H5 files."
    )
    parser.add_argument("--fasta", nargs="+", required=True, type=Path,
                        help="Input FASTA file(s).")
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS,
                        help=f"Transformer layers to extract (1-indexed block outputs). "
                             f"Default: {DEFAULT_LAYERS}. ProtT5-XL has 24 blocks.")
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_CACHE,
                        help="Path to HuggingFace ProtT5 model snapshot.")
    parser.add_argument("--device", default="cuda",
                        help="Torch device: cuda, mps, or cpu.")
    parser.add_argument("--max-length", type=int, default=6000,
                        help="Skip sequences longer than this (0 = no limit).")
    parser.add_argument("--max-amino-acids", type=int, default=4000,
                        help="Max total residues per batch (lower than bio_embeddings "
                             "due to output_hidden_states memory overhead).")
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--store-float32", dest="store_half", action="store_false",
                        help="Store embeddings as float32 instead of float16.")
    parser.add_argument("--compression", default=None,
                        help="H5 compression (e.g. 'gzip').")
    parser.add_argument("--remapped-output", type=Path, default=None,
                        help="Override remapped FASTA output path (single-FASTA mode only).")
    parser.set_defaults(store_half=True)
    args = parser.parse_args()

    if args.remapped_output and len(args.fasta) != 1:
        parser.error("--remapped-output requires exactly one --fasta")
    for layer in args.layers:
        if not (0 <= layer <= 24):
            parser.error(f"Layer {layer} out of range [0, 24] for ProtT5-XL (24 encoder blocks).")
    return args


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_directory(args.model_directory)
    print(f"Model directory: {model_dir}")
    print(f"Layers to extract: {args.layers}")

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
    tokenizer = T5Tokenizer.from_pretrained(str(model_dir), do_lower_case=False)
    model = T5EncoderModel.from_pretrained(str(model_dir))
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()

    for fasta in args.fasta:
        print(f"\n=== {fasta} ===")
        generate_for_fasta(args, fasta, tokenizer, model, device)

    if device.type == "cuda":
        print(f"\nPeak CUDA memory: {torch.cuda.max_memory_allocated() // 1024**2} MiB")


if __name__ == "__main__":
    main()
