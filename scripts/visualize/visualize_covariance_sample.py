#!/usr/bin/env python3
"""Generate covariance and UniProt-alignment visualizations for one protein."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_deeploc_uniprot_site_overlap import (  # noqa: E402
    OFFICIAL_ALL_TYPES,
    average_precision,
    fetch_uniprot_json,
    parse_features,
    threshold_metrics,
)
from scripts.visualize.covariance_visualization_utils import (  # noqa: E402
    combined_residue_scores,
    covariance_analysis,
    load_covariance_model,
    plot_covariance_matrix,
    plot_pair_windows,
    plot_significant_pairs,
    resolve_device,
    select_pairs,
    slug,
)


def clean_sequence(value: str) -> str:
    sequence = "".join(value.split()).upper()
    if not sequence or any(aa not in "ACDEFGHIKLMNPQRSTVWYUBZOXJ" for aa in sequence):
        raise ValueError("--sequence must contain only amino-acid letters")
    return sequence


def fetch_sequence(accession: str, cache_dir: Path) -> tuple[str, dict]:
    data = fetch_uniprot_json(accession, cache_dir)
    if data is None:
        raise RuntimeError(f"Could not retrieve UniProt record {accession}")
    sequence = data.get("sequence", {}).get("value", "")
    if not sequence:
        raise RuntimeError(f"UniProt record {accession} has no sequence")
    return sequence, data


@torch.no_grad()
def embed_sequence(sequence: str, model_directory: Path, device: torch.device) -> torch.Tensor:
    from transformers import T5EncoderModel, T5Tokenizer
    from scripts.embed_bio_embeddings_h5 import resolve_model_directory

    model_path = resolve_model_directory(model_directory)
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    embedder = T5EncoderModel.from_pretrained(model_path).to(device).eval()
    cleaned = sequence.translate(str.maketrans({aa: "X" for aa in "UOBZJ"}))
    encoded = tokenizer(" ".join(cleaned), return_tensors="pt", add_special_tokens=True)
    output = embedder(
        input_ids=encoded["input_ids"].to(device),
        attention_mask=encoded["attention_mask"].to(device),
    ).last_hidden_state[0, :len(sequence)]
    return output.float()


def load_precomputed_embedding(path: Path, key: str | None) -> torch.Tensor:
    with h5py.File(path, "r") as h5:
        if key is None:
            keys = list(h5.keys())
            if len(keys) != 1:
                raise ValueError("--embedding-key is required when the H5 has multiple entries")
            key = keys[0]
        if key not in h5:
            raise KeyError(f"H5 key not found: {key}")
        return torch.tensor(h5[key][:]).float()


def rank_feature_instances(scores: np.ndarray, features: list[dict]) -> pd.DataFrame:
    rows = []
    for index, feature in enumerate(features):
        mask = np.zeros(len(scores), dtype=bool)
        mask[feature["start"] - 1:feature["end"]] = True
        ap = average_precision(scores, mask)
        background = float(mask.mean())
        row = {
            "feature_index": index,
            **feature,
            "length": int(mask.sum()),
            "average_precision": ap,
            "ap_enrichment": ap / background if background > 0 else np.nan,
        }
        for threshold in (1, 5):
            metrics = threshold_metrics(scores, mask, threshold)
            row[f"top{threshold}_precision"] = metrics["precision"]
            row[f"top{threshold}_recall"] = metrics["recall"]
            row[f"top{threshold}_enrichment"] = metrics["enrichment_vs_background"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["ap_enrichment", "top5_enrichment", "top5_recall"], ascending=False
    ).reset_index(drop=True)


def plot_significance_features(
    accession: str,
    scores: np.ndarray,
    ranked: pd.DataFrame,
    out_path: Path,
    top_n: int,
) -> None:
    length = len(scores)
    normalized = scores / scores.max() if scores.max() > 0 else scores
    top1_n = max(1, math.ceil(length * 0.01))
    top5_n = max(1, math.ceil(length * 0.05))
    top1 = np.argpartition(scores, -top1_n)[-top1_n:]
    top5 = np.argpartition(scores, -top5_n)[-top5_n:]
    shown = ranked.head(top_n)
    fig, (ax, feat_ax) = plt.subplots(
        2, 1, figsize=(15, max(5.0, 3.5 + 0.35 * len(shown))), sharex=True,
        gridspec_kw={"height_ratios": [2.4, max(1.2, 0.32 * len(shown))]}, constrained_layout=True,
    )
    positions = np.arange(1, length + 1)
    ax.plot(positions, normalized, color="#202020", linewidth=0.85)
    ax.scatter(top5 + 1, normalized[top5], s=10, color="#f4a261", label="top 5% residues", zorder=3)
    ax.scatter(top1 + 1, normalized[top1], s=15, color="#d62828", label="top 1% residues", zorder=4)
    ax.set_ylabel("normalized significance")
    ax.set_title(f"{accession or 'protein'} | top {len(shown)} UniProt feature matches")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for row_index, row in enumerate(shown.itertuples()):
        feat_ax.broken_barh([(row.start, row.end - row.start + 1)], (row_index - 0.36, 0.72),
                           facecolors="#277da1", alpha=0.85)
    feat_ax.set_yticks(np.arange(len(shown)))
    feat_ax.set_yticklabels([
        f"{row.type}: {row.start}-{row.end} ({row.ap_enrichment:.2f}x AP)"
        for row in shown.itertuples()
    ], fontsize=7.5)
    feat_ax.invert_yaxis()
    feat_ax.set_xlabel("sequence position")
    feat_ax.set_ylabel("UniProt features")
    feat_ax.set_xlim(1, length)
    feat_ax.grid(axis="x", alpha=0.2)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Run directory or checkpoint.pt")
    parser.add_argument("--sequence", help="Raw amino-acid sequence")
    parser.add_argument("--uniprot-id", help="UniProt accession; supplies sequence unless --sequence is given")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--embedder-model", type=Path, default=PROJECT_ROOT / "embedder_model")
    parser.add_argument("--embedding-h5", type=Path, help="Optional precomputed embedding, mainly for reproducibility")
    parser.add_argument("--embedding-key", help="Dataset key for --embedding-h5")
    parser.add_argument("--top-pairs", type=int, default=5)
    parser.add_argument("--pair-sign", choices=["positive", "negative", "absolute"], default="positive")
    parser.add_argument("--top-positions", type=int, default=6)
    parser.add_argument("--window-radius", type=int, default=12)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--uniprot-cache", type=Path, default=PROJECT_ROOT / "figures" / "unified_uniprot_cache")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures" / "unified_sample")
    args = parser.parse_args()
    if not args.sequence and not args.uniprot_id:
        parser.error("provide --sequence or --uniprot-id")
    if args.embedding_h5 and not args.sequence and not args.uniprot_id:
        parser.error("a sequence is still required with --embedding-h5")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    uniprot_data = None
    if args.uniprot_id:
        fetched_sequence, uniprot_data = fetch_sequence(args.uniprot_id, args.uniprot_cache)
        sequence = clean_sequence(args.sequence) if args.sequence else fetched_sequence
    else:
        sequence = clean_sequence(args.sequence)

    embedding = load_precomputed_embedding(args.embedding_h5, args.embedding_key) if args.embedding_h5 else (
        embed_sequence(sequence, args.embedder_model, device)
    )
    if len(sequence) != len(embedding):
        raise ValueError(f"Sequence length {len(sequence)} != embedding length {len(embedding)}")
    embedding = embedding.to(device)
    model, train_args, run_dir = load_covariance_model(args.checkpoint, embedding.shape[1], device)
    analysis = covariance_analysis(model, embedding)
    pairs = select_pairs(analysis, args.top_pairs, args.pair_sign)
    scores = combined_residue_scores(pairs)

    label = slug(args.uniprot_id or "sequence")
    out_dir = args.out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_covariance_matrix(
        analysis["matrix"], out_dir / "covariance_matrix.png",
        f"{args.uniprot_id or 'protein'} compressed covariance ({analysis['kind']})",
        selected_pairs=pairs,
    )
    plot_significant_pairs(pairs, out_dir / "significant_entries.png")
    np.save(out_dir / "covariance_matrix.npy", analysis["matrix"])

    pair_rows, window_frames = [], []
    for pair in pairs:
        pair_rows.append({key: pair[key] for key in ("rank", "left_dim", "right_dim", "C_value")})
        frame = plot_pair_windows(
            sequence, pair, out_dir / f"pair_{pair['rank']:02d}_L{pair['left_dim']}_R{pair['right_dim']}.png",
            args.top_positions, args.window_radius, f"pair rank {pair['rank']}",
        )
        frame.insert(0, "pair_rank", pair["rank"])
        frame.insert(1, "left_dim", pair["left_dim"])
        frame.insert(2, "right_dim", pair["right_dim"])
        frame.insert(3, "C_value", pair["C_value"])
        window_frames.append(frame)
    pd.DataFrame(pair_rows).to_csv(out_dir / "significant_pairs.csv", index=False)
    pd.concat(window_frames, ignore_index=True).to_csv(out_dir / "significant_sequence_windows.csv", index=False)
    pd.DataFrame({"position": np.arange(1, len(sequence) + 1), "aa": list(sequence), "score": scores}).to_csv(
        out_dir / "residue_significance.csv", index=False
    )

    feature_count = 0
    if uniprot_data is not None:
        features, sequence_match = parse_features(uniprot_data, sequence)
        features = [feature for feature in features if feature["type"] in OFFICIAL_ALL_TYPES]
        ranked = rank_feature_instances(scores, features)
        ranked.to_csv(out_dir / "uniprot_feature_matches.csv", index=False)
        feature_count = len(ranked)
        plot_significance_features(args.uniprot_id, scores, ranked, out_dir / "uniprot_feature_alignment_top10.png",
                                   args.top_features)
    else:
        sequence_match = "not_requested"

    metadata = {
        "checkpoint": str(run_dir),
        "pooling": getattr(model, "pooling", analysis["kind"]),
        "sequence_length": len(sequence),
        "uniprot_id": args.uniprot_id,
        "uniprot_sequence_match": sequence_match,
        "n_uniprot_features": feature_count,
        "feature_set": "official",
        "pair_selection": args.pair_sign,
        "top_pairs": args.top_pairs,
        "residue_score": "sum of absolute per-residue contributions over selected covariance entries",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved single-protein visualizations to {out_dir}")


if __name__ == "__main__":
    main()
