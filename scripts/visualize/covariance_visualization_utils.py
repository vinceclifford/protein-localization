#!/usr/bin/env python3
"""Shared model, covariance, sequence, and plotting utilities."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import models  # noqa: E402


AA_GROUPS = [
    ("Positive charged", "KRH"),
    ("Polar uncharged", "STNQC"),
    ("Negative charged", "DE"),
    ("Hydrophobic aliphatic", "AVILM"),
    ("Aromatic", "FWY"),
    ("Special flexible", "GP"),
]
AA_ORDER = list("KRHSTNQCDEAVILMFWYGP")


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def resolve_checkpoint(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.is_dir():
        run_dir = path
        checkpoint = path / "checkpoint.pt"
    else:
        run_dir = path.parent
        checkpoint = path
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not (run_dir / "train_arguments.yaml").exists():
        raise FileNotFoundError(f"Missing train_arguments.yaml in {run_dir}")
    return run_dir, checkpoint


def load_covariance_model(checkpoint_path: Path, embedding_dim: int, device: torch.device):
    run_dir, checkpoint_file = resolve_checkpoint(checkpoint_path)
    train_args = load_yaml(run_dir / "train_arguments.yaml")
    model_cls = getattr(models, train_args["model_type"])
    model = model_cls(
        embeddings_dim=embedding_dim,
        **(train_args.get("model_parameters") or {}),
    )
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    if not ((hasattr(model, "proj_L") and hasattr(model, "proj_R")) or hasattr(model, "seq_proj")):
        raise RuntimeError("The checkpoint does not use cov/cov_n pooling.")
    return model, train_args, run_dir


def read_fasta(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    header = None
    parts: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append({"id": header.split()[0], "description": header, "sequence": "".join(parts)})
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        records.append({"id": header.split()[0], "description": header, "sequence": "".join(parts)})
    return records


def key_from_record(record: dict[str, str], key_format: str) -> str:
    if key_format == "hash":
        return record["id"]
    if key_format == "fasta_descriptor":
        return record["description"].replace(".", "_").replace("/", "_")
    if key_format == "fasta_descriptor_old":
        return record["description"]
    raise ValueError(f"Unsupported key_format: {key_format}")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


@torch.no_grad()
def covariance_analysis(model: torch.nn.Module, embedding: torch.Tensor) -> dict:
    """Return the compressed matrix and ingredients for per-pair attribution."""
    x = embedding.float()
    if hasattr(model, "proj_L") and hasattr(model, "proj_R"):
        left = model.proj_L(x)
        right = model.proj_R(x)
        matrix = left.T @ right / max(len(x), 1)
        return {
            "kind": "cov",
            "matrix": matrix.detach().cpu().numpy(),
            "left": left.detach().cpu().numpy(),
            "right": right.detach().cpu().numpy(),
        }

    logits = model.seq_proj(x)
    assignment = torch.softmax(logits.T, dim=-1)
    slots = assignment @ x
    matrix = slots @ slots.T / max(int(slots.shape[-1]), 1)
    return {
        "kind": "cov_n",
        "matrix": matrix.detach().cpu().numpy(),
        "embedding": x.detach().cpu().numpy(),
        "assignment": assignment.detach().cpu().numpy(),
        "slots": slots.detach().cpu().numpy(),
    }


def select_pairs(analysis: dict, top_k: int, sign: str = "positive") -> list[dict]:
    matrix = analysis["matrix"]
    flat = matrix.ravel()
    if sign == "positive":
        candidates = np.flatnonzero(flat > 0)
        order = candidates[np.argsort(flat[candidates])[::-1]]
    elif sign == "negative":
        candidates = np.flatnonzero(flat < 0)
        order = candidates[np.argsort(flat[candidates])]
    else:
        candidates = np.arange(flat.size)
        order = candidates[np.argsort(np.abs(flat))[::-1]]
    if not len(order):
        raise ValueError(f"No {sign} covariance entries found")
    rows = []
    for rank, flat_idx in enumerate(order[:top_k], 1):
        left_dim, right_dim = np.unravel_index(int(flat_idx), matrix.shape)
        contribution = pair_contribution(analysis, left_dim, right_dim)
        rows.append({
            "rank": rank,
            "left_dim": int(left_dim),
            "right_dim": int(right_dim),
            "C_value": float(matrix[left_dim, right_dim]),
            "contribution": contribution,
        })
    return rows


def pair_contribution(analysis: dict, left_dim: int, right_dim: int) -> np.ndarray:
    if analysis["kind"] == "cov":
        return analysis["left"][:, left_dim] * analysis["right"][:, right_dim] / len(analysis["left"])
    x = analysis["embedding"]
    assignment = analysis["assignment"]
    slots = analysis["slots"]
    d = max(x.shape[1], 1)
    left = assignment[left_dim] * (x @ slots[right_dim]) / d
    right = assignment[right_dim] * (x @ slots[left_dim]) / d
    return 0.5 * (left + right)


def combined_residue_scores(pairs: list[dict]) -> np.ndarray:
    return np.sum([np.abs(pair["contribution"]) for pair in pairs], axis=0)


def plot_covariance_matrix(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    selected_pairs: list[dict] | None = None,
) -> None:
    vmax = max(float(np.nanmax(np.abs(matrix))), 1e-12)
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_title(title)
    ax.set_xlabel("right projection dimension")
    ax.set_ylabel("left projection dimension")
    for pair in selected_pairs or []:
        ax.scatter(pair["right_dim"], pair["left_dim"], s=95, facecolors="none",
                   edgecolors="black", linewidths=1.4)
        ax.text(pair["right_dim"] + 0.35, pair["left_dim"] - 0.35, str(pair["rank"]),
                fontsize=7, fontweight="bold", color="black")
    fig.colorbar(image, ax=ax, label="covariance entry C")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_significant_pairs(pairs: list[dict], out_path: Path) -> None:
    labels = [f"#{pair['rank']}  L{pair['left_dim']} x R{pair['right_dim']}" for pair in pairs]
    values = np.asarray([pair["C_value"] for pair in pairs], dtype=float)
    colors = np.where(values >= 0, "#d1495b", "#277da1")
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.48 * len(pairs) + 1.4)), constrained_layout=True)
    y = np.arange(len(pairs))
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="#202020", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("covariance entry C")
    ax.set_title("Selected significant covariance entries")
    for row, value in enumerate(values):
        ax.text(value, row, f" {value:+.4g}" if value >= 0 else f"{value:+.4g} ",
                ha="left" if value >= 0 else "right", va="center", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pair_windows(
    sequence: str,
    pair: dict,
    out_path: Path,
    top_positions: int,
    radius: int,
    title_prefix: str,
) -> pd.DataFrame:
    contribution = np.asarray(pair["contribution"], dtype=float)
    order = np.argsort(np.abs(contribution))[::-1][:top_positions]
    vmax = max(float(np.max(np.abs(contribution))), 1e-12)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(
        len(order), 1, figsize=(13.5, max(2.5, 0.62 * len(order) + 1.3)),
        squeeze=False, constrained_layout=True,
    )
    rows = []
    for rank, (ax, center) in enumerate(zip(axes[:, 0], order), 1):
        start, end = max(0, center - radius), min(len(sequence), center + radius + 1)
        values = contribution[start:end]
        positions = np.arange(start + 1, end + 1)
        aas = list(sequence[start:end])
        ax.imshow(values[None, :], cmap="coolwarm", norm=norm, aspect="auto")
        ax.set_yticks([])
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels([f"{p}{aa}" for p, aa in zip(positions, aas)], rotation=90, fontsize=7)
        center_local = center - start
        ax.scatter(center_local, 0, s=125, facecolors="none", edgecolors="black", linewidths=1.2)
        for idx, (aa, value) in enumerate(zip(aas, values)):
            ax.text(idx, 0, aa, ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.55 * vmax else "black", fontweight="bold")
        ax.set_ylabel(f"{center + 1}{sequence[center]}\n{contribution[center]:+.2g}", rotation=0,
                      ha="right", va="center", fontsize=8, labelpad=22)
        rows.append({
            "position_rank": rank,
            "position": int(center + 1),
            "aa": sequence[center],
            "contribution": float(contribution[center]),
            "abs_contribution": float(abs(contribution[center])),
            "window": sequence[start:end],
            "window_start": start + 1,
            "window_end": end,
        })
    fig.suptitle(
        f"{title_prefix} | L{pair['left_dim']} x R{pair['right_dim']} | C={pair['C_value']:+.4g}",
        y=1.02,
    )
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=axes[:, 0],
                 fraction=0.025, pad=0.02, label="contribution to C")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def infer_fasta(h5_path: Path, train_args: dict, run_dir: Path) -> Path:
    target = h5_path.resolve()
    for split in ("train", "val", "test"):
        emb_value = train_args.get(f"{split}_embeddings")
        fasta_value = train_args.get(f"{split}_remapping")
        if not emb_value or not fasta_value:
            continue
        emb_path = Path(emb_value)
        emb_path = emb_path if emb_path.is_absolute() else PROJECT_ROOT / emb_path
        if emb_path.resolve() == target or emb_path.name == target.name:
            fasta = Path(fasta_value)
            return fasta if fasta.is_absolute() else PROJECT_ROOT / fasta
    candidates = [
        h5_path.with_name(f"{h5_path.stem}_remapped.fasta"),
        h5_path.with_suffix(".fasta"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not infer the remapping FASTA; pass --fasta explicitly")


def infer_class(description: str) -> str:
    known = [
        "Cytoplasm-Nucleus", "Cell.membrane", "Cytoplasm", "Endoplasmic.reticulum",
        "Extracellular", "Golgi.apparatus", "Lysosome/Vacuole", "Mitochondrion",
        "Nucleus", "Peroxisome", "Plastid",
    ]
    for label in known:
        if re.search(rf"(?:^|\s){re.escape(label)}(?:-[A-Za-z]+)?(?:\s|$)", description):
            return label
    return "All"


def infer_accession(description: str) -> str:
    pattern = r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z0-9]{3}[0-9]){1,2}"
    match = re.search(rf"(?:^|[|\s])({pattern})(?:[|\s]|$)", description)
    return match.group(1) if match else ""


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "sample"
