#!/usr/bin/env python3
"""Create dataset-level AA enrichment and UniProt feature-alignment heatmaps."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_deeploc_uniprot_site_overlap import (  # noqa: E402
    FEATURE_SCOPE_LABELS,
    INDIVIDUAL_FUNCTIONAL_REGION_TYPE_SCOPES,
    INDIVIDUAL_OFFICIAL_TYPE_SCOPES,
    THRESHOLDS,
    aggregate_by_class,
    aggregate_overall,
    fetch_all_uniprot,
    masks_from_features,
    parse_features,
    threshold_metrics,
)
from scripts.visualize.covariance_visualization_utils import (  # noqa: E402
    AA_GROUPS,
    AA_ORDER,
    combined_residue_scores,
    covariance_analysis,
    infer_accession,
    infer_class,
    infer_fasta,
    key_from_record,
    load_covariance_model,
    read_fasta,
    resolve_checkpoint,
    resolve_device,
    select_pairs,
)


def load_accession_mapping(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    accession_col = next(
        (c for c in ("accession", "uniprot_accession", "uniprot_id", "Entry") if c in frame),
        None,
    )
    key_col = next((c for c in ("h5_key", "key", "record_id", "fasta_id", "id") if c in frame), None)
    if accession_col is None or key_col is None:
        raise ValueError("--mapping-csv needs an accession column and an H5-key column")
    return dict(zip(frame[key_col].astype(str), frame[accession_col].astype(str)))


def centered_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    vmax = max(1.25, float(np.max(finite))) if finite.size else 2.0
    return TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)


def add_aa_group_axis(ax: plt.Axes) -> None:
    ax.set_xticks(np.arange(len(AA_ORDER)))
    ax.set_xticklabels(AA_ORDER, fontsize=10)
    offset = 0
    for group, aas in AA_GROUPS:
        end = offset + len(aas)
        ax.text((offset + end - 1) / 2, -0.25, group.replace(" ", "\n"), ha="center", va="top",
                fontsize=8, transform=ax.get_xaxis_transform(), clip_on=False)
        if end < len(AA_ORDER):
            ax.axvline(end - 0.5, color="white", linewidth=3.0)
            ax.axvline(end - 0.5, color="#202020", linewidth=0.9)
        offset = end


def plot_fold_heatmap(
    frame: pd.DataFrame,
    row_col: str,
    value_col: str,
    columns: list[str],
    title: str,
    out_path: Path,
    grouped_aa_axis: bool = False,
) -> None:
    rows = list(dict.fromkeys(frame[row_col].tolist()))
    pivot = frame.pivot(index=row_col, columns="item", values=value_col).reindex(index=rows, columns=columns)
    values = pivot.to_numpy(dtype=float)
    fig_width = max(10.0, 1.0 + 0.72 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_width, max(3.2, 0.62 * len(rows) + 2.0)), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=centered_norm(values), aspect="auto")
    ax.set_title(title)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_ylabel("class" if not (len(rows) == 1 and rows[0] == "All") else "")
    if grouped_aa_axis:
        add_aa_group_axis(ax)
        ax.set_xlabel("amino-acid type grouped by chemical class", labelpad=46)
    else:
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=30, ha="right")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}x", ha="center", va="center", fontsize=7.5)
    fig.colorbar(image, ax=ax, label="fold enrichment vs sequence abundance")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_aa(accumulators: dict[str, dict], groups: bool = False) -> pd.DataFrame:
    rows = []
    for class_label, payload in accumulators.items():
        sequence_counts = payload["sequence_counts"]
        contribution = payload["contribution"]
        total_count = sum(sequence_counts.values())
        total_contribution = sum(contribution.values())
        definitions = AA_GROUPS if groups else [(aa, aa) for aa in AA_ORDER]
        for label, members in definitions:
            count = sum(sequence_counts[aa] for aa in members)
            score = sum(contribution[aa] for aa in members)
            count_share = count / total_count if total_count else np.nan
            score_share = score / total_contribution if total_contribution else np.nan
            rows.append({
                "class_label": class_label,
                "item": label,
                "members": members,
                "count": count,
                "abs_contribution": score,
                "count_share": count_share,
                "contribution_share": score_share,
                "fold_enrichment": score_share / count_share if count_share > 0 else np.nan,
                "n_proteins": len(payload["proteins"]),
            })
    return pd.DataFrame(rows)


def plot_feature_overall(overall: pd.DataFrame, scopes: list[str], out_path: Path, title: str) -> None:
    available = [scope for scope in scopes if scope in set(overall.loc[overall.n_proteins_with_annotations > 0, "feature_scope"])]
    if not available:
        return
    pivot = overall.pivot(index="feature_scope", columns="threshold_pct", values="enrichment_vs_background").reindex(
        index=available, columns=THRESHOLDS
    )
    proteins = overall.pivot(index="feature_scope", columns="threshold_pct", values="n_proteins_with_annotations").reindex(
        index=available, columns=THRESHOLDS
    )
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.48 * len(available))), constrained_layout=True)
    image = ax.imshow(values, cmap="coolwarm", norm=centered_norm(values), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("top residue threshold (%)")
    ax.set_ylabel("UniProt feature type")
    ax.set_xticks(np.arange(len(THRESHOLDS)))
    ax.set_xticklabels(THRESHOLDS)
    ax.set_yticks(np.arange(len(available)))
    ax.set_yticklabels([FEATURE_SCOPE_LABELS[scope] for scope in available])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}x\n(P={int(proteins.iloc[i, j]):,})",
                        ha="center", va="center", fontsize=6.6)
    fig.colorbar(image, ax=ax, label="enrichment vs background")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_class(
    class_stats: pd.DataFrame,
    scopes: list[str],
    threshold: int,
    out_path: Path,
    title: str,
) -> None:
    data = class_stats[class_stats.threshold_pct == threshold]
    available = [scope for scope in scopes if scope in set(data.loc[data.n_proteins_with_annotations > 0, "feature_scope"])]
    if not available:
        return
    classes = list(dict.fromkeys(data.class_label.tolist()))
    pivot = data.pivot(index="class_label", columns="feature_scope", values="enrichment_vs_background").reindex(
        index=classes, columns=available
    )
    proteins = data.pivot(index="class_label", columns="feature_scope", values="n_proteins_with_annotations").reindex(
        index=classes, columns=available
    )
    values = pivot.to_numpy(dtype=float)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(max(14, 2 + 0.85 * len(available)), max(3.8, 0.52 * len(classes) + 1.8)),
                           constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=centered_norm(values), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("UniProt feature type")
    ax.set_ylabel("class")
    ax.set_xticks(np.arange(len(available)))
    ax.set_xticklabels([FEATURE_SCOPE_LABELS[scope] for scope in available], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}x\n(P={int(proteins.iloc[i, j]):,})",
                        ha="center", va="center", fontsize=6.2)
    fig.colorbar(image, ax=ax, label="enrichment vs background")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Run directory or checkpoint.pt")
    parser.add_argument("dataset_h5", type=Path, help="Per-residue embedding H5")
    parser.add_argument("--fasta", type=Path, help="Remapping FASTA; inferred from checkpoint by default")
    parser.add_argument("--mapping-csv", type=Path, help="Optional H5-key to UniProt-accession mapping")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--aa-top-pairs", type=int, default=1)
    parser.add_argument("--feature-top-pairs", type=int, default=5)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--uniprot-cache", type=Path, default=PROJECT_ROOT / "figures" / "unified_uniprot_cache")
    parser.add_argument("--cache-fallback-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures" / "unified_dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_dir, _ = resolve_checkpoint(args.checkpoint)
    from scripts.visualize.covariance_visualization_utils import load_yaml
    train_args = load_yaml(run_dir / "train_arguments.yaml")
    fasta_path = args.fasta or infer_fasta(args.dataset_h5, train_args, run_dir)
    records = read_fasta(fasta_path)
    if args.sample_limit:
        records = records[:args.sample_limit]
    key_format = train_args.get("key_format", "hash")
    mapping = load_accession_mapping(args.mapping_csv)
    for record in records:
        record["h5_key"] = key_from_record(record, key_format)
        record["class_label"] = infer_class(record["description"])
        description_parts = record["description"].split()
        original_fasta_id = description_parts[1] if len(description_parts) > 1 else ""
        record["accession"] = mapping.get(
            record["h5_key"],
            mapping.get(original_fasta_id, infer_accession(record["description"])),
        )

    accessions = [record["accession"] for record in records if record["accession"]]
    uniprot = fetch_all_uniprot(
        accessions, args.uniprot_cache, args.workers, cache_only=args.cache_only,
        fallback_dirs=args.cache_fallback_dir,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset_h5, "r") as h5:
        first_record = next(record for record in records if record["h5_key"] in h5)
        embedding_dim = h5[first_record["h5_key"]].shape[1]
    model, _, _ = load_covariance_model(args.checkpoint, embedding_dim, device)

    aa_acc = defaultdict(lambda: {
        "sequence_counts": defaultdict(float), "contribution": defaultdict(float), "proteins": set()
    })
    threshold_rows = []
    skipped = []
    max_pairs = max(args.aa_top_pairs, args.feature_top_pairs)
    scopes = list(INDIVIDUAL_OFFICIAL_TYPE_SCOPES)
    analysis_scopes = list(dict.fromkeys(scopes + list(INDIVIDUAL_FUNCTIONAL_REGION_TYPE_SCOPES)))

    with h5py.File(args.dataset_h5, "r") as h5:
        for index, record in enumerate(records):
            key = record["h5_key"]
            if key not in h5:
                skipped.append({"index": index, "h5_key": key, "reason": "missing_h5_key"})
                continue
            embedding = torch.tensor(h5[key][:]).float().to(device)
            sequence = record["sequence"][:len(embedding)]
            if len(sequence) != len(embedding):
                skipped.append({"index": index, "h5_key": key, "reason": "sequence_embedding_length_mismatch"})
                continue
            try:
                analysis = covariance_analysis(model, embedding)
                pairs = select_pairs(analysis, max_pairs, "positive")
            except ValueError as exc:
                skipped.append({"index": index, "h5_key": key, "reason": str(exc)})
                continue

            aa_score = combined_residue_scores(pairs[:args.aa_top_pairs])
            for class_label in {"All", record["class_label"]}:
                payload = aa_acc[class_label]
                payload["proteins"].add(key)
                for aa, value in zip(sequence, aa_score):
                    if aa in AA_ORDER:
                        payload["sequence_counts"][aa] += 1
                        payload["contribution"][aa] += float(abs(value))

            data = uniprot.get(record["accession"])
            if data is not None:
                features, _ = parse_features(data, sequence)
                masks = masks_from_features(len(sequence), features)
                feature_score = combined_residue_scores(pairs[:args.feature_top_pairs])
                for scope in analysis_scopes:
                    mask = masks[scope]
                    for threshold in THRESHOLDS:
                        row = threshold_metrics(feature_score, mask, threshold)
                        row.update({
                            "sample_index": index,
                            "h5_key": key,
                            "accession": record["accession"],
                            "class_label": record["class_label"],
                            "feature_scope": scope,
                        })
                        threshold_rows.append(row)
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"Processed {index + 1}/{len(records)}", flush=True)

    aa_stats = summarize_aa(aa_acc, groups=False)
    aa_group_stats = summarize_aa(aa_acc, groups=True)
    aa_stats.to_csv(args.out_dir / "aa_enrichment.csv", index=False)
    aa_group_stats.to_csv(args.out_dir / "aa_group_enrichment.csv", index=False)
    class_order = [label for label in aa_stats.class_label.unique() if label != "All"]
    if not class_order:
        class_order = ["All"]
    aa_plot = aa_stats[aa_stats.class_label.isin(class_order)].copy()
    group_plot = aa_group_stats[aa_group_stats.class_label.isin(class_order)].copy()
    plot_fold_heatmap(aa_plot, "class_label", "fold_enrichment", AA_ORDER,
                      "Positive covariance-pair amino-acid enrichment by class",
                      args.out_dir / "aa_enrichment_by_class_heatmap.png", grouped_aa_axis=True)
    plot_fold_heatmap(group_plot, "class_label", "fold_enrichment", [name for name, _ in AA_GROUPS],
                      "Positive covariance-pair amino-acid-group enrichment by class",
                      args.out_dir / "aa_group_enrichment_by_class_heatmap.png")

    per_threshold = pd.DataFrame(threshold_rows)
    if not per_threshold.empty:
        per_threshold.to_csv(args.out_dir / "uniprot_feature_overlap_per_protein.csv", index=False)
        overall = aggregate_overall(per_threshold)
        by_class = aggregate_by_class(per_threshold)
        overall.to_csv(args.out_dir / "uniprot_feature_overlap_overall.csv", index=False)
        by_class.to_csv(args.out_dir / "uniprot_feature_overlap_by_class.csv", index=False)
        plot_feature_overall(overall, scopes, args.out_dir / "official_feature_type_enrichment_heatmap.png",
                             "Top-residue enrichment by official UniProt feature type")
        plot_feature_overall(overall, list(INDIVIDUAL_FUNCTIONAL_REGION_TYPE_SCOPES),
                             args.out_dir / "functional_region_type_enrichment_heatmap.png",
                             "Top-residue enrichment by functional-region type")
        for threshold in (1, 5):
            plot_feature_class(by_class, scopes, threshold,
                               args.out_dir / f"official_feature_type_class_enrichment_top{threshold}.png",
                               f"Top {threshold}% residue enrichment by class and official UniProt feature type")
            plot_feature_class(by_class, list(INDIVIDUAL_FUNCTIONAL_REGION_TYPE_SCOPES), threshold,
                               args.out_dir / f"functional_region_type_class_enrichment_top{threshold}.png",
                               f"Top {threshold}% residue enrichment by class and functional-region type")
    else:
        print("No UniProt feature rows were available; AA plots were still generated.")

    pd.DataFrame(skipped).to_csv(args.out_dir / "skipped_records.csv", index=False)
    print(f"Saved dataset visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
