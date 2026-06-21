# Second-Order Pooling for Protein Language Models

**PP1 SoSe2026** — Technical University of Munich

---

## Motivation

Protein language models (pLMs) like ProtX produce rich per-residue embeddings, but most downstream tasks need a single fixed-size vector per protein. The standard approach is **mean pooling**: average all residue embeddings into one vector, then feed it to a probe head. It is simple and competitive, but has a structural blind spot — it averages each feature dimension independently, destroying information about feature **co-occurrence**.

**Covariance pooling** uses the second moment of residue activations rather than the first, capturing how features co-activate within a sequence and preserving combinatorial structure that mean pooling discards.

---

## Research Question

> Does covariance-based pooling of per-residue ProtX embeddings improve per-protein downstream task performance compared to mean pooling, and at what embedding size does it become parameter-efficient?

**Hypothesis:** Covariance pooling should preserve residue feature co-occurrence structure that mean pooling destroys, yielding improved downstream performance potentially at comparable or smaller embedding sizes than the original pLM hidden dimension.

---

## Method

Given per-residue embeddings **X** ∈ ℝ^{L×d} from a frozen pLM:

**Mean pooling (baseline):**  `μ = (1/L) Xᵀ 1_L  ∈ ℝ^d`

**Covariance pooling:** learn projections L, R ∈ ℝ^{d×d_c}, then compute  
`C = (1/L)(XL)ᵀ(XR)  ∈ ℝ^{d_c × d_c}` — flattened to d_c² features.

Two training regimes for the projections:
- **Supervised:** train L, R end-to-end with the probe head.
- **Unsupervised:** train L, R to reconstruct XᵀX, then freeze and reuse across tasks.

> Proper masking of padded positions is critical for both methods.

---

## Repository layout

```
.
├── train_subcellular_localization.py   # DeepLoc training entry point
├── train_meltome.py                    # Meltome regression training entry point
├── train_cov_unsup.py                  # Unsupervised covariance pretraining (PCA init + SGD)
├── inference.py / inference_meltome.py
├── solver.py                           # Localisation training/eval loop
├── solver_meltome.py                   # Meltome training/eval loop (Spearman R)
│
├── models/
│   ├── light_attention.py              # LightAttention (conv attention-weighted + max pool)
│   ├── light_attention_cov.py          # LightAttention + bilinear covariance branch
│   ├── pooling_ffn.py                  # PoolingFFN: mean / cov / hybrid pooling + FFN head
│   ├── ffn.py                          # FFN probe on reduced (sequence-wise) embeddings
│   └── loss_functions.py
│
├── configs/
│   ├── subcellular_localization/       # DeepLoc (localisation) sweep configs
│   │   ├── mean.yaml                   #   seed 969
│   │   ├── cov.yaml                    #   seed 123
│   │   ├── hybrid.yaml                 #   seed 123
│   │   ├── la.yaml                     #   seed 123  (plain LightAttention)
│   │   └── la_cov.yaml                 #   seed 123  (LightAttention + covariance)
│   └── meltome/                        # Meltome regression sweep configs
│       ├── mean.yaml / cov.yaml / hybrid.yaml
│       ├── la.yaml                     #   seed 123  (plain LightAttention)
│       └── la_cov.yaml                 #   seed 123  (LightAttention + covariance)
│
└── scripts/
    ├── data/                           # input generation
    │   ├── embed_layers_h5.py          #   ProtT5 H5 embeddings (all layers + final)
    │   ├── stack_layers.py             #   Concatenate per-layer H5 → *_layerStacked.h5
    │   └── prepare_flip_meltome_fastas.py  # FLIP CSV → train/val/test FASTAs
    ├── experiments/                    # run drivers
    │   ├── run_sweep.py                #   Pooling-method + d_c sweep (Exp 1 & 2)
    │   ├── run_cov_unsup_pretrain.py   #   Pretrain frozen projections for every d_c (Exp 2)
    │   ├── run_projection_comparison.py#   Part A: reconstruction across projection datasets (Exp 2)
    │   └── run_layer_sweep.py          #   Layer sweep — head-to-head at each layer (Exp 3)
    └── analysis/                       # results, figures, stats
        ├── collect_results.py          #   Parse run dirs → master results.csv (--runs-dir, layer column)
        ├── plot_dc.py                  #   d_c sweep figures (bars, d_c curves, head-to-head; --seeds)
        ├── plot_efficiency.py          #   Size-vs-performance curve (feature dim log scale + crossover)
        ├── plot_layers.py              #   Layer sweep figures: metric-vs-layer curve + method×layer heatmap
        ├── make_results_xlsx.py        #   results.csv → results.xlsx (one tab per task)
        ├── significance_tests.py       #   Per-seed tests + Fisher (McNemar / Wilcoxon / Williams)
        ├── extract_features.py         #   Dump test-set representations (random/untrained/mean/cov)
        ├── plot_tsne.py                #   4-panel UMAP/t-SNE of those representations + silhouette/kNN
        └── eval_run.py                 #   Re-evaluate a single run on demand
```

---

## Pooling methods

| Method | Extra params | Output dim | Notes |
|---|---|---|---|
| Mean pooling | 0 | d | baseline |
| Covariance (supervised) | 2·d·d_c | d_c² | L, R trained end-to-end |
| Covariance (unsupervised) | 2·d·d_c (frozen) | d_c² | L, R pretrained by reconstruction, then frozen (Exp 2) |
| Hybrid [μ; flat(C)] | 2·d·d_c | d + d_c² | concatenation |
| LightAttention | — | 2d | conv attention-weighted + max pool |
| LightAttentionCov | 2·d·d_c | 2d + d_c² | LA + covariance branch |

---

## Tasks

| Task | Type | Dataset | Metric | Train script |
|---|---|---|---|---|
| Subcellular localisation | 10-class classification | DeepLoc | Q10 accuracy | `train_subcellular_localization.py` |
| Meltome (thermostability) | Regression | FLIP Meltome | Spearman R | `train_meltome.py` |

---

## Setup

```bash
conda env create -f environment.yml
conda activate bio
```

---

## Data preparation

### DeepLoc (localisation)

```
data_files/deeploc_our_train_set.{h5,fasta}
data_files/deeploc_our_val_set.{h5,fasta}
data_files/deeploc_test_set.{h5,fasta}
```

The `*_remapped.fasta` files are written automatically by `embed_layers_h5.py`
when generating the embeddings, and are bundled with the H5 files on the project
Google Drive — download them there if missing.

### FLIP Meltome (regression)

```bash
# 1. Convert FLIP CSV → train/val/test FASTAs
python scripts/data/prepare_flip_meltome_fastas.py \
    --csv  data_files/flip_meltome/splits/human_cell.csv \
    --output-dir data_files/flip_meltome/prepared/human_cell \
    --prefix human_cell

# 2. Generate ProtT5 embeddings (one-time, GPU-intensive; --layers 24 = final layer)
python scripts/data/embed_layers_h5.py \
    --fasta data_files/flip_meltome/prepared/human_cell/human_cell_train.fasta \
            data_files/flip_meltome/prepared/human_cell/human_cell_val.fasta \
            data_files/flip_meltome/prepared/human_cell/human_cell_test.fasta \
    --layers 24
```

---

## Two separate experiment pipelines

```
PIPELINE A — Pooling method + d_c sweep (Experiments 1 & 2)

  embed_layers_h5.py --layers 24      final-layer H5 files (one-time)
          │
          ▼
  run_sweep.py --task meltome|loc     trains all (method, d_c) pairs
          │
          ▼
  collect_results.py                  parses run dirs → master results.csv
          │
          ▼
  plot_dc.py                          bars / d_c curves / head-to-head


PIPELINE B — Layer sweep (Experiment 3)

  embed_layers_h5.py --layers 6 12 18 24    per-layer H5 files (one-time)
          │
          ▼
  run_layer_sweep.py --task meltome|loc     trains all (layer, method) pairs
          │
          ▼
  collect_results.py → plot_layers.py       layer_curve_averaged.png
```

**Key difference:** Pipeline A always uses the same (final-layer) H5 files. Pipeline B requires separate per-layer H5 files generated by `embed_layers_h5.py`.

---

## Experiments

### Experiment 1 — d_c sweep (mean / cov / hybrid, 3 seeds, both tasks)

3 methods × 5 d_c values × 3 seeds × 2 tasks = 90 runs.  
(`mean` has no d_c so contributes 1 × 3 × 2 = 6 runs; `cov` and `hybrid` contribute 5 × 3 × 2 = 30 each.)

```bash
python scripts/experiments/run_sweep.py \
    --tasks loc meltome \
    --methods mean cov hybrid \
    --seeds 123 969 309 \
    --dcs 8 16 24 32 48
```

Collect and visualise:
```bash
python scripts/analysis/collect_results.py --sweep sweeps/<tag> --out results/results.csv
python scripts/analysis/plot_dc.py         --csv results/results.csv --out results/figures/dc_sweep --seeds 657 921 969
```

Outputs bars, d_c curves (metric vs d_c per method, averaged over seeds), head-to-head, and a summary table.

### Experiment 2 — Supervised vs. unsupervised bottleneck

Learn the projections `L, R` **without labels** by reconstructing the embedding
covariance, freeze them, and reuse across both tasks — then compare against the
per-task **supervised** bottleneck (Experiment 1).

The pretrainer is a **linear covariance autoencoder**: encode `C = Lᵀ M R`,
decode `M̂ = L C Rᵀ = (LLᵀ) M (RRᵀ)` (tied weights), trained on reconstruction
loss `‖M − M̂‖²_F` with `M = XᵀX`. Its global optimum is **PCA**, so we
initialize `L = R =` the top-`d_c` eigenvectors of the *average* covariance and
then refine with SGD (`--pca_init`).

**Step 1 — pretrain the projections once, on the union of all task train splits:**
```bash
# all d_c in one go (writes runs/cov_unsup_pretrained/cov_unsup_dc{dc}.pt)
python scripts/experiments/run_cov_unsup_pretrain.py --pca_init

# or a single d_c
python train_cov_unsup.py --config configs/cov_unsup_pretrain/union.yaml --proj_dim 48 --pca_init
```
`configs/cov_unsup_pretrain/union.yaml` lists the train-only data sources (DeepLoc +
Meltome). `--pca_only` saves the closed-form PCA solution without SGD refinement.

**Step 2 — train downstream probes with the frozen projections:**
```bash
python scripts/experiments/run_sweep.py --tasks loc --methods cov_unsup \
    --dcs 8 16 24 32 48 --seeds 657 921 969 \
    --cov_unsup_dir runs/cov_unsup_pretrained
```
`cov_unsup` loads `L, R` from the matching per-`d_c` checkpoint and freezes them
(`freeze_cov_projections: true`); only the probe head trains. Point
`--cov_unsup_dir` at `runs/cov_unsup_<task>` to use per-task (split) projections
instead of the shared union, for the ablation.

> Reconstruction quality (`rel_err = ‖M−M̂‖_F/‖M‖_F`) is a **diagnostic**, not the
> objective — the reported metric is downstream task performance.

### Experiment 3 — Layer sweep

Repeat the comparison at a handful of ProtT5 layers (early / middle / late / last) to test whether covariance pooling's advantage holds across model depth.

**Step 1 — generate per-layer embeddings (one-time):**
```bash
python scripts/data/embed_layers_h5.py \
    --fasta data_files/flip_meltome/prepared/human_cell/human_cell_train.fasta \
            data_files/flip_meltome/prepared/human_cell/human_cell_val.fasta \
            data_files/flip_meltome/prepared/human_cell/human_cell_test.fasta \
    --layers 6 12 18 24
```

**Step 2 — run the layer × method sweep:**
```bash
python scripts/experiments/run_layer_sweep.py \
    --train-fasta data_files/flip_meltome/prepared/human_cell/human_cell_train.fasta \
    --val-fasta   data_files/flip_meltome/prepared/human_cell/human_cell_val.fasta \
    --test-fasta  data_files/flip_meltome/prepared/human_cell/human_cell_test.fasta \
    --task meltome --methods mean cov hybrid --layers 6 12 18 24
```

**Step 3 — collect and plot:**
```bash
python scripts/analysis/collect_results.py --runs-dir runs --out results/results.csv
python scripts/analysis/plot_layers.py     --csv results/results.csv --out results/figures/layer_sweep --seeds 657 921 969
```

Produces `layer_curve_averaged_{loc,meltome}.png`: metric vs. layer depth, one line per pooling method (averaged over seeds).

---

## Evaluation metrics

| Task | Primary | Secondary |
|---|---|---|
| Localisation | Q10 accuracy | MCC, per-class F1 |
| Meltome | Spearman R | MSE, MAE |

All metrics include bootstrap standard errors (200 resamples, `--n-draws`).

---

## Statistical significance

Per-seed tests combined across seeds with **Fisher's method** (independent
experiments → combine p-values, not pool predictions):

```bash
# matched d_c across methods; mean has no d_c
python scripts/significance_tests.py --runs_root <dir> --dc 48
```
- DeepLoc: McNemar's exact test on per-protein correctness.
- Meltome: Wilcoxon signed-rank on |error| and Williams' test on Spearman R.

## Stretch goals

- Pool PaRTI (PageRank-based pooling)
- Binary membrane/soluble classification as a third task
