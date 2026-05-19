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
    ├── embed_bio_embeddings_h5.py      # Generate ProtX H5 embeddings — final layer
    ├── embed_layers_h5.py              # Generate H5 files for N transformer layers
    ├── prepare_flip_meltome_fastas.py  # FLIP CSV → train/val/test FASTAs
    ├── run_sweep.py                    # Pooling-method + d_c sweep (Exp 1 & 2)
    ├── run_layer_sweep.py              # Layer sweep — head-to-head at each layer (Exp 4)
    ├── collect_results.py              # Parse run dirs → results.csv
    ├── plot_sweep.py                   # Visualise pooling sweep results
    └── plot_layer_sweep.py             # Visualise layer × method grid
```

---

## Pooling methods

| Method | Extra params | Output dim | Notes |
|---|---|---|---|
| Mean pooling | 0 | d | baseline |
| Covariance (supervised) | 2·d·d_c | d_c² | L, R trained end-to-end |
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

If FASTA files are missing, regenerate from H5 keys:
```bash
python scripts/regen_fasta_from_h5.py
```

### FLIP Meltome (regression)

```bash
# 1. Convert FLIP CSV → train/val/test FASTAs
python scripts/prepare_flip_meltome_fastas.py \
    --csv  data_files/flip_meltome/splits/human_cell.csv \
    --output-dir data_files/flip_meltome/prepared/human_cell \
    --prefix human_cell

# 2. Generate final-layer ProtX embeddings (one-time, GPU-intensive)
python scripts/embed_bio_embeddings_h5.py \
    data_files/flip_meltome/prepared/human_cell/human_cell_train.fasta \
    data_files/flip_meltome/prepared/human_cell/human_cell_val.fasta \
    data_files/flip_meltome/prepared/human_cell/human_cell_test.fasta
```

---

## Two separate experiment pipelines

```
PIPELINE A — Pooling method + d_c sweep (Experiments 1 & 2)

  embed_bio_embeddings_h5.py          final-layer H5 files (one-time)
          │
          ▼
  run_sweep.py --task meltome|loc     trains all (method, d_c) pairs
          │
          ▼
  collect_results.py                  parses run dirs → results.csv
          │
          ▼
  plot_sweep.py                       sweep_dc_curve.png + sweep_bars.png


PIPELINE B — Layer sweep (Experiment 3)

  embed_layers_h5.py --layers 6 12 18 24    per-layer H5 files (one-time)
          │
          ▼
  run_layer_sweep.py --task meltome|loc     trains all (layer, method) pairs
          │
          ▼
  plot_layer_sweep.py                       layer_curve.png
```

**Key difference:** Pipeline A always uses the same (final-layer) H5 files. Pipeline B requires separate per-layer H5 files generated by `embed_layers_h5.py`.

---

## Experiments

### Experiment 1 — d_c sweep (mean / cov / hybrid, 3 seeds, both tasks)

3 methods × 5 d_c values × 3 seeds × 2 tasks = 90 runs.  
(`mean` has no d_c so contributes 1 × 3 × 2 = 6 runs; `cov` and `hybrid` contribute 5 × 3 × 2 = 30 each.)

```bash
# Run all three seeds for each task
python scripts/run_sweep.py --task loc     --methods mean cov hybrid --seed 123 --dcs 8 16 24 32 48
python scripts/run_sweep.py --task loc     --methods mean cov hybrid --seed 969 --dcs 8 16 24 32 48
python scripts/run_sweep.py --task loc     --methods mean cov hybrid --seed 309 --dcs 8 16 24 32 48

python scripts/run_sweep.py --task meltome --methods mean cov hybrid --seed 123 --dcs 8 16 24 32 48
python scripts/run_sweep.py --task meltome --methods mean cov hybrid --seed 969 --dcs 8 16 24 32 48
python scripts/run_sweep.py --task meltome --methods mean cov hybrid --seed 309 --dcs 8 16 24 32 48
```

Collect and visualise:
```bash
python scripts/collect_results.py --sweep sweeps/<tag>
python scripts/plot_sweep.py      --sweep sweeps/<tag>
```

Outputs `sweep_dc_curve.png` (metric vs d_c² per method, averaged over seeds) and `sweep_bars.png`.

### Experiment 2 — Supervised vs. unsupervised bottleneck

Train L, R to reconstruct XᵀX (unsupervised), freeze them, then reuse across tasks. Compare against per-task supervised bottlenecks.

> Not yet implemented.

### Experiment 3 — Layer sweep

Repeat the comparison at a handful of ProtT5 layers (early / middle / late / last) to test whether covariance pooling's advantage holds across model depth.

**Step 1 — generate per-layer embeddings (one-time):**
```bash
python scripts/embed_layers_h5.py \
    --fasta data_files/flip_meltome/prepared/human_cell/human_cell_train.fasta \
            data_files/flip_meltome/prepared/human_cell/human_cell_val.fasta \
            data_files/flip_meltome/prepared/human_cell/human_cell_test.fasta \
    --layers 6 12 18 24
```

**Step 2 — run the layer × method sweep:**
```bash
python scripts/run_layer_sweep.py \
    --train-fasta data_files/flip_meltome/prepared/human_cell/human_cell_train.fasta \
    --val-fasta   data_files/flip_meltome/prepared/human_cell/human_cell_val.fasta \
    --test-fasta  data_files/flip_meltome/prepared/human_cell/human_cell_test.fasta \
    --task meltome --methods mean cov hybrid --layers 6 12 18 24
```

**Step 3 — plot:**
```bash
python scripts/plot_layer_sweep.py --csv sweeps/layer_<tag>/layer_results.csv
```

Produces `layer_curve.png`: metric vs. layer depth, one line per pooling method.

---

## Evaluation metrics

| Task | Primary | Secondary |
|---|---|---|
| Localisation | Q10 accuracy | MCC, per-class F1 |
| Meltome | Spearman R | MSE, MAE |

All metrics include bootstrap standard errors (200 resamples, `--n-draws`).

---

## Stretch goals

- Unsupervised covariance bottleneck (Experiment 2)
- Pool PaRTI (PageRank-based pooling)
- Binary membrane/soluble classification as a third task
