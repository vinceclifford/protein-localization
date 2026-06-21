# Results

Artifacts for the second-order pooling experiments. Everything derives from a
single master CSV; the figures are regenerated from it with the plotting scripts.

## Layout

```
results/
├── results.csv                  master: one row per trained head
│                                 (all groups, both tasks — see columns below)
├── results.xlsx                 derived human view: one tab per task (deeploc / meltome)
├── figures/
│   ├── dc_sweep/                 scripts/analysis/plot_dc.py + plot_efficiency.py output
│   ├── layer_sweep/              scripts/analysis/plot_layers.py output (curve + heatmap)
│   └── tsne_umap/                scripts/analysis/{extract_features,plot_tsne}.py output (UMAP + t-SNE)
├── reconstruction_summary.csv    Part A: projection-dataset reconstruction (rel_err)
└── significance/
    └── significance.txt          per-seed tests + Fisher combination across seeds
```

`results.csv` is the machine-readable source the plot scripts read; `results.xlsx`
is a convenience workbook (one sheet per task) built from it via
`scripts/analysis/make_results_xlsx.py` — regenerate it whenever `results.csv` changes.

## `results.csv`

One row per head run in `checkpoints/heads/`, with columns:

| column | meaning |
|---|---|
| `run_dir` | source run directory name |
| `task` | `loc` (DeepLoc) or `meltome` (FLIP human_cell) |
| `method` | `mean`, `cov`, `cov_unsup`, `hybrid` |
| `layer` | ProtT5 layer for the layer sweep (`06/12/18/24/Stacked`); **empty** for the d_c sweep |
| `proj_dim` | d_c (empty for `mean`) |
| `seed` | RNG seed |
| `test_acc`, `test_mcc`, `test_f1` (+ `_stderr`) | loc metrics |
| `test_spearman` (+ `_stderr`), `test_mse`, `test_mae` | meltome metrics |

The `layer` column is the discriminator: **empty ⇒ d_c sweep** (supervised +
unsupervised heads, plotted by `plot_dc.py`); **non-empty ⇒ layer sweep**
(plotted by `plot_layers.py`). Both scripts read this one file and self-filter.

## Regenerate

**Everything at once** (tables + all figures + significance + UMAP/t-SNE):

```bash
bash scripts/make_all_results.sh          # UMAP projections (pass 'tsne' for t-SNE)
```

Or step by step:

```bash
# 1. master CSV from the organized head checkpoints
python scripts/analysis/collect_results.py --runs-dir checkpoints/heads --out results/results.csv

# 2. d_c sweep figures (mean / cov / cov_unsup / hybrid × d_c), matched seeds
python scripts/analysis/plot_dc.py     --csv results/results.csv --out results/figures/dc_sweep    --seeds 657 921 969

# 3. layer sweep figures (mean / cov / hybrid × layer)
python scripts/analysis/plot_layers.py --csv results/results.csv --out results/figures/layer_sweep --seeds 657 921 969
```