# scripts

Command-line tools, grouped by role. Every script is standalone (`python
scripts/<group>/<name>.py --help`) and finds the repo root by walking up to the
`configs/`+`models/` markers, so it works regardless of where it's invoked from.

> Run these in a **Python 3.9+** environment with the project dependencies
> (torch, h5py, pandas, matplotlib, scikit-learn, scipy, pyyaml). The 2.7 `bio`
> conda env will **not** work.

## `data/` — input generation

| script | what it does |
|---|---|
| `embed_layers_h5.py` | ProtT5-XL per-residue embeddings → one H5 per layer (`--layers 6 12 18 24`); `--layers 24` = final layer. Also writes `*_remapped.fasta`. |
| `stack_layers.py` | Concatenate per-layer H5 files along the channel dim → `*_layerStacked.h5` (the "Stacked" layer-sweep input). |
| `prepare_flip_meltome_fastas.py` | FLIP Meltome CSV → train/val/test FASTAs. |

## `experiments/` — run drivers

These build per-run configs and launch `train_*.py` as subprocesses (so they
need the training env). Output lands in `runs/`.

| script | what it does |
|---|---|
| `run_sweep.py` | Pooling-method × d_c sweep (Exp 1 supervised, Exp 2 `cov_unsup`). |
| `run_cov_unsup_pretrain.py` | Pretrain the frozen covariance projections for every d_c on the union train split (Exp 2). |
| `run_projection_comparison.py` | Part A: reconstruction `rel_err` across projection datasets (union/deeploc/meltome) × seeds at d_c=48. |
| `run_layer_sweep.py` | Layer × method head-to-head at each ProtT5 layer (Exp 3). |

## `analysis/` — results, figures, stats

Read `runs/` (or `checkpoints/heads/`) and produce everything in `results/`.

| script | what it does |
|---|---|
| `collect_results.py` | Parse run dirs → master `results.csv` (`--runs-dir`, parses the `layer` token). |
| `plot_dc.py` | d_c sweep figures (bars, d_c curves, head-to-head). Ignores layer-sweep rows. |
| `plot_layers.py` | Layer sweep figures (metric vs layer). Uses only layer-sweep rows. |
| `make_results_xlsx.py` | `results.csv` → `results.xlsx` with one tab per task. |
| `significance_tests.py` | Per-seed McNemar / Wilcoxon / Williams + Fisher combination. |
| `extract_features.py` | Dump test-set representations (random / untrained-cov / mean / trained-cov) as `features_*.npy`. |
| `plot_tsne.py` | 4-panel UMAP/t-SNE of those representations with silhouette + 5-NN metrics. |
| `eval_run.py` | Re-run the bootstrap test evaluation for one existing run. |

## Typical flows

**Pooling / d_c sweep (Exp 1 & 2)**
```bash
python scripts/experiments/run_sweep.py --tasks loc meltome --methods mean cov hybrid --dcs 8 16 24 32 48 --seeds 657 921 969
python scripts/analysis/collect_results.py --runs-dir runs --out results/results.csv
python scripts/analysis/plot_dc.py --csv results/results.csv --out results/figures/dc_sweep --seeds 657 921 969
```

**Layer sweep (Exp 3)**
```bash
python scripts/data/embed_layers_h5.py --fasta <train> <val> <test> --layers 6 12 18 24
python scripts/data/stack_layers.py --task loc --layers 6 12 18 24          # for the Stacked cell
python scripts/experiments/run_layer_sweep.py --task loc --methods mean cov hybrid
python scripts/analysis/plot_layers.py --csv results/results.csv --out results/figures/layer_sweep --seeds 657 921 969
```

**Representation t-SNE (visualisation)**
```bash
python scripts/analysis/extract_features.py --checkpoint <cov_dc48 run>/checkpoint.pt \
    --test-h5 data_files/deeploc_test_set_layer24.h5 \
    --test-fasta data_files/deeploc_test_set_remapped.fasta \
    --out-dir features/loc --proj-dim 48
python scripts/analysis/plot_tsne.py --features-dir features/loc --method tsne
```
