#!/usr/bin/env bash
#
# Regenerate every results table and figure from the trained head checkpoints.
# Run from anywhere, in the Python 3 env with the project deps (e.g. `conda activate base`):
#
#     bash scripts/make_all_results.sh
#
# Steps 1-6 only need checkpoints/heads + results.csv. Step 7 (t-SNE/UMAP) also
# needs torch and the test embeddings/FASTAs in data_files/ — it's last so a
# missing dependency there doesn't block the rest.

cd "$(dirname "$0")/.."          # repo root
set -u

SEEDS="657 921 969"
HEADS="checkpoints/heads"
CSV="results/results.csv"
A="scripts/analysis"
# Use the active conda env's interpreter (inherited via CONDA_PREFIX), so this
# works even when a `bash` subshell's PATH puts a system python2/python3 first.
PY="${CONDA_PREFIX:+$CONDA_PREFIX/bin/}python"

echo "== 1/7  master results.csv =="
"$PY" "$A/collect_results.py" --runs-dir "$HEADS" --out "$CSV"

echo "== 2/7  d_c sweep figures (bars, curves, head-to-head all d_c, scatter) =="
"$PY" "$A/plot_dc.py" --csv "$CSV" --out results/figures/dc_sweep \
    --seeds $SEEDS --paper-loc 82 --paper-meltome 0.708

echo "== 3/7  efficiency curve =="
"$PY" "$A/plot_efficiency.py" --csv "$CSV" --out results/figures/dc_sweep --seeds $SEEDS

echo "== 4/7  layer sweep (curve + heatmap) =="
"$PY" "$A/plot_layers.py" --csv "$CSV" --out results/figures/layer_sweep --seeds $SEEDS

echo "== 5/7  results.xlsx (one tab per task) =="
"$PY" "$A/make_results_xlsx.py" --csv "$CSV"

echo "== 6/7  significance tests =="
mkdir -p results/significance
"$PY" "$A/significance_tests.py" --runs_root "$HEADS" --dc 48 | tee results/significance/significance.txt

echo "== 7/7  representation projections (UMAP) — needs torch + test embeddings =="
mkdir -p results/figures/tsne_umap
METHOD="${1:-umap}"               # pass 'tsne' as arg 1 to use t-SNE instead
for S in $SEEDS; do
  CK=$(ls -d "$HEADS"/supervised/deeploc/PoolingFFN_loc_cov_dc48_seed${S}_*/ 2>/dev/null | head -1)
  "$PY" "$A/extract_features.py" --checkpoint "${CK}checkpoint.pt" \
      --test-h5 data_files/deeploc_test_set_layer24.h5 \
      --test-fasta data_files/deeploc_test_set_remapped.fasta \
      --task loc --out-dir features/loc_seed${S} --proj-dim 48 --seed ${S} &&
  "$PY" "$A/plot_tsne.py" --features-dir features/loc_seed${S} --method "$METHOD" \
      --task-name "DeepLoc setDeepLoc (seed ${S})" \
      --out results/figures/tsne_umap/deeploc_${METHOD}_seed${S}.png

  CK=$(ls -d "$HEADS"/supervised/meltome/PoolingFFN_meltome_cov_dc48_seed${S}_*/ 2>/dev/null | head -1)
  "$PY" "$A/extract_features.py" --checkpoint "${CK}checkpoint.pt" \
      --test-h5 data_files/human_cell_test_layer24.h5 \
      --test-fasta data_files/human_cell_test_remapped.fasta \
      --task meltome --out-dir features/meltome_seed${S} --proj-dim 48 --seed ${S} &&
  "$PY" "$A/plot_tsne.py" --features-dir features/meltome_seed${S} --method "$METHOD" \
      --task-name "Meltome (FLIP human_cell, seed ${S})" \
      --out results/figures/tsne_umap/meltome_${METHOD}_seed${S}.png
done

echo "All done."
