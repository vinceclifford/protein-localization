#!/usr/bin/env bash
# Overnight preparation: download data and generate all H5 embeddings needed
# for Experiments 1 (d_c sweep) and 3 (layer sweep) on both tasks.
#
# Single pass through ProtT5 per dataset: extracts layers 6, 12, 18, 24 in one
# forward call. The final-layer file (_layer24.h5) is symlinked to the plain
# .h5 name that Exp 1 configs expect.
#
# Outputs per split:
#   {stem}_layer{06,12,18,24}.h5   (Exp 3)
#   {stem}.h5  → symlink to {stem}_layer24.h5   (Exp 1)
#
# Usage:
#   bash scripts/prepare_all_overnight.sh
#   bash scripts/prepare_all_overnight.sh 2>&1 | tee prepare_overnight.log

set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-mps}"
MAX_AA="${MAX_AA:-20000}"
MAX_SEQ="${MAX_SEQ:-128}"
LAYERS=(6 12 18 24)
START_TS=$(date +%s)

log() { echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] === $* ===\n"; }
elapsed_min() { echo "$(( ($(date +%s) - START_TS) / 60 ))"; }

log "STARTING OVERNIGHT PREPARATION (device=$DEVICE, max_aa=$MAX_AA, max_seq=$MAX_SEQ)"

# ── 1. Download FASTAs (no embedding yet) ─────────────────────────────────────
log "[1/4] Downloading DeepLoc FASTAs"
python scripts/prepare_deeploc.py --download-only

log "[2/4] Downloading Meltome FASTAs"
python scripts/prepare_meltome.py --download-only

# ── 2. Per-layer embeddings for DeepLoc ───────────────────────────────────────
log "[3/4] DeepLoc per-layer embeddings (layers ${LAYERS[*]})"
python scripts/embed_layers_h5.py \
    --fasta data_files/deeploc_our_train_set.fasta \
            data_files/deeploc_our_val_set.fasta \
            data_files/deeploc_test_set.fasta \
    --layers "${LAYERS[@]}" \
    --device "$DEVICE" \
    --max-amino-acids "$MAX_AA" --max-sequences "$MAX_SEQ"

for split in deeploc_our_train_set deeploc_our_val_set deeploc_test_set; do
    ln -sf "${split}_layer24.h5" "data_files/${split}.h5"
done

# ── 3. Per-layer embeddings for Meltome ───────────────────────────────────────
log "[4/4] Meltome per-layer embeddings (layers ${LAYERS[*]})"
MELTOME_DIR="data_files/flip_meltome/prepared/human_cell"
python scripts/embed_layers_h5.py \
    --fasta ${MELTOME_DIR}/human_cell_train.fasta \
            ${MELTOME_DIR}/human_cell_val.fasta \
            ${MELTOME_DIR}/human_cell_test.fasta \
    --layers "${LAYERS[@]}" \
    --device "$DEVICE" \
    --max-amino-acids "$MAX_AA" --max-sequences "$MAX_SEQ"

for split in human_cell_train human_cell_val human_cell_test; do
    ln -sf "${split}_layer24.h5" "${MELTOME_DIR}/${split}.h5"
done

log "DONE — total elapsed: $(elapsed_min) min"
echo "You can now run Experiments 1 and 3."
