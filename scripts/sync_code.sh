#!/usr/bin/env bash
# Push local project files to the Mac Studio for running experiments.
# Data files (H5, FASTA), run outputs, and sweep results are excluded.
#
# Usage:
#   scripts/sync_code.sh

set -euo pipefail

REMOTE_HOST="shimpitech@100.69.218.75"
REMOTE_DST="${REMOTE_HOST}:/Users/shimpitech/vincent/protein-localization/"
LOCAL_ROOT="$(dirname "$0")/.."

echo "Pushing code → ${REMOTE_DST}"
rsync -av \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='data_files/' \
    --exclude='runs/' \
    --exclude='sweeps/' \
    --exclude='*.h5' \
    --exclude='*.fasta' \
    "${LOCAL_ROOT}/" "$REMOTE_DST"

echo "Done."
