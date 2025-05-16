#!/usr/bin/env bash
#
# clean_wandb.sh
#
# Deletes all files and subdirectories inside the "wandb" folder,
# but leaves the empty "wandb" directory in place.
#
# Usage:
#   chmod +x clean_wandb.sh
#   ./clean_wandb.sh
#

set -euo pipefail

WANDB_DIR="wandb"

if [ ! -d "$WANDB_DIR" ]; then
  echo "[!] Directory '$WANDB_DIR' not found in $(pwd)"
  exit 1
fi

echo "Cleaning out all contents of '$WANDB_DIR'..."
# remove all files
find "$WANDB_DIR" -maxdepth 1 -type f -exec rm -f {} \;
# remove all subdirectories
find "$WANDB_DIR" -mindepth 1 -type d -exec rm -rf {} \;

echo "Done."
