#!/usr/bin/env bash
# Generate full grid of safetensor files for the chosen type.
# Set ATTN_MODE and GRID_DTYPE below (or export them), then run. No prompts.

set -e
cd "$(dirname "$0")"

# Use project venv if present (avoids "externally-managed-environment" on Debian/Ubuntu)
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# What to generate:
#   ATTN_MODE = mha | gqa
#   GRID_DTYPE = all | float16 | bfloat16
ATTN_MODE="${ATTN_MODE:-mha}"
GRID_DTYPE="${GRID_DTYPE:-all}"

echo "Full grid: ATTN_MODE=$ATTN_MODE, GRID_DTYPE=$GRID_DTYPE"

dtype_arg="--dtype $GRID_DTYPE"
if [ "$GRID_DTYPE" = "all" ]; then
  dtype_arg=""
fi

if [ "$ATTN_MODE" = "gqa" ]; then
  echo "Running: python3 generate_all_gqa.py -o datasets_gqa $dtype_arg"
  python3 generate_all_gqa.py -o datasets_gqa $dtype_arg
  echo "Done. Files in datasets_gqa/. Index CSV: datasets_gqa/index.csv"
else
  echo "Running: python3 generate_all_mha.py -o datasets_mha $dtype_arg"
  python3 generate_all_mha.py -o datasets_mha $dtype_arg
  echo "Done. Files in datasets_mha/. Index CSV: datasets_mha/index.csv"
fi
