#!/usr/bin/env bash
# Generate test data, then run attention tests.

set -e
cd "$(dirname "$0")"

echo "=== Generating test data (6 .safetensors) ==="
python tools/generate_test_data.py

echo ""
echo "=== Running attention tests ==="
python tools/test_attention.py
