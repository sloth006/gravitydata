"""
Load a gravity .safetensors file and show how to get the data out.

Use this to understand the layout of saved tensors and how to use them
(e.g. for attention kernels or benchmarking).

Run from project root:
  python -m examples.load_safetensor path/to/file.safetensors

  python -m examples.load_safetensor path/to/file.safetensors --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or from examples/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from safetensors.torch import load_file


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Load a gravity .safetensors file and show how to get Q, K, V, and metadata out."
    )
    ap.add_argument(
        "path",
        type=Path,
        help="Path to a .safetensors file (e.g. gravity_float16_constant_kv500_b8_q3_1_1_64_gqa.safetensors)",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="Device to load tensors onto (e.g. cuda, cpu). Default: cpu.",
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="Only print keys and shapes, then exit.",
    )
    args = ap.parse_args()

    path = args.path
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Load the entire file. Returns a dict[str, torch.Tensor].
    data = load_file(str(path))

    if args.list_only:
        print("Keys and shapes:")
        for k, t in data.items():
            if hasattr(t, "shape"):
                print(f"  {k}: shape={t.shape}, dtype={t.dtype}")
            else:
                print(f"  {k}: {t}")
        return

    # --- How to get the data out ---

    # Scalar metadata (stored as 1-element tensors; use .item() to get Python int)
    batch_size = data["batch_size"].item()
    q_length = data["q_length"].item()
    head_size = data["head_size"].item()
    num_heads = data["num_heads"].item()
    num_kv_heads = data["num_kv_heads"].item()

    # Query: (batch, num_heads, q_length, head_size)
    q = data["q"]

    # Optional: move to device for GPU use
    device = args.device
    if device != "cpu":
        dev = torch.device(device)
        q = q.to(dev)

    # If the file was generated with KV cache (--kv-cache), you also have:
    has_kv = "k" in data and "v" in data

    if has_kv:
        k = data["k"]  # (batch, num_kv_heads, max_kv_len, head_size)
        v = data["v"]  # (batch, num_kv_heads, max_kv_len, head_size)
        kv_lengths = data["kv_lengths"]  # (batch,) int64, actual KV length per batch item
        kv_mask = data["kv_mask"]        # (batch, 1, 1, max_kv_len), 1=valid 0=pad
        attn_out = data["attn_out"]      # (batch, num_heads, q_length, head_size) reference output

        if device != "cpu":
            dev = torch.device(device)
            k = k.to(dev)
            v = v.to(dev)
            kv_lengths = kv_lengths.to(dev)
            kv_mask = kv_mask.to(dev)
            attn_out = attn_out.to(dev)

    # --- Print summary ---
    print(f"Loaded: {path.name}")
    print(f"  batch_size={batch_size}, q_length={q_length}, head_size={head_size}")
    print(f"  num_heads={num_heads}, num_kv_heads={num_kv_heads}")
    print(f"  q shape: {q.shape}")
    if has_kv:
        print(f"  k shape: {k.shape}, v shape: {v.shape}")
        print(f"  kv_lengths: {kv_lengths.shape} (per-batch KV lengths)")
        print(f"  kv_mask: {kv_mask.shape}")
        print(f"  attn_out: {attn_out.shape}")
    print(f"  device: {q.device}")

    # Example: use in your code
    print("\nExample usage in your code:")
    print("  from safetensors.torch import load_file")
    print("  data = load_file('your_file.safetensors')")
    print("  q = data['q']           # (B, num_heads, q_len, head_size)")
    print("  k = data['k']           # (B, num_kv_heads, kv_len, head_size)")
    print("  v = data['v']")
    print("  batch_size = data['batch_size'].item()")
    print("  num_heads = data['num_heads'].item()")
    print("  num_kv_heads = data['num_kv_heads'].item()")


if __name__ == "__main__":
    main()
