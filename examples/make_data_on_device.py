"""
Make data on device with custom settings.

Use this when you want to generate attention/KV-cache .safetensors locally
with your own batch size, dtype, KV distribution, lengths, etc., without
using the full grid or Drive workflow.

Run from project root:
  python -m examples.make_data_on_device --output my_data.safetensors --batch-size 16 --dtype float16 --q-length 128 --head-size 64 --device cuda

All generate_dataset options are exposed so you can tune for your benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or from examples/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import DEFAULT_SEED, generate_dataset, save_dataset


def _parse_kv_dist(s: str):
    """Parse KV cache size distribution from string. See dataset._parse_kv_dist for full spec."""
    s = s.strip().lower()
    if s == "fixed":
        return "fixed"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 1 and parts[0].isdigit():
        return int(parts[0])
    if parts[0] == "uniform" and len(parts) == 3:
        return ("uniform", int(parts[1]), int(parts[2]))
    if parts[0] == "normal" and len(parts) == 3:
        return ("normal", float(parts[1]), float(parts[2]))
    if parts[0] == "poisson" and len(parts) == 2:
        return ("poisson", float(parts[1]))
    if parts[0] == "constant" and len(parts) == 2:
        return ("constant", float(parts[1]))
    # Single mean form: uniform, normal, exp_soft, etc. with one value
    if len(parts) == 2:
        kind, m = parts[0], float(parts[1])
        if kind in ("uniform", "normal", "exp_soft", "exp_hard", "exp_soft_rev", "exp_hard_rev",
                    "beta_soft", "beta_hard", "lognormal", "poisson", "constant"):
            return (kind, m)
    raise ValueError(
        "kv-cache-size: use 'fixed', int, 'constant,m', 'uniform,m' or 'uniform,min,max', "
        "'normal,m' or 'normal,mean,std', 'poisson,lam', etc."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate one .safetensors dataset on device with custom settings."
    )
    ap.add_argument("--output", "-o", type=Path, required=True, help="Output .safetensors path")
    ap.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--kv-cache", action="store_true", default=True, help="Include K,V cache (default True)")
    ap.add_argument(
        "--kv-cache-size",
        default="constant,1000",
        help="KV length: 'fixed' | int | 'constant,m' | 'uniform,m' | 'normal,m' | 'poisson,lam' | ...",
    )
    ap.add_argument("--q-length", "-q", type=int, default=128, help="Query sequence length")
    ap.add_argument("--head-size", type=int, default=64, help="Head dimension")
    ap.add_argument("--num-heads", type=int, default=8, help="Number of query heads")
    ap.add_argument("--num-kv-heads", type=int, default=None, help="KV heads (default: num_heads for MHA, 1 for GQA)")
    ap.add_argument("--attn-type", choices=["mha", "gqa"], default="mha")
    ap.add_argument("--num-batches", type=int, default=1, help="Number of batch groups (samples = batch_size * num_batches)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default=None, help="cuda, cuda:0, or cpu (default: cuda if available)")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    kv_dist = _parse_kv_dist(args.kv_cache_size)
    num_kv = args.num_kv_heads
    if num_kv is None:
        num_kv = 1 if args.attn_type == "gqa" else args.num_heads

    data = generate_dataset(
        batch_size=args.batch_size,
        dtype=args.dtype,
        kv_cache=args.kv_cache,
        kv_cache_size_dist=kv_dist,
        q_length=args.q_length,
        head_size=args.head_size,
        num_heads=args.num_heads,
        num_kv_heads=num_kv,
        attn_type=args.attn_type,
        q_phase="prefill" if args.q_length > 2 else "causal",
        num_batches=args.num_batches,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
    )
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(data, str(out_path))
    print(f"Saved to {out_path}: {list(data.keys())}")


if __name__ == "__main__":
    main()
