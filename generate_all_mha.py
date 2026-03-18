"""
generate_all_mha.py
Generate safetensors files for all grid combinations, **MHA only**.

- Uses the same grids (batch size, dtype, kv length, q length, head size, KV dists)
  as `generate_all.py`.
- Generates only MHA (num_heads = num_kv_heads = 1 for the grid).
- Writes an `index.csv` in the output directory listing all generated files.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch

from dataset import DEFAULT_SEED, generate_dataset, save_dataset
from generate_all import (
    BATCH_SIZES,
    DTYPES,
    KV_LENGTHS,
    Q_LENGTHS,
    HEAD_SIZES,
    KV_DISTRIBUTIONS,
    _resolve_q_length,
    _filename,
)


def _count_mha_jobs(dtypes: list[str]) -> int:
    n = 0
    for batch in BATCH_SIZES:
        for dtype in dtypes:
            for kv_mean in KV_LENGTHS:
                dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                for _ in dists:
                    for _ in Q_LENGTHS:
                        for _ in HEAD_SIZES:
                            n += 1
    return n


def generate_all_mha(
    out_dir: str | Path,
    *,
    seed: int | None = DEFAULT_SEED,
    device: str | None = None,
    dtype_filter: str | None = None,
    start_job: int = 0,
    deterministic: bool = False,
    force: bool = False,
) -> list[dict]:
    """
    Generate all MHA combinations into out_dir. Returns metadata rows.

    Layout:
      out_dir/fp16/...
      out_dir/bf16/...

    If dtype_filter is set ("float16" or "bfloat16"), only that dtype is generated.
    Each row: path, dtype, kv_dist_type, kv_mean, batch, q_len, num_kv_heads, num_heads, head_dim, attn_type.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_heads = 1
    num_kv_heads = 1
    attn_type = "mha"

    rows: list[dict] = []
    oom_jobs: list[dict] = []
    job_idx = 0
    done_count = 0
    total_bytes = 0

    dtypes = DTYPES if dtype_filter is None else [dtype_filter]
    total_jobs = _count_mha_jobs(dtypes)
    oom_path = out_path.resolve() / "oom_jobs.csv"

    for batch in BATCH_SIZES:
        for dtype in dtypes:
            dtype_tag = "fp16" if dtype == "float16" else "bf16"
            dtype_dir = out_path / dtype_tag
            dtype_dir.mkdir(parents=True, exist_ok=True)
            for kv_mean in KV_LENGTHS:
                dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                for kv_dist_type, dist_key in dists:
                    kv_dist = (dist_key, kv_mean)
                    for q_len in Q_LENGTHS:
                        q_resolved = _resolve_q_length(q_len)
                        for head in HEAD_SIZES:
                            name = _filename(
                                dtype,
                                kv_dist_type,
                                kv_mean,
                                batch,
                                q_len,
                                num_kv_heads,
                                num_heads,
                                head,
                                attn_type,
                            )
                            path = dtype_dir / name
                            if job_idx < start_job:
                                job_idx += 1
                                continue
                            cur_seed = seed if seed is not None else DEFAULT_SEED
                            row_data = {
                                "path": str(path),
                                "dtype": dtype,
                                "kv_dist_type": kv_dist_type,
                                "kv_mean": kv_mean,
                                "batch": batch,
                                "q_len": q_len,
                                "num_kv_heads": num_kv_heads,
                                "num_heads": num_heads,
                                "head_dim": head,
                                "attn_type": attn_type,
                            }
                            q_phase = "prefill" if q_resolved > 2 else "causal"
                            if path.exists() and not force:
                                total_bytes += path.stat().st_size
                                rows.append(row_data)
                                done_count += 1
                                if done_count % 50 == 0:
                                    pct = 100.0 * (start_job + done_count) / total_jobs
                                    print(f"  Progress: {start_job + done_count}/{total_jobs} ({pct:.1f}%) [skipped, already exists]")
                            else:
                                try:
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    data = generate_dataset(
                                        batch_size=batch,
                                        dtype=dtype,
                                        kv_cache=True,
                                        kv_cache_size_dist=kv_dist,
                                        q_length=q_resolved,
                                        head_size=head,
                                        num_heads=num_heads,
                                        num_kv_heads=num_kv_heads,
                                        attn_type=attn_type,
                                        q_phase=q_phase,
                                        num_batches=1,
                                        seed=cur_seed,
                                        device=device,
                                        deterministic=deterministic,
                                    )
                                    save_dataset(data, str(path))
                                    del data
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    total_bytes += path.stat().st_size
                                    rows.append(row_data)
                                    done_count += 1
                                    if done_count % 50 == 0:
                                        pct = 100.0 * (start_job + done_count) / total_jobs
                                        print(f"  Progress: {start_job + done_count}/{total_jobs} ({pct:.1f}%)")
                                except torch.OutOfMemoryError:
                                    oom_jobs.append({**row_data, "error": "OOM"})
                                    oom_fieldnames = list(row_data.keys()) + ["error"]
                                    with oom_path.open("w", newline="") as f:
                                        w = csv.DictWriter(f, fieldnames=oom_fieldnames)
                                        w.writeheader()
                                        w.writerows(oom_jobs)
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    if done_count % 50 == 0 or done_count == 0:
                                        pct = 100.0 * (start_job + done_count) / total_jobs
                                        print(f"  Progress: {start_job + done_count}/{total_jobs} ({pct:.1f}%) — OOM, logged to {oom_path.name}")
                            job_idx += 1

    index_path = out_path.resolve() / "index.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with index_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows, index_path, total_bytes, oom_jobs, oom_path


def _fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MHA safetensor files for all grid combinations")
    ap.add_argument("--output-dir", "-o", default="datasets_mha", help="Output directory for .safetensors files")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    ap.add_argument("--device", default=None, help="Device: cuda, cuda:0, cpu, etc.")
    ap.add_argument(
        "--dtype",
        default="all",
        choices=["all", "float16", "bfloat16"],
        help="Limit generation to a single dtype or both (all)",
    )
    ap.add_argument("--start", type=int, default=0, metavar="N", help="Skip first N jobs (0-based); e.g. --start 1850 to resume from job 1850")
    ap.add_argument("--deterministic", action="store_true", help="Enable strict deterministic settings")
    ap.add_argument("--force", action="store_true", help="Regenerate even if output files already exist")
    args = ap.parse_args()

    dtype_filter = None if args.dtype == "all" else args.dtype
    rows, index_path, total_bytes, oom_jobs, oom_path = generate_all_mha(
        args.output_dir,
        seed=args.seed,
        device=args.device,
        dtype_filter=dtype_filter,
        start_job=args.start,
        deterministic=args.deterministic,
        force=args.force,
    )
    print(f"Generated {len(rows)} MHA files in {args.output_dir}")
    print(f"Total file size: {_fmt_size(total_bytes)}")
    print(f"Index CSV: {index_path}")
    if oom_jobs:
        print(f"OOM jobs ({len(oom_jobs)}): run with --device cuda:1 (or another GPU) to retry; list in {oom_path}")


if __name__ == "__main__":
    main()
