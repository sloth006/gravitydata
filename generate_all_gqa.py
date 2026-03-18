"""
generate_all_gqa.py
Generate safetensors files for all grid combinations, **GQA only**.

- Uses the same grids (batch size, dtype, kv length, q length, head size, KV dists)
  as `generate_all.py`.
- Generates both GQA variants:
    - 4:1  -> 8 query heads, 2 KV heads
    - 8:1  -> 8 query heads, 1 KV head
- Always runs on GPU index 1 by default (`cuda:1`) so it uses the **second GPU**
  on a 2‑GPU server.
- Additionally writes an `index.csv` file listing all generated files and their
  parameters.
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


def _count_gqa_jobs(dtypes: list[str]) -> int:
    n = 0
    for _num_heads, _num_kv_heads in GQA_TYPES:
        for batch in BATCH_SIZES:
            for dtype in dtypes:
                for kv_mean in KV_LENGTHS:
                    dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                    for _ in dists:
                        for _ in Q_LENGTHS:
                            for _ in HEAD_SIZES:
                                n += 1
    return n


def _fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GiB"
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


GQA_TYPES: list[tuple[int, int]] = [
    (8, 2),  # 4:1  -> 8 query, 2 KV heads
    (8, 1),  # 8:1  -> 8 query, 1 KV head
]


def generate_all_gqa(
    out_dir: str | Path,
    *,
    seed: int | None = DEFAULT_SEED,
    device: str | None = None,
    dtype_filter: str | None = None,
    start_job: int = 0,
    split_parts: int = 1,
    split_index: int = 0,
    deterministic: bool = False,
) -> list[dict]:
    """
    Generate all GQA combinations into out_dir, with a separate subfolder per ratio
    and per dtype.

    Layout:
      out_dir/
        gqa_8_1/fp16/...
        gqa_8_1/bf16/...
        gqa_8_2/fp16/...
        gqa_8_2/bf16/...

    If dtype_filter is set ("float16" or "bfloat16"), only that dtype is generated.
    Returns all metadata rows (one per file).
    """
    base_path = Path(out_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda:1"

    all_rows: list[dict] = []
    oom_jobs: list[dict] = []
    job_idx = 0
    done_count = 0
    total_bytes = 0

    dtypes = DTYPES if dtype_filter is None else [dtype_filter]
    total_jobs = _count_gqa_jobs(dtypes)
    base_path_resolved = base_path.resolve()
    oom_path = base_path_resolved / "oom_jobs.csv"

    for num_heads, num_kv_heads in GQA_TYPES:
        # One folder per GQA ratio: gqa_8_1, gqa_8_2
        ratio_dir = base_path / f"gqa_{num_heads}_{num_kv_heads}"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []

        for batch in BATCH_SIZES:
            for dtype in dtypes:
                dtype_tag = "fp16" if dtype == "float16" else "bf16"
                dtype_dir = ratio_dir / dtype_tag
                dtype_dir.mkdir(parents=True, exist_ok=True)
                for kv_mean in KV_LENGTHS:
                    dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                    for kv_dist_type, dist_key in dists:
                        kv_dist = (dist_key, kv_mean)
                        for q_len in Q_LENGTHS:
                            q_resolved = _resolve_q_length(q_len)
                            for head in HEAD_SIZES:
                                attn_type = "gqa"
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
                                if split_parts > 1:
                                    if split_index < 0 or split_index >= split_parts:
                                        raise ValueError(f"split_index {split_index} must be in [0, {split_parts})")
                                    # Only process a fraction of jobs (e.g. 50% at a time when split_parts=2).
                                    if (job_idx - start_job) % split_parts != split_index:
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
                                if path.exists():
                                    total_bytes += path.stat().st_size
                                    row = {**row_data}
                                    rows.append(row)
                                    all_rows.append(row)
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
                                        row = {**row_data}
                                        rows.append(row)
                                        all_rows.append(row)
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

        # index.csv per ratio folder
        if rows:
            fieldnames = list(rows[0].keys())
            with (ratio_dir / "index.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    # Top-level index.csv listing all files
    index_path = base_path_resolved / "index.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with index_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    return all_rows, index_path, total_bytes, oom_jobs, oom_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GQA safetensor files for all grid combinations (GPU 1)")
    ap.add_argument(
        "--output-dir",
        "-o",
        default="datasets_gqa",
        help="Output directory for .safetensors files",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    ap.add_argument(
        "--device",
        default=None,
        help='Device string (default: "cuda:1" — second GPU).',
    )
    ap.add_argument(
        "--dtype",
        default="all",
        choices=["all", "float16", "bfloat16"],
        help="Limit generation to a single dtype or both (all)",
    )
    ap.add_argument("--start", type=int, default=0, metavar="N", help="Skip first N jobs (0-based); e.g. --start 1850 to resume from job 1850")
    ap.add_argument(
        "--split-parts",
        type=int,
        default=1,
        metavar="K",
        help="Split the job list into K equal parts (e.g. 2 for 50% at a time).",
    )
    ap.add_argument(
        "--split-index",
        type=int,
        default=0,
        metavar="I",
        help="Which part to run (0-based index, must be < --split-parts).",
    )
    ap.add_argument("--deterministic", action="store_true", help="Enable strict deterministic settings")
    args = ap.parse_args()

    dtype_filter = None if args.dtype == "all" else args.dtype
    rows, index_path, total_bytes, oom_jobs, oom_path = generate_all_gqa(
        args.output_dir,
        seed=args.seed,
        device=args.device,
        dtype_filter=dtype_filter,
        start_job=args.start,
        split_parts=args.split_parts,
        split_index=args.split_index,
        deterministic=args.deterministic,
    )
    print(f"Generated {len(rows)} GQA files under {args.output_dir}")
    print(f"Total file size: {_fmt_size(total_bytes)}")
    print(f"Index CSV: {index_path}")
    print("Per-ratio CSVs: gqa_8_1/index.csv, gqa_8_2/index.csv")
    if oom_jobs:
        print(f"OOM jobs ({len(oom_jobs)}): run with another GPU to retry; list in {oom_path}")


if __name__ == "__main__":
    main()

