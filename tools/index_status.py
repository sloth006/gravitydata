from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

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
from generate_all_gqa import GQA_TYPES


def _iter_mha_grid(dtypes: list[str]) -> Iterable[dict]:
    num_heads = 1
    num_kv_heads = 1
    attn_type = "mha"
    for batch in BATCH_SIZES:
        for dtype in dtypes:
            for kv_mean in KV_LENGTHS:
                dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                for kv_dist_type, dist_key in dists:
                    kv_dist = (dist_key, kv_mean)
                    for q_len in Q_LENGTHS:
                        q_resolved = _resolve_q_length(q_len)
                        for head in HEAD_SIZES:
                            yield {
                                "dtype": dtype,
                                "kv_dist_type": kv_dist_type,
                                "kv_mean": kv_mean,
                                "batch": batch,
                                "q_len": q_len,
                                "q_resolved": q_resolved,
                                "num_kv_heads": num_kv_heads,
                                "num_heads": num_heads,
                                "head_dim": head,
                                "attn_type": attn_type,
                                "kv_dist": kv_dist,
                            }


def _iter_gqa_grid(dtypes: list[str]) -> Iterable[dict]:
    attn_type = "gqa"
    for num_heads, num_kv_heads in GQA_TYPES:
        for batch in BATCH_SIZES:
            for dtype in dtypes:
                for kv_mean in KV_LENGTHS:
                    dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                    for kv_dist_type, dist_key in dists:
                        kv_dist = (dist_key, kv_mean)
                        for q_len in Q_LENGTHS:
                            q_resolved = _resolve_q_length(q_len)
                            for head in HEAD_SIZES:
                                yield {
                                    "dtype": dtype,
                                    "kv_dist_type": kv_dist_type,
                                    "kv_mean": kv_mean,
                                    "batch": batch,
                                    "q_len": q_len,
                                    "q_resolved": q_resolved,
                                    "num_kv_heads": num_kv_heads,
                                    "num_heads": num_heads,
                                    "head_dim": head,
                                    "attn_type": attn_type,
                                    "kv_dist": kv_dist,
                                }


def build_index(
    out_dir: str | Path,
    *,
    attn: str,
    dtype_filter: str,
) -> Path:
    """
    Build a full index.csv with a status column:
      - status=1   -> file exists
      - status=oom -> listed in oom_jobs.csv
      - status=0   -> not generated yet
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    dtypes = [dtype_filter]
    oom_csv = base / "oom_jobs.csv"
    oom_names: set[str] = set()
    if oom_csv.exists():
        with oom_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                oom_names.add(Path(row["path"]).name)

    rows: list[dict] = []

    if attn == "mha":
        grid = _iter_mha_grid(dtypes)
        for spec in grid:
            dtype = spec["dtype"]
            dtype_tag = "fp16" if dtype == "float16" else "bf16"
            dtype_dir = base / dtype_tag
            name = _filename(
                dtype,
                spec["kv_dist_type"],
                spec["kv_mean"],
                spec["batch"],
                spec["q_len"],
                spec["num_kv_heads"],
                spec["num_heads"],
                spec["head_dim"],
                spec["attn_type"],
            )
            path = dtype_dir / name
            status: str
            if path.exists():
                status = "1"
            elif name in oom_names:
                status = "oom"
            else:
                status = "0"
            row = {
                "path": str(path),
                "dtype": dtype,
                "kv_dist_type": spec["kv_dist_type"],
                "kv_mean": spec["kv_mean"],
                "batch": spec["batch"],
                "q_len": spec["q_len"],
                "num_kv_heads": spec["num_kv_heads"],
                "num_heads": spec["num_heads"],
                "head_dim": spec["head_dim"],
                "attn_type": spec["attn_type"],
                "status": status,
            }
            rows.append(row)
    elif attn == "gqa":
        grid = _iter_gqa_grid(dtypes)
        for spec in grid:
            dtype = spec["dtype"]
            dtype_tag = "fp16" if dtype == "float16" else "bf16"
            ratio_dir = base / f"gqa_{spec['num_heads']}_{spec['num_kv_heads']}"
            dtype_dir = ratio_dir / dtype_tag
            name = _filename(
                dtype,
                spec["kv_dist_type"],
                spec["kv_mean"],
                spec["batch"],
                spec["q_len"],
                spec["num_kv_heads"],
                spec["num_heads"],
                spec["head_dim"],
                spec["attn_type"],
            )
            path = dtype_dir / name
            status: str
            if path.exists():
                status = "1"
            elif name in oom_names:
                status = "oom"
            else:
                status = "0"
            row = {
                "path": str(path),
                "dtype": dtype,
                "kv_dist_type": spec["kv_dist_type"],
                "kv_mean": spec["kv_mean"],
                "batch": spec["batch"],
                "q_len": spec["q_len"],
                "num_kv_heads": spec["num_kv_heads"],
                "num_heads": spec["num_heads"],
                "head_dim": spec["head_dim"],
                "attn_type": spec["attn_type"],
                "status": status,
            }
            rows.append(row)
    else:
        raise ValueError(f"Unknown attn type: {attn}")

    index_path = base / "index.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with index_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    return index_path


def run_pending(index_csv: str | Path, *, device: str | None = None) -> None:
    """
    Read an index.csv with a status column and run all jobs
    where status == "0". Successful runs are marked to "1",
    OOMs are marked to "oom".
    """
    index_path = Path(index_csv)
    if not index_path.exists():
        raise SystemExit(f"index.csv not found: {index_path}")

    with index_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No rows in index.csv.")
        return

    dev_str = device

    total = len(rows)
    for i, row in enumerate(rows):
        if row.get("status") != "0":
            continue
        out_file = row["path"]
        kv_dist = (row["kv_dist_type"], int(float(row["kv_mean"])))
        try:
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            data = generate_dataset(
                batch_size=int(row["batch"]),
                dtype=row["dtype"],
                kv_cache=True,
                kv_cache_size_dist=kv_dist,
                q_length=int(row["q_len"]),
                head_size=int(row["head_dim"]),
                num_heads=int(row["num_heads"]),
                num_kv_heads=int(row["num_kv_heads"]),
                attn_type=row["attn_type"],
                q_phase="causal" if int(row["q_len"]) > 2 else "prefill",
                num_batches=1,
                seed=DEFAULT_SEED,
                device=dev_str,
            )
            save_dataset(data, out_file)
            del data
            row["status"] = "1"
            print(f"[{i+1}/{total}] OK: {Path(out_file).name}")
        except torch.OutOfMemoryError:
            row["status"] = "oom"
            print(f"[{i+1}/{total}] OOM: {Path(out_file).name}")
        finally:
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fieldnames = rows[0].keys()
    with index_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and use status-aware index.csv files")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build", help="Build index.csv with status column")
    ap_build.add_argument(
        "--out-dir",
        "-o",
        required=True,
        help="Base output directory (e.g. mha_fp16, mha_bf16, gqa_fp16)",
    )
    ap_build.add_argument(
        "--attn",
        choices=["mha", "gqa"],
        required=True,
        help="Attention type for this directory.",
    )
    ap_build.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        required=True,
        help="Dtype for this directory.",
    )

    ap_run = sub.add_parser("run", help="Run all jobs with status == 0 in an index.csv")
    ap_run.add_argument("index_csv", help="Path to index.csv")
    ap_run.add_argument(
        "--device",
        default=None,
        help="Device to use for generation (e.g. cuda, cuda:1, cpu).",
    )

    args = ap.parse_args()
    if args.cmd == "build":
        index_path = build_index(args.out_dir, attn=args.attn, dtype_filter=args.dtype)
        print(f"Wrote index to {index_path}")
    elif args.cmd == "run":
        run_pending(args.index_csv, device=args.device)


if __name__ == "__main__":
    main()

