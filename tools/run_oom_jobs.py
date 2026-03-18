"""
Run jobs listed in oom_jobs.csv (e.g. on another GPU).
Usage: python3 run_oom_jobs.py path/to/oom_jobs.csv --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch

from dataset import DEFAULT_SEED, generate_dataset, save_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-run OOM jobs from oom_jobs.csv on another device")
    ap.add_argument("csv_path", help="Path to oom_jobs.csv")
    ap.add_argument("--device", default="cuda:1", help="Device to use (e.g. cuda:1, cpu)")
    args = ap.parse_args()

    path = Path(args.csv_path)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No jobs in CSV.")
        return

    device = args.device
    for i, row in enumerate(rows):
        row.pop("error", None)
        out_file = row["path"]
        kv_dist = (row["kv_dist_type"], int(float(row["kv_mean"])))
        try:
            gc.collect()
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
                q_phase="prefill" if int(row["q_len"]) > 2 else "causal",
                num_batches=1,
                seed=DEFAULT_SEED,
                device=device,
            )
            save_dataset(data, out_file)
            del data
            print(f"  [{i+1}/{len(rows)}] {Path(out_file).name}")
        except torch.OutOfMemoryError:
            print(f"  [{i+1}/{len(rows)}] OOM again: {Path(out_file).name}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Done. {len(rows)} jobs processed.")


if __name__ == "__main__":
    main()

