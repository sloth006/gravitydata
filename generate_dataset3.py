"""
generate_dataset3.py

Generate a small subset of attention datasets where:
  - Workloads = 4 attention head configs x 5 KV length distributions = 20 total
  - Each workload uses:
    - `batch_size` in {8, 16}
    - `kv_mean` in {500, 1000}
    - `q_len` in {1, 5}
  - Index CSV stores the ACTUAL mean of sampled KV lengths computed from `kv_lengths`
    (column: `kv_len_mean_actual`).

This script keeps the same Drive upload + Ctrl+C resume behavior and the same
chunked/streaming attention kernels as `generate_dataset2.py`.
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import signal
from pathlib import Path

import torch

import generate_dataset2 as gen2
from dataset import DEFAULT_SEED, generate_dataset, save_dataset
from tools.drive_utils import DriveUploader, ensure_memory_budget


STATUS_PENDING = gen2.STATUS_PENDING
STATUS_GENERATED = gen2.STATUS_GENERATED
STATUS_UPLOADED = gen2.STATUS_UPLOADED
STATUS_OOM = gen2.STATUS_OOM

INDEX_FILENAME = gen2.INDEX_FILENAME
OOM_FILENAME = gen2.OOM_FILENAME

DTYPE = gen2.DTYPE

CHUNK_KV_DEFAULT = 128

# Attention head configs (num_heads, num_kv_heads, head_dim).
MHA_TYPES: list[tuple[int, int, int]] = [
    (32, 32, 64),
    (16, 16, 128),
]

GQA_TYPES: list[tuple[int, int, int]] = [
    (32, 8, 128),
    (64, 8, 128),
]

# Exactly the 5 distribution types.
KV_DISTRIBUTIONS = [
    ("constant", "constant"),
    ("exp_soft", "exp_soft"),
    ("exp_hard", "exp_hard"),
    ("exp_soft_rev", "exp_soft_rev"),
    ("exp_hard_rev", "exp_hard_rev"),
]

# Workload parameter sets requested by the user.
# Keep these small for fast runs.
BATCH_SIZE_CHOICES = [8, 16]
KV_MEAN_CHOICES = [500, 1000]
Q_LEN_CHOICES = [1, 5]


def _build_job_rows(out_dir: Path, *, seed_choice: int) -> list[dict]:
    rows: list[dict] = []
    for head_cfg_idx, (num_heads, num_kv_heads, head_dim) in enumerate(MHA_TYPES):
        batch_size = BATCH_SIZE_CHOICES[head_cfg_idx % len(BATCH_SIZE_CHOICES)]
        ratio_dir = out_dir / f"mha_{num_heads}_{num_kv_heads}" / "bf16"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        for dist_idx, (kv_dist_type, dist_key) in enumerate(KV_DISTRIBUTIONS):
            kv_mean = KV_MEAN_CHOICES[dist_idx % len(KV_MEAN_CHOICES)]
            q_len = Q_LEN_CHOICES[dist_idx % len(Q_LEN_CHOICES)]
            name = gen2._filename(
                DTYPE,
                kv_dist_type,
                kv_mean,
                batch_size,
                q_len,
                num_kv_heads,
                num_heads,
                head_dim,
                "mha",
            )
            path = ratio_dir / name
            rows.append(
                {
                    "path": str(path),
                    "dtype": DTYPE,
                    "kv_dist_type": kv_dist_type,
                    "kv_mean": kv_mean,
                    "kv_len_mean_actual": "",
                    "batch": batch_size,
                    "q_len": q_len,
                    "num_kv_heads": num_kv_heads,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "attn_type": "mha",
                    "status": STATUS_PENDING,
                    "drive_file_id": "",
                    "kv_dist": (dist_key, kv_mean),
                }
            )

    for head_cfg_idx, (num_heads, num_kv_heads, head_dim) in enumerate(GQA_TYPES):
        batch_size = BATCH_SIZE_CHOICES[head_cfg_idx % len(BATCH_SIZE_CHOICES)]
        ratio_dir = out_dir / f"gqa_{num_heads}_{num_kv_heads}" / "bf16"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        for dist_idx, (kv_dist_type, dist_key) in enumerate(KV_DISTRIBUTIONS):
            kv_mean = KV_MEAN_CHOICES[dist_idx % len(KV_MEAN_CHOICES)]
            q_len = Q_LEN_CHOICES[dist_idx % len(Q_LEN_CHOICES)]
            name = gen2._filename(
                DTYPE,
                kv_dist_type,
                kv_mean,
                batch_size,
                q_len,
                num_kv_heads,
                num_heads,
                head_dim,
                "gqa",
            )
            path = ratio_dir / name
            rows.append(
                {
                    "path": str(path),
                    "dtype": DTYPE,
                    "kv_dist_type": kv_dist_type,
                    "kv_mean": kv_mean,
                    "kv_len_mean_actual": "",
                    "batch": batch_size,
                    "q_len": q_len,
                    "num_kv_heads": num_kv_heads,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "attn_type": "gqa",
                    "status": STATUS_PENDING,
                    "drive_file_id": "",
                    "kv_dist": (dist_key, kv_mean),
                }
            )

    return rows


def _load_index(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    with index_path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("status", STATUS_PENDING)
        row.setdefault("drive_file_id", "")
        row.setdefault("kv_len_mean_actual", "")
    return rows


def _merge_with_existing_index(fresh_rows: list[dict], index_path: Path) -> list[dict]:
    """
    Preserve `kv_len_mean_actual` from an existing index when resuming.
    """
    existing = _load_index(index_path)
    path_to_existing = {r.get("path", ""): r for r in existing if r.get("path", "")}

    for row in fresh_rows:
        ex = path_to_existing.get(row["path"])
        if not ex:
            continue
        row["status"] = ex.get("status", STATUS_PENDING)
        row["drive_file_id"] = ex.get("drive_file_id", "") or ""
        if "kv_len_mean_actual" in ex and str(ex.get("kv_len_mean_actual", "")).strip() != "":
            row["kv_len_mean_actual"] = ex.get("kv_len_mean_actual", "")
    return fresh_rows


def _write_index(rows: list[dict], index_path: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "path",
        "dtype",
        "kv_dist_type",
        "kv_mean",
        "kv_len_mean_actual",
        "batch",
        "q_len",
        "num_kv_heads",
        "num_heads",
        "head_dim",
        "attn_type",
        "status",
        "drive_file_id",
    ]

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fieldnames} for row in rows])


def _write_oom(oom_rows: list[dict], oom_path: Path) -> None:
    # Reuse generate_dataset2 behavior (writes all keys it finds).
    gen2._write_oom(oom_rows, oom_path)


def run(
    out_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
    deterministic: bool = False,
    drive_folder_id: str | None = None,
    auth: str = "oauth",
    credentials_json: str | None = None,
    token_json: str | None = None,
    delete_after_upload: bool = True,
    min_cpu_free_gb: float = 1.0,
    min_gpu_free_gb: float = 1.0,
    chunk_kv: int = CHUNK_KV_DEFAULT,
) -> None:
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    index_path = out_path / INDEX_FILENAME
    oom_path = out_path / OOM_FILENAME

    # Separate RNG seed for choosing kv_mean targets (so it's stable across runs).
    seed_choice = seed + 1337

    fresh_rows = _build_job_rows(out_path, seed_choice=seed_choice)
    rows = _merge_with_existing_index(fresh_rows, index_path)
    total = len(rows)

    def is_pending(row: dict) -> bool:
        status = (row.get("status") or "").strip().lower()
        # Don't retry known bad jobs (prevents infinite OOM loops).
        if status in {STATUS_GENERATED, STATUS_UPLOADED, STATUS_OOM}:
            return False
        if (row.get("drive_file_id") or "").strip():
            return False
        if Path(row["path"]).exists():
            return False
        return True

    pending = sum(1 for r in rows if is_pending(r))
    print(f"Total jobs: {total}, pending: {pending}. Resuming/starting.")
    if pending == 0:
        print("All jobs already generated or skipped. Exiting.")
        return

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    uploader: DriveUploader | None = None
    if drive_folder_id:
        if auth == "oauth" and not credentials_json:
            raise ValueError("--credentials-json is required for oauth when using --drive-folder-id")
        uploader = DriveUploader(
            drive_folder_id,
            auth_mode=auth,
            credentials_json=credentials_json,
            token_json=token_json,
        )

    oom_rows: list[dict] = []
    if oom_path.exists():
        oom_rows = _load_index(oom_path)

    interrupted = [False]

    def on_sigint(*_args):  # noqa: ARG001
        interrupted[0] = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, on_sigint)

    try:
        done = 0
        for job_idx, row in enumerate(rows):
            if interrupted[0]:
                print("\nCtrl+C received. Saving index and exiting.")
                break
            if not is_pending(row):
                done += 1
                continue

            data = None
            q = None
            k = None
            v = None
            kv_lengths = None
            attn_out = None
            oom_hit = False

            try:
                ensure_memory_budget(
                    device=device,
                    min_cpu_free_gb=min_cpu_free_gb,
                    min_gpu_free_gb=min_gpu_free_gb,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Generate q/k/v on CPU to avoid GPU allocation spikes.
                q_len = int(row["q_len"])
                # For q_len=5 we want causal triangular masking for all configs.
                q_phase = "causal" if q_len > 2 else "prefill"

                data = generate_dataset(
                    batch_size=int(row["batch"]),
                    dtype=row["dtype"],
                    kv_cache=True,
                    kv_cache_size_dist=row["kv_dist"],
                    q_length=q_len,
                    head_size=int(row["head_dim"]),
                    num_heads=int(row["num_heads"]),
                    num_kv_heads=int(row["num_kv_heads"]),
                    attn_type=row["attn_type"],
                    compute_attn_out=False,
                    q_phase=q_phase,
                    num_batches=1,
                    seed=seed,
                    device="cpu",
                    deterministic=deterministic,
                )

                # Actual mean of sampled KV cache lengths (computed from kv_lengths).
                row["kv_len_mean_actual"] = float(data["kv_lengths"].float().mean().item())

                # Compute attention on GPU with chunked KV softmax.
                q = data["q"].to(device)
                k = data["k"]
                v = data["v"]
                kv_lengths = data["kv_lengths"].to(torch.int64)

                if row["attn_type"] == "mha":
                    attn_out = gen2._attention_forward_chunked_mha(
                        q, k, v, kv_lengths, q_phase=q_phase, chunk_kv=chunk_kv
                    )
                else:
                    attn_out = gen2._attention_forward_chunked_gqa(
                        q, k, v, kv_lengths, q_phase=q_phase, chunk_kv=chunk_kv
                    )

                # Move back to CPU for saving.
                data["attn_out"] = attn_out.cpu()
                save_dataset(data, row["path"])

                row["status"] = STATUS_GENERATED

                if uploader is not None:
                    file_id = uploader.upload_file(
                        Path(row["path"]),
                        remote_name=Path(row["path"]).name,
                    )
                    row["drive_file_id"] = file_id
                    row["status"] = STATUS_UPLOADED
                    if delete_after_upload:
                        Path(row["path"]).unlink(missing_ok=True)

                done += 1
                # For 20 jobs, progress upload every 50 might never trigger.
                if uploader is not None and (done % 10 == 0 or done == pending):
                    gen2._upload_csv_to_drive_replace(uploader, index_path)
                    print(f"  Progress: {done}/{total} ({100.0 * done / total:.1f}%)")
            except torch.OutOfMemoryError:
                oom_hit = True
                row["status"] = STATUS_OOM
                oom_row = {k: row.get(k, "") for k in row.keys() if k != "kv_dist"}
                oom_row["error"] = "OOM"
                oom_rows.append(oom_row)
                _write_oom(oom_rows, oom_path)
                # Don't empty_cache() until references are deleted in finally.
                gc.collect()
                print(f"  OOM: {Path(row['path']).name} (logged to {OOM_FILENAME})")
            finally:
                # 1) Delete GPU refs first to avoid “phantom leak” cascades.
                if attn_out is not None:
                    del attn_out
                    attn_out = None
                if kv_lengths is not None:
                    del kv_lengths
                    kv_lengths = None
                if q is not None:
                    del q
                    q = None

                # 2) Drop CPU-side refs too (helps overall GC).
                if data is not None:
                    del data
                    data = None
                k = None
                v = None

                gc.collect()

                # 3) Only clear VRAM after refs are gone; only when OOM hit.
                if oom_hit and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                _write_index(rows, index_path)

        if uploader is not None:
            gen2._upload_csv_to_drive_replace(uploader, index_path)
            if oom_path.exists():
                gen2._upload_csv_to_drive_replace(uploader, oom_path)

        print(f"Done. Index written to {index_path}. Completed {done}/{total} jobs.")
    except KeyboardInterrupt:
        print("\nCtrl+C received. Saving index and exiting.")
        _write_index(rows, index_path)
        if oom_rows:
            _write_oom(oom_rows, oom_path)
        if uploader is not None:
            gen2._upload_csv_to_drive_replace(uploader, index_path)
            if oom_path.exists():
                gen2._upload_csv_to_drive_replace(uploader, oom_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate small dataset subset (20 jobs) with Drive + resume.")
    ap.add_argument("--output-dir", "-o", type=Path, default=Path("datasets3"), help="Output directory")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default=None, help="CUDA device for attention, e.g. cuda:0 (generation runs on CPU)")
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--drive-folder-id", default=None, help="If set, upload each generated .safetensors to this Drive folder")
    ap.add_argument("--auth", choices=["oauth", "service-account"], default="oauth")
    ap.add_argument("--credentials-json", default=None, help="OAuth client secret or service account JSON")
    ap.add_argument("--token-json", default=None, help="OAuth token cache (optional)")
    ap.add_argument(
        "--keep-local",
        action="store_true",
        help="If set with --drive-folder-id, keep local .safetensors after upload (default deletes).",
    )

    ap.add_argument("--min-cpu-free-gb", type=float, default=1.0, help="Minimum free CPU RAM (GiB) before each job")
    ap.add_argument("--min-gpu-free-gb", type=float, default=1.0, help="Minimum free GPU VRAM (GiB) before each job")
    ap.add_argument("--chunk-kv", type=int, default=CHUNK_KV_DEFAULT, help="KV chunk size for streaming attention softmax")

    args = ap.parse_args()
    run(
        args.output_dir,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
        drive_folder_id=args.drive_folder_id,
        auth=args.auth,
        credentials_json=args.credentials_json,
        token_json=args.token_json,
        delete_after_upload=not args.keep_local,
        min_cpu_free_gb=args.min_cpu_free_gb,
        min_gpu_free_gb=args.min_gpu_free_gb,
        chunk_kv=args.chunk_kv,
    )


if __name__ == "__main__":
    main()

