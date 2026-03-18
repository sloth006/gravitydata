"""
generate_dataset1.py
Generate a fixed subset of GQA datasets and track progress in a CSV.

Grid:
  - batch: 1, 8, 16, 32, 64, 128
  - dtype: bfloat16 only
  - kv mean: 1, 500, 1000, 2000
  - kv distribution: constant, exp_soft, exp_hard, exp_soft_rev, exp_hard_rev
  - Q length: 1, 5
  - head: 128
  - attention: GQA (32q/8kv) and (64q/8kv)

Stores .safetensors in out_dir and records work in index.csv. Supports Ctrl+C
to stop (saves CSV) and continue on next run (skips already-generated files).
"""

from __future__ import annotations

import argparse
import csv
import gc
import signal
from pathlib import Path

import torch

from dataset import DEFAULT_SEED, generate_dataset, save_dataset
from tools.drive_utils import DriveUploader, ensure_memory_budget


# --- Grid for generate_dataset1 ---
BATCH_SIZES = [1, 8, 16, 32, 64, 128]
DTYPE = "bfloat16"
KV_MEANS = [1, 500, 1000, 2000]
# (dist_type for filename, dist_key for generate_dataset). For kv_mean==1 only constant.
KV_DISTRIBUTIONS = [
    ("constant", "constant"),
    ("exp_soft", "exp_soft"),
    ("exp_hard", "exp_hard"),
    ("exp_soft_rev", "exp_soft_rev"),
    ("exp_hard_rev", "exp_hard_rev"),
]
Q_LENGTHS = [1, 5]
HEAD_SIZE = 128
# (num_heads, num_kv_heads)
# 32q/8kv = 4:1 grouping; 64q/8kv = 8:1 grouping
GQA_TYPES = [
    (32, 8),  # 4:1
    (64, 8),  # 8:1
]

STATUS_PENDING = "pending"
STATUS_GENERATED = "generated"
STATUS_UPLOADED = "uploaded"
STATUS_OOM = "oom"

INDEX_FILENAME = "index.csv"
OOM_FILENAME = "oom_jobs.csv"

def _upload_csv_to_drive_replace(
    uploader: DriveUploader,
    local_path: Path,
) -> str | None:
    """
    Upload a CSV to Drive, replacing any existing file in the target folder
    with the same remote name.
    """
    if not local_path.exists():
        return None

    # Best-effort delete same-name files to keep "latest" semantics.
    try:
        for f in uploader.list_files_in_folder():
            if f.get("name") == local_path.name and f.get("id"):
                uploader.service.files().delete(fileId=f["id"]).execute()
    except Exception:
        # If delete fails, we still attempt upload (may create duplicates).
        pass

    return uploader.upload_file(local_path, remote_name=local_path.name)


def _filename(
    dtype: str,
    kv_dist_type: str,
    kv_mean: int,
    batch: int,
    q_len: int,
    num_kv_heads: int,
    num_heads: int,
    head: int,
    attn_type: str,
) -> str:
    return (
        f"gravity_{dtype}_{kv_dist_type}_kv{kv_mean}_b{batch}_q{q_len}_"
        f"{num_kv_heads}_{num_heads}_{head}_{attn_type}.safetensors"
    )


def _build_job_rows(out_dir: Path) -> list[dict]:
    """Build full list of job rows (path, params, status=pending)."""
    rows: list[dict] = []
    for num_heads, num_kv_heads in GQA_TYPES:
        ratio_dir = out_dir / f"gqa_{num_heads}_{num_kv_heads}" / "bf16"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        for batch in BATCH_SIZES:
            for kv_mean in KV_MEANS:
                dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                for kv_dist_type, dist_key in dists:
                    kv_dist = (dist_key, kv_mean)
                    for q_len in Q_LENGTHS:
                        name = _filename(
                            DTYPE,
                            kv_dist_type,
                            kv_mean,
                            batch,
                            q_len,
                            num_kv_heads,
                            num_heads,
                            HEAD_SIZE,
                            "gqa",
                        )
                        path = ratio_dir / name
                        rows.append({
                            "path": str(path),
                            "dtype": DTYPE,
                            "kv_dist_type": kv_dist_type,
                            "kv_mean": kv_mean,
                            "batch": batch,
                            "q_len": q_len,
                            "num_kv_heads": num_kv_heads,
                            "num_heads": num_heads,
                            "head_dim": HEAD_SIZE,
                            "attn_type": "gqa",
                            "status": STATUS_PENDING,
                            "kv_dist": kv_dist,
                        })
    return rows


def _load_index(index_path: Path) -> list[dict]:
    if not index_path.exists():
        return []
    with index_path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("status", STATUS_PENDING)
        row.setdefault("drive_file_id", "")
    return rows


def _merge_with_existing_index(fresh_rows: list[dict], index_path: Path) -> list[dict]:
    """Merge fresh job list with existing index: keep status for same path."""
    existing = _load_index(index_path)
    path_to_status = {r["path"]: r.get("status", STATUS_PENDING) for r in existing}
    path_to_drive_id = {r["path"]: (r.get("drive_file_id", "") or "") for r in existing}
    for row in fresh_rows:
        row["status"] = path_to_status.get(row["path"], STATUS_PENDING)
        row["drive_file_id"] = path_to_drive_id.get(row["path"], "")
    return fresh_rows


def _write_index(rows: list[dict], index_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "path", "dtype", "kv_dist_type", "kv_mean", "batch", "q_len",
        "num_kv_heads", "num_heads", "head_dim", "attn_type", "status", "drive_file_id",
    ]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fieldnames} for row in rows])


def _write_oom(oom_rows: list[dict], oom_path: Path) -> None:
    if not oom_rows:
        return
    fieldnames = list(oom_rows[0].keys())
    with oom_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(oom_rows)


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
    retry_oom_once: bool = True,
    force: bool = False,
    regenerate_q_lengths: list[int] | None = None,
) -> None:
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    index_path = out_path / INDEX_FILENAME
    oom_path = out_path / OOM_FILENAME

    # Build full job list and merge with existing index so we preserve status
    fresh_rows = _build_job_rows(out_path)
    rows = _merge_with_existing_index(fresh_rows, index_path)
    total = len(rows)
    def is_pending(row: dict) -> bool:
        # Regenerate override modes.
        if force:
            return True

        if regenerate_q_lengths is not None:
            try:
                q_len = int(row.get("q_len"))
            except (TypeError, ValueError):
                q_len = int(float(row.get("q_len")))
            if q_len in regenerate_q_lengths:
                return True

        status = (row.get("status") or "").strip().lower()
        if status in {STATUS_GENERATED, STATUS_UPLOADED}:
            return False
        # If we have a Drive file id recorded, treat it as done even if the local file
        # was deleted after upload.
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

    interrupted = [False]  # use list so inner function can set it

    def on_sigint(*_args):  # noqa: ARG001
        interrupted[0] = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow second Ctrl+C to kill

    signal.signal(signal.SIGINT, on_sigint)

    try:
        done = 0
        for job_idx, row in enumerate(rows):
            if interrupted[0]:
                print("\nCtrl+C received. Saving index and exiting.")
                break
            path = Path(row["path"])
            if not is_pending(row):
                done += 1
                continue
            try:
                ensure_memory_budget(
                    device=device,
                    min_cpu_free_gb=min_cpu_free_gb,
                    min_gpu_free_gb=min_gpu_free_gb,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                data = generate_dataset(
                    batch_size=int(row["batch"]),
                    dtype=row["dtype"],
                    kv_cache=True,
                    kv_cache_size_dist=row["kv_dist"],
                    q_length=int(row["q_len"]),
                    head_size=int(row["head_dim"]),
                    num_heads=int(row["num_heads"]),
                    num_kv_heads=int(row["num_kv_heads"]),
                    attn_type=row["attn_type"],
                    q_phase="prefill" if int(row["q_len"]) > 2 else "causal",
                    num_batches=1,
                    seed=seed,
                    device=device,
                    deterministic=deterministic,
                )
                save_dataset(data, row["path"])
                del data
                row["status"] = STATUS_GENERATED

                if uploader is not None:
                    # Upload and optionally delete local file.
                    file_id = uploader.upload_file(path, remote_name=path.name)
                    row["drive_file_id"] = file_id
                    row["status"] = STATUS_UPLOADED
                    if delete_after_upload:
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass

                done += 1
                if done % 50 == 0:
                    if uploader is not None:
                        _upload_csv_to_drive_replace(uploader, index_path)
                    print(f"  Progress: {done}/{total} ({100.0 * done / total:.1f}%)")
            except torch.OutOfMemoryError:
                row["status"] = STATUS_OOM
                oom_row = {k: row.get(k, "") for k in row.keys() if k != "kv_dist"}
                oom_row["error"] = "OOM"
                oom_rows.append(oom_row)
                _write_oom(oom_rows, oom_path)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  OOM: {path.name} (logged to {OOM_FILENAME})")
            finally:
                _write_index(rows, index_path)

        # Optional: retry OOM jobs once at the end. This is useful if memory pressure
        # was temporary (other processes, fragmentation, etc.).
        if retry_oom_once and not interrupted[0]:
            oom_candidates = [r for r in rows if (r.get("status") == STATUS_OOM)]
            if oom_candidates:
                print(f"\nRetrying OOM jobs once: {len(oom_candidates)} job(s).")
                still_oom: list[dict] = []
                for row in oom_candidates:
                    if interrupted[0]:
                        print("\nCtrl+C received during OOM retry. Saving index and exiting.")
                        break
                    path = Path(row["path"])
                    try:
                        ensure_memory_budget(
                            device=device,
                            min_cpu_free_gb=min_cpu_free_gb,
                            min_gpu_free_gb=min_gpu_free_gb,
                        )
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        data = generate_dataset(
                            batch_size=int(row["batch"]),
                            dtype=row["dtype"],
                            kv_cache=True,
                            kv_cache_size_dist=row["kv_dist"],
                            q_length=int(row["q_len"]),
                            head_size=int(row["head_dim"]),
                            num_heads=int(row["num_heads"]),
                            num_kv_heads=int(row["num_kv_heads"]),
                            attn_type=row["attn_type"],
                            q_phase="prefill" if int(row["q_len"]) > 2 else "causal",
                            num_batches=1,
                            seed=seed,
                            device=device,
                            deterministic=deterministic,
                        )
                        save_dataset(data, row["path"])
                        del data
                        row["status"] = STATUS_GENERATED

                        if uploader is not None:
                            file_id = uploader.upload_file(path, remote_name=path.name)
                            row["drive_file_id"] = file_id
                            row["status"] = STATUS_UPLOADED
                            if delete_after_upload:
                                try:
                                    path.unlink()
                                except FileNotFoundError:
                                    pass
                    except torch.OutOfMemoryError:
                        # Keep as OOM; rewrite oom_jobs.csv after retry pass.
                        row["status"] = STATUS_OOM
                        oom_row = {k: row.get(k, "") for k in row.keys() if k != "kv_dist"}
                        oom_row["error"] = "OOM"
                        still_oom.append(oom_row)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    finally:
                        _write_index(rows, index_path)

                # Replace oom_rows file with only the ones that still failed after retry.
                if still_oom:
                    _write_oom(still_oom, oom_path)
                    print(f"OOM retry finished. Still OOM: {len(still_oom)} (logged to {OOM_FILENAME}).")
                else:
                    # If everything succeeded, keep an empty file (or leave existing) by writing header-less nothing.
                    # We only write the file when there are OOM rows to record.
                    print("OOM retry finished. All previously OOM jobs succeeded.")

        if uploader is not None:
            _upload_csv_to_drive_replace(uploader, index_path)
            if oom_path.exists():
                _upload_csv_to_drive_replace(uploader, oom_path)
        print(f"Done. Index written to {index_path}. Completed {done}/{total} jobs.")
    except KeyboardInterrupt:
        print("\nCtrl+C received. Saving index and exiting.")
        _write_index(rows, index_path)
        if oom_rows:
            _write_oom(oom_rows, oom_path)
        if uploader is not None:
            _upload_csv_to_drive_replace(uploader, index_path)
            if oom_path.exists():
                _upload_csv_to_drive_replace(uploader, oom_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate GQA dataset subset (batch 1,8,16,32,64,128; bf16; kv 1,500,1000,2000; Q 1,5; head 128; (32q/8kv) and (64q/8kv))."
    )
    ap.add_argument("--output-dir", "-o", type=Path, default=Path("datasets1"), help="Output directory")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu")
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
    ap.add_argument("--min-cpu-free-gb", type=float, default=1.0, help="Minimum free CPU RAM (GiB) before generating each job")
    ap.add_argument("--min-gpu-free-gb", type=float, default=1.0, help="Minimum free GPU VRAM (GiB) before generating each job")
    ap.add_argument("--no-retry-oom", action="store_true", help="Disable the one-time retry of OOM jobs at the end")
    ap.add_argument("--force", action="store_true", help="Regenerate and re-upload all jobs (overrides resume/skips)")
    ap.add_argument(
        "--regenerate-q-lengths",
        default="all",
        help='Comma list of q lengths to regenerate (e.g. "5" or "1,5"). Default "all".',
    )
    args = ap.parse_args()

    q_lengths: list[int] | None
    if args.regenerate_q_lengths.lower() == "all":
        q_lengths = None
    else:
        q_lengths = [int(x.strip()) for x in args.regenerate_q_lengths.split(",") if x.strip()]

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
        retry_oom_once=not args.no_retry_oom,
        force=args.force,
        regenerate_q_lengths=q_lengths,
    )


if __name__ == "__main__":
    main()
