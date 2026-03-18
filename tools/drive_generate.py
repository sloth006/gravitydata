from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from typing import Iterable

import torch

from dataset import DEFAULT_SEED, generate_dataset, save_dataset
from tools.drive_utils import DriveUploader, ensure_memory_budget
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


STATUS_PENDING = "0"
STATUS_GENERATED = "generated"
STATUS_UPLOADED = "uploaded"
STATUS_OOM = "oom"


def _parse_csv_values(value: str | None, cast=str) -> list | None:
    if value is None or value.lower() == "all":
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not path.exists():
            path.write_text("path,dtype,kv_dist_type,kv_mean,batch,q_len,num_kv_heads,num_heads,head_dim,attn_type,status,drive_file_id\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.setdefault("status", STATUS_PENDING)
        row.setdefault("drive_file_id", "")
    return rows


def _iter_mha_jobs(
    out_dir: Path,
    dtype_filter: str,
    batches: list[int] | None,
    kv_means: list[int] | None,
    q_lens: list[int] | None,
    head_dims: list[int] | None,
    kv_dists: list[str] | None,
) -> Iterable[dict]:
    num_heads = 1
    num_kv_heads = 1
    attn_type = "mha"
    dtypes = DTYPES if dtype_filter == "all" else [dtype_filter]
    batches = BATCH_SIZES if batches is None else batches
    kv_means = KV_LENGTHS if kv_means is None else kv_means
    q_lens = Q_LENGTHS if q_lens is None else q_lens
    head_dims = HEAD_SIZES if head_dims is None else head_dims

    for batch in batches:
        for dtype in dtypes:
            dtype_tag = "fp16" if dtype == "float16" else "bf16"
            dtype_dir = out_dir / dtype_tag
            for kv_mean in kv_means:
                dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                for kv_dist_type, dist_key in dists:
                    if kv_dists is not None and kv_dist_type not in kv_dists:
                        continue
                    kv_dist = (dist_key, kv_mean)
                    for q_len in q_lens:
                        for head in head_dims:
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
                            yield {
                                "path": str(dtype_dir / name),
                                "dtype": dtype,
                                "kv_dist_type": kv_dist_type,
                                "kv_mean": kv_mean,
                                "batch": batch,
                                "q_len": q_len,
                                "num_kv_heads": num_kv_heads,
                                "num_heads": num_heads,
                                "head_dim": head,
                                "attn_type": attn_type,
                                "kv_dist": kv_dist,
                            }


def _iter_gqa_jobs(
    out_dir: Path,
    dtype_filter: str,
    batches: list[int] | None,
    kv_means: list[int] | None,
    q_lens: list[int] | None,
    head_dims: list[int] | None,
    kv_dists: list[str] | None,
    gqa_ratios: list[str] | None,
) -> Iterable[dict]:
    attn_type = "gqa"
    dtypes = DTYPES if dtype_filter == "all" else [dtype_filter]
    batches = BATCH_SIZES if batches is None else batches
    kv_means = KV_LENGTHS if kv_means is None else kv_means
    q_lens = Q_LENGTHS if q_lens is None else q_lens
    head_dims = HEAD_SIZES if head_dims is None else head_dims

    ratio_map = {f"{h}_{k}": (h, k) for h, k in GQA_TYPES}
    ratio_items = GQA_TYPES if gqa_ratios is None else [ratio_map[r] for r in gqa_ratios if r in ratio_map]

    for num_heads, num_kv_heads in ratio_items:
        for batch in batches:
            for dtype in dtypes:
                dtype_tag = "fp16" if dtype == "float16" else "bf16"
                dtype_dir = out_dir / f"gqa_{num_heads}_{num_kv_heads}" / dtype_tag
                for kv_mean in kv_means:
                    dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                    for kv_dist_type, dist_key in dists:
                        if kv_dists is not None and kv_dist_type not in kv_dists:
                            continue
                        kv_dist = (dist_key, kv_mean)
                        for q_len in q_lens:
                            for head in head_dims:
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
                                yield {
                                    "path": str(dtype_dir / name),
                                    "dtype": dtype,
                                    "kv_dist_type": kv_dist_type,
                                    "kv_mean": kv_mean,
                                    "batch": batch,
                                    "q_len": q_len,
                                    "num_kv_heads": num_kv_heads,
                                    "num_heads": num_heads,
                                    "head_dim": head,
                                    "attn_type": attn_type,
                                    "kv_dist": kv_dist,
                                }


def _build_rows(
    attn: str,
    out_dir: str | Path,
    dtype_filter: str,
    batches: list[int] | None = None,
    kv_means: list[int] | None = None,
    q_lens: list[int] | None = None,
    head_dims: list[int] | None = None,
    kv_dists: list[str] | None = None,
    gqa_ratios: list[str] | None = None,
    start: int = 0,
    split_parts: int = 1,
    split_index: int = 0,
) -> list[dict]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if attn == "mha":
        jobs = list(_iter_mha_jobs(out_path, dtype_filter, batches, kv_means, q_lens, head_dims, kv_dists))
    elif attn == "gqa":
        jobs = list(_iter_gqa_jobs(out_path, dtype_filter, batches, kv_means, q_lens, head_dims, kv_dists, gqa_ratios))
    else:
        raise ValueError(f"Unknown attn type: {attn}")

    if start > 0:
        jobs = jobs[start:]
    if split_parts > 1:
        if split_index < 0 or split_index >= split_parts:
            raise ValueError(f"split_index must be in [0, {split_parts})")
        jobs = [job for i, job in enumerate(jobs) if i % split_parts == split_index]

    rows = []
    for job in jobs:
        row = {
            "path": job["path"],
            "dtype": job["dtype"],
            "kv_dist_type": job["kv_dist_type"],
            "kv_mean": job["kv_mean"],
            "batch": job["batch"],
            "q_len": job["q_len"],
            "num_kv_heads": job["num_kv_heads"],
            "num_heads": job["num_heads"],
            "head_dim": job["head_dim"],
            "attn_type": job["attn_type"],
            "status": STATUS_PENDING,
            "drive_file_id": "",
        }
        rows.append(row)
    return rows


def _row_to_generate_kwargs(row: dict, device: str | None, seed: int | None, _job_idx: int) -> dict:
    kv_dist = (row["kv_dist_type"], int(row["kv_mean"]))
    return {
        "batch_size": int(row["batch"]),
        "dtype": row["dtype"],
        "kv_cache": True,
        "kv_cache_size_dist": kv_dist,
        "q_length": int(row["q_len"]),
        "head_size": int(row["head_dim"]),
        "num_heads": int(row["num_heads"]),
        "num_kv_heads": int(row["num_kv_heads"]),
        "attn_type": row["attn_type"],
        "num_batches": 1,
        "seed": seed if seed is not None else DEFAULT_SEED,
        "device": device,
    }


def _upload_and_cleanup(
    uploader: DriveUploader,
    rows: list[dict],
    row: dict,
    delete_after_upload: bool,
) -> None:
    path = Path(row["path"])
    if not path.exists():
        return
    file_id = uploader.upload_file(path, remote_name=path.name)
    row["drive_file_id"] = file_id
    row["status"] = STATUS_UPLOADED
    if delete_after_upload:
        path.unlink(missing_ok=True)


def _flush_pending(
    uploader: DriveUploader | None,
    rows: list[dict],
    delete_after_upload: bool,
) -> None:
    if uploader is None:
        return
    for row in rows:
        if row.get("status") in {STATUS_PENDING, STATUS_GENERATED}:
            _upload_and_cleanup(uploader, rows, row, delete_after_upload)


def run_jobs(
    *,
    index_csv: str | Path,
    drive_folder_id: str,
    auth_mode: str,
    credentials_json: str | None,
    token_json: str | None,
    device: str | None,
    seed: int | None = DEFAULT_SEED,
    min_cpu_free_gb: float,
    min_gpu_free_gb: float,
    delete_after_upload: bool,
    retry_oom: bool,
) -> None:
    index_path = Path(index_csv)
    rows = _load_rows_csv(index_path)
    if not rows:
        raise SystemExit(f"No rows found in {index_path}. Build the index first.")

    uploader = DriveUploader(
        drive_folder_id,
        auth_mode=auth_mode,
        credentials_json=credentials_json,
        token_json=token_json,
    )
    oom_path = index_path.parent / "oom_jobs.csv"
    oom_rows = []
    if oom_path.exists():
        oom_rows = _load_rows_csv(oom_path)

    try:
        for job_idx, row in enumerate(rows):
            status = row.get("status", STATUS_PENDING)
            path = Path(row["path"])
            if status == STATUS_UPLOADED:
                continue
            if status == STATUS_OOM and not retry_oom:
                continue

            if path.exists():
                row["status"] = STATUS_GENERATED
                _write_rows_csv(index_path, rows)
                _upload_and_cleanup(uploader, rows, row, delete_after_upload)
                _write_rows_csv(index_path, rows)
                print(f"[{job_idx + 1}/{len(rows)}] uploaded existing {path.name}")
                continue

            ensure_memory_budget(
                device=device,
                min_cpu_free_gb=min_cpu_free_gb,
                min_gpu_free_gb=min_gpu_free_gb,
            )

            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                data = generate_dataset(**_row_to_generate_kwargs(row, device, seed, job_idx))
                row["status"] = STATUS_GENERATED
                _write_rows_csv(index_path, rows)

                save_dataset(data, str(path))
                del data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                _upload_and_cleanup(uploader, rows, row, delete_after_upload)
                _write_rows_csv(index_path, rows)
                print(f"[{job_idx + 1}/{len(rows)}] uploaded {path.name}")
            except torch.OutOfMemoryError:
                row["status"] = STATUS_OOM
                oom_row = {k: row.get(k, "") for k in row.keys() if k != "drive_file_id"}
                oom_row["error"] = "OOM"
                oom_rows.append(oom_row)
                _write_rows_csv(index_path, rows)
                if oom_rows:
                    fieldnames = list(oom_rows[0].keys())
                    with oom_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(oom_rows)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[{job_idx + 1}/{len(rows)}] OOM {path.name}")
            except KeyboardInterrupt:
                print("Ctrl+C received. Uploading any generated files and deleting local safetensors.")
                _flush_pending(uploader, rows, delete_after_upload)
                _write_rows_csv(index_path, rows)
                if oom_rows:
                    fieldnames = list(oom_rows[0].keys())
                    with oom_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(oom_rows)
                return

        _write_rows_csv(index_path, rows)
        if oom_rows:
            fieldnames = list(oom_rows[0].keys())
            with oom_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(oom_rows)
    except KeyboardInterrupt:
        print("Ctrl+C received. Uploading any generated files and deleting local safetensors.")
        _flush_pending(uploader, rows, delete_after_upload)
        _write_rows_csv(index_path, rows)
        if oom_rows:
            fieldnames = list(oom_rows[0].keys())
            with oom_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(oom_rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build and run Drive-backed MHA/GQA dataset jobs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common_build_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--out-dir", "-o", required=True, help="Output directory for the dataset tree")
        parser.add_argument("--dtype", default="all", choices=["all", "float16", "bfloat16"], help="dtype filter")
        parser.add_argument("--batch", default="all", help="Comma-separated batch sizes or 'all'")
        parser.add_argument("--kv-mean", default="all", help="Comma-separated KV means or 'all'")
        parser.add_argument("--q-len", default="all", help="Comma-separated q lengths or 'all'")
        parser.add_argument("--head-dim", default="all", help="Comma-separated head dims or 'all'")
        parser.add_argument("--kv-dist", default="all", help="Comma-separated KV distribution names or 'all'")
        parser.add_argument("--start", type=int, default=0, help="Skip first N jobs after filtering")
        parser.add_argument("--split-parts", type=int, default=1, help="Split filtered jobs into K parts")
        parser.add_argument("--split-index", type=int, default=0, help="Which split part to build")
        parser.add_argument("--gqa-ratios", default="all", help="For GQA: comma-separated ratio folders like 8_1,8_2")

    build = sub.add_parser("build", help="Build a status CSV for MHA or GQA")
    build.add_argument("--attn", choices=["mha", "gqa"], required=True, help="Attention type")
    add_common_build_args(build)

    run = sub.add_parser("run", help="Run pending jobs from an index CSV")
    run.add_argument("--index-csv", required=True, help="Path to the index.csv file")
    run.add_argument("--drive-folder-id", required=True, help="Google Drive target folder ID")
    run.add_argument(
        "--auth",
        choices=["oauth", "service-account"],
        default="oauth",
        help="Authentication mode for Google Drive (default: oauth for personal Drive).",
    )
    run.add_argument(
        "--credentials-json",
        required=False,
        help="OAuth client secrets JSON (oauth) or service account JSON (service-account).",
    )
    run.add_argument(
        "--token-json",
        default="drive_token.json",
        help="OAuth token cache file (used only with --auth oauth).",
    )
    run.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, etc.")
    run.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed (default: 42)")
    run.add_argument("--min-cpu-free-gb", type=float, default=1.0, help="Minimum free CPU RAM in GB")
    run.add_argument("--min-gpu-free-gb", type=float, default=1.0, help="Minimum free GPU VRAM in GB")
    run.add_argument("--delete-after-upload", action="store_true", default=True, help="Delete local .safetensors after upload")
    run.add_argument("--keep-local", action="store_true", help="Keep local .safetensors after upload")
    run.add_argument("--retry-oom", action="store_true", help="Retry rows marked oom")

    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()

    if args.cmd == "build":
        out_dir = Path(args.out_dir)
        batches = _parse_csv_values(args.batch, int)
        kv_means = _parse_csv_values(args.kv_mean, int)
        q_lens = _parse_csv_values(args.q_len, int)
        head_dims = _parse_csv_values(args.head_dim, int)
        kv_dists = _parse_csv_values(args.kv_dist, str)
        gqa_ratios = _parse_csv_values(args.gqa_ratios, str)

        rows = _build_rows(
            args.attn,
            out_dir,
            args.dtype,
            batches=batches,
            kv_means=kv_means,
            q_lens=q_lens,
            head_dims=head_dims,
            kv_dists=kv_dists,
            gqa_ratios=gqa_ratios,
            start=args.start,
            split_parts=args.split_parts,
            split_index=args.split_index,
        )
        index_path = out_dir / "index.csv"
        _write_rows_csv(index_path, rows)
        print(f"Wrote {len(rows)} rows to {index_path}")
        return

    if args.cmd == "run":
        run_jobs(
            index_csv=args.index_csv,
            drive_folder_id=args.drive_folder_id,
            auth_mode=args.auth,
            credentials_json=args.credentials_json,
            token_json=args.token_json,
            device=args.device,
            seed=args.seed,
            min_cpu_free_gb=args.min_cpu_free_gb,
            min_gpu_free_gb=args.min_gpu_free_gb,
            delete_after_upload=args.delete_after_upload and not args.keep_local,
            retry_oom=args.retry_oom,
        )
        return


if __name__ == "__main__":
    main()

