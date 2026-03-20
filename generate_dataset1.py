"""
generate_dataset2.py

Generate a fixed subset of MHA + GQA datasets where:
  - MHA: q=1 and q=5
  - GQA: q=5 only
and save to .safetensors,
with optional Google Drive upload + Ctrl+C resume via index.csv.

Configs:
  MHA:
  - num_heads=32, num_kv_heads=32, head_dim=64
  - num_heads=16, num_kv_heads=16, head_dim=128
  GQA:
  - num_heads=32, num_kv_heads=8, head_dim=128
  - num_heads=64, num_kv_heads=8, head_dim=128

This generator uses a CUDA-friendly attention path implemented with
"memory streaming" (chunked KV softmax) to reduce peak attention memory.
If generation/attention OOMs, the job is logged into oom_jobs.csv.
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


STATUS_PENDING = "pending"
STATUS_GENERATED = "generated"
STATUS_UPLOADED = "uploaded"
STATUS_OOM = "oom"

INDEX_FILENAME = "index.csv"
OOM_FILENAME = "oom_jobs.csv"


DTYPE = "bfloat16"
HEAD_SIZE_128 = 128
HEAD_SIZE_64 = 64

# Same batch sizes and KV distributions as generate_dataset1.py
BATCH_SIZES = [1, 8, 16, 32, 64, 128]
KV_MEANS = [1, 500, 1000, 2000]
KV_DISTRIBUTIONS = [
    ("constant", "constant"),
    ("exp_soft", "exp_soft"),
    ("exp_hard", "exp_hard"),
    ("exp_soft_rev", "exp_soft_rev"),
    ("exp_hard_rev", "exp_hard_rev"),
]

Q_LENGTHS = [1, 5]

# MHA: (num_heads, num_kv_heads, head_dim)
MHA_TYPES: list[tuple[int, int, int]] = [
    (32, 32, HEAD_SIZE_64),
    (16, 16, HEAD_SIZE_128),
]

# GQA: (num_heads, num_kv_heads, head_dim)
GQA_TYPES: list[tuple[int, int, int]] = [
    (32, 8, HEAD_SIZE_128),
    (64, 8, HEAD_SIZE_128),
]


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


def _upload_csv_to_drive_replace(
    uploader: DriveUploader,
    local_path: Path,
) -> str | None:
    if not local_path.exists():
        return None
    try:
        for f in uploader.list_files_in_folder():
            if f.get("name") == local_path.name and f.get("id"):
                uploader.service.files().delete(fileId=f["id"]).execute()
    except Exception:
        pass
    return uploader.upload_file(local_path, remote_name=local_path.name)


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
        "path",
        "dtype",
        "kv_dist_type",
        "kv_mean",
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
    if not oom_rows:
        return
    fieldnames = list(oom_rows[0].keys())
    with oom_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(oom_rows)


def _build_job_rows(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for num_heads, num_kv_heads, head_dim in MHA_TYPES:
        ratio_dir = out_dir / f"mha_{num_heads}_{num_kv_heads}" / "bf16"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        for batch in BATCH_SIZES:
            for kv_mean in KV_MEANS:
                dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                for kv_dist_type, dist_key in dists:
                    for q_len in Q_LENGTHS:
                        name = _filename(
                            DTYPE,
                            kv_dist_type,
                            kv_mean,
                            batch,
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
                                "batch": batch,
                                "q_len": q_len,
                                "num_kv_heads": num_kv_heads,
                                "num_heads": num_heads,
                                "head_dim": head_dim,
                                "attn_type": "mha",
                                "status": STATUS_PENDING,
                                "kv_dist": (dist_key, kv_mean),
                            }
                        )

    for num_heads, num_kv_heads, head_dim in GQA_TYPES:
        ratio_dir = out_dir / f"gqa_{num_heads}_{num_kv_heads}" / "bf16"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        for batch in BATCH_SIZES:
            for kv_mean in KV_MEANS:
                dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                for kv_dist_type, dist_key in dists:
                    # For GQA, only generate q_length=5.
                    for q_len in [5]:
                        name = _filename(
                            DTYPE,
                            kv_dist_type,
                            kv_mean,
                            batch,
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
                                "batch": batch,
                                "q_len": q_len,
                                "num_kv_heads": num_kv_heads,
                                "num_heads": num_heads,
                                "head_dim": head_dim,
                                "attn_type": "gqa",
                                "status": STATUS_PENDING,
                                "kv_dist": (dist_key, kv_mean),
                            }
                        )
    return rows


@torch.inference_mode()
def _attention_forward_chunked_mha(
    q: torch.Tensor,  # (B,H,Q,d) on CUDA
    k: torch.Tensor,  # (B,H,K,d) on CPU or CUDA
    v: torch.Tensor,  # (B,H,K,d) on CPU or CUDA
    kv_lengths: torch.Tensor,  # (B,) int64 (CPU or CUDA)
    *,
    q_phase: str,
    chunk_kv: int = 128,
) -> torch.Tensor:
    """
    Memory-streaming attention for MHA that avoids allocating (B,H,Q,K).

    Mask semantics match `dataset.py`:
    - "prefill": only padding mask (kv_pos < kv_length)
    - "causal": additionally apply the `causal_allow` condition from `dataset.py`.
    """
    assert q.is_cuda
    if q_phase not in {"prefill", "causal"}:
        raise ValueError(f"Unknown q_phase: {q_phase}")

    B, H, Q_len, d = q.shape
    K = k.shape[2]
    scale = d**-0.5

    q_device = q.device
    kv_lengths = kv_lengths.to(q_device)

    # Float32 streaming accumulators (prevents dtype mismatch).
    qf = q.float()
    m = torch.full((B, H, Q_len), float("-inf"), device=q_device, dtype=torch.float32)
    l = torch.zeros((B, H, Q_len), device=q_device, dtype=torch.float32)
    out = torch.zeros((B, H, Q_len, d), device=q_device, dtype=torch.float32)

    q_pos = torch.arange(Q_len, device=q_device, dtype=kv_lengths.dtype).view(1, 1, Q_len)
    if q_phase == "causal":
        # dataset.py causal: abs_q_pos = (seq_lens - Q_len) + q_pos
        abs_q_pos = (kv_lengths.view(B, 1, 1) - Q_len) + q_pos  # (B,1,Q_len)
    else:
        abs_q_pos = None

    for start in range(0, K, chunk_kv):
        end = min(K, start + chunk_kv)
        kv_pos_chunk = torch.arange(start, end, device=q_device, dtype=kv_lengths.dtype)  # (chunk,)

        # Move only the current KV chunk to GPU.
        k_chunk = k[:, :, start:end, :].to(q_device)  # (B,H,chunk,d)
        v_chunk = v[:, :, start:end, :].to(q_device)  # (B,H,chunk,d)
        kf_chunk = k_chunk.float()
        vf_chunk = v_chunk.float()

        # scores: (B,H,Q,chunk)
        scores = torch.matmul(qf, kf_chunk.transpose(-1, -2)) * scale

        # padding: kv_pos < kv_length
        padding_valid = kv_pos_chunk.view(1, 1, 1, -1) < kv_lengths.view(B, 1, 1, 1)  # (B,1,1,chunk)->broadcast
        allowed = padding_valid

        if q_phase == "causal":
            # causal_allow: kv_pos <= abs_q_pos
            causal_allow = kv_pos_chunk.view(1, 1, 1, -1) <= abs_q_pos.view(B, 1, Q_len, 1)  # (B,1,Q,chunk)
            allowed = allowed & causal_allow

        scores = scores.masked_fill(~allowed, float("-inf"))

        max_chunk = scores.max(dim=-1).values  # (B,H,Q)
        m_new = torch.maximum(m, max_chunk)

        # Safe m_new: replace -inf with 0.0 just for the subtraction to prevent NaNs.
        m_safe = m_new.masked_fill(m_new == float("-inf"), 0.0)
        exp_scores = torch.exp(scores - m_safe.unsqueeze(-1))  # (B,H,Q,chunk)

        exp_m_factor = torch.exp(m - m_new)  # (B,H,Q)
        exp_m_factor = torch.nan_to_num(exp_m_factor, nan=0.0)

        l = l * exp_m_factor + exp_scores.sum(dim=-1)
        out = out * exp_m_factor.unsqueeze(-1) + torch.matmul(exp_scores, vf_chunk)
        m = m_new

    out = out / l.unsqueeze(-1).clamp(min=1e-9)
    return out.to(q.dtype)


def _attention_forward_chunked_gqa(
    q: torch.Tensor,  # (B,Hq,Q,d) on CUDA
    k: torch.Tensor,  # (B,Hkv,K,d) on CPU or CUDA
    v: torch.Tensor,  # (B,Hkv,K,d) on CPU or CUDA
    kv_lengths: torch.Tensor,  # (B,) int64
    *,
    q_phase: str,
    chunk_kv: int = 128,
) -> torch.Tensor:
    """
    Memory-streaming attention for GQA without explicitly repeating K/V heads.
    Groups query heads by their KV head mapping (consistent with `repeat_interleave`).
    """
    assert q.is_cuda
    if q_phase not in {"prefill", "causal"}:
        raise ValueError(f"Unknown q_phase: {q_phase}")

    B, Hq, Q_len, d = q.shape
    Hkv = k.shape[1]
    if Hq % Hkv != 0:
        raise ValueError("For GQA, Hq must be divisible by Hkv")
    repeat_q = Hq // Hkv
    K = k.shape[2]
    scale = d**-0.5

    q_device = q.device
    kv_lengths = kv_lengths.to(q_device)
    qf_group = q.float().view(B, Hkv, repeat_q, Q_len, d)  # (B,Hkv,repeat,Q,d)

    # Float32 accumulators for grouped heads.
    m = torch.full((B, Hkv, repeat_q, Q_len), float("-inf"), device=q_device, dtype=torch.float32)
    l = torch.zeros((B, Hkv, repeat_q, Q_len), device=q_device, dtype=torch.float32)
    out = torch.zeros((B, Hkv, repeat_q, Q_len, d), device=q_device, dtype=torch.float32)

    q_pos = torch.arange(Q_len, device=q_device, dtype=kv_lengths.dtype).view(1, Q_len)  # (1,Q)
    if q_phase == "causal":
        abs_q_pos = (kv_lengths.view(B, 1) - Q_len) + q_pos  # (B,Q)
    else:
        abs_q_pos = None

    for start in range(0, K, chunk_kv):
        end = min(K, start + chunk_kv)
        kv_pos_chunk = torch.arange(start, end, device=q_device, dtype=kv_lengths.dtype)  # (chunk,)

        k_chunk = k[:, :, start:end, :].to(q_device)  # (B,Hkv,chunk,d)
        v_chunk = v[:, :, start:end, :].to(q_device)  # (B,Hkv,chunk,d)
        kf_chunk = k_chunk.float().unsqueeze(2)
        vf_chunk = v_chunk.float().unsqueeze(2)

        # scores: (B,Hkv,repeat,Q,chunk)
        scores = torch.matmul(
            qf_group,
            kf_chunk.transpose(-1, -2),  # (B,Hkv,d,chunk)
        ) * scale

        # padding_valid: kv_pos < kv_length
        padding_valid = kv_pos_chunk.view(1, 1, 1, 1, -1) < kv_lengths.view(B, 1, 1, 1, 1)  # (B,1,1,1,chunk)
        allowed = padding_valid

        if q_phase == "causal":
            # causal_allow: kv_pos <= abs_q_pos
            causal_allow = kv_pos_chunk.view(1, 1, 1, 1, -1) <= abs_q_pos.view(B, 1, 1, Q_len, 1)  # (B,1,1,Q,chunk)
            allowed = allowed & causal_allow

        scores = scores.masked_fill(~allowed, float("-inf"))

        max_chunk = scores.max(dim=-1).values  # (B,Hkv,repeat,Q)
        m_new = torch.maximum(m, max_chunk)

        # Safe m_new: replace -inf with 0.0 just for the subtraction to prevent NaNs.
        m_safe = m_new.masked_fill(m_new == float("-inf"), 0.0)
        exp_scores = torch.exp(scores - m_safe.unsqueeze(-1))  # (B,Hkv,repeat,Q,chunk)

        exp_m_factor = torch.exp(m - m_new)  # (B,Hkv,repeat,Q)
        exp_m_factor = torch.nan_to_num(exp_m_factor, nan=0.0)

        l = l * exp_m_factor + exp_scores.sum(dim=-1)
        out = out * exp_m_factor.unsqueeze(-1) + torch.matmul(exp_scores, vf_chunk)
        m = m_new

    out = out / l.unsqueeze(-1).clamp(min=1e-9)
    out = out.reshape(B, Hq, Q_len, d)
    return out.to(q.dtype)


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
    chunk_kv: int = 128,
) -> None:
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    index_path = out_path / INDEX_FILENAME
    oom_path = out_path / OOM_FILENAME

    fresh_rows = _build_job_rows(out_path)
    rows = _merge_with_existing_index(fresh_rows, index_path)
    total = len(rows)

    def is_pending(row: dict) -> bool:
        status = (row.get("status") or "").strip().lower()
        if status in {STATUS_GENERATED, STATUS_UPLOADED}:
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
                q_phase = "causal" if q_len > 2 else "prefill"
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
                    compute_attn_out=False,
                    q_phase=q_phase,
                    num_batches=1,
                    seed=seed,
                    device="cpu",
                    deterministic=deterministic,
                )

                # Compute attention on GPU with chunked KV softmax.
                q = data["q"].to(device)
                k = data["k"]
                v = data["v"]
                kv_lengths = data["kv_lengths"].to(torch.int64)

                if row["attn_type"] == "mha":
                    attn_out = _attention_forward_chunked_mha(
                        q, k, v, kv_lengths, q_phase=q_phase, chunk_kv=chunk_kv
                    )
                else:
                    attn_out = _attention_forward_chunked_gqa(
                        q, k, v, kv_lengths, q_phase=q_phase, chunk_kv=chunk_kv
                    )
                # Move back to CPU for saving.
                data["attn_out"] = attn_out.cpu()

                save_dataset(data, row["path"])
                del data, q, k, v, attn_out
                row["status"] = STATUS_GENERATED

                if uploader is not None:
                    file_id = uploader.upload_file(Path(row["path"]), remote_name=Path(row["path"]).name)
                    row["drive_file_id"] = file_id
                    row["status"] = STATUS_UPLOADED
                    if delete_after_upload:
                        Path(row["path"]).unlink(missing_ok=True)

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
                print(f"  OOM: {Path(row['path']).name} (logged to {OOM_FILENAME})")
            finally:
                _write_index(rows, index_path)

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
    ap = argparse.ArgumentParser(description="Generate MHA subset (q=5 only) with Drive + resume.")
    ap.add_argument("--output-dir", "-o", type=Path, default=Path("datasets2"), help="Output directory")
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
    ap.add_argument("--chunk-kv", type=int, default=128, help="KV chunk size for streaming attention softmax")

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

