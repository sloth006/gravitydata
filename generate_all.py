"""
generate_all.py
Generate one safetensors file per combination of batch size, dtype, kv length, q length, head size.
Optionally per distribution of kv length (extend KV_DISTRIBUTIONS later).

Speed:
  - Multi-GPU: --num-gpus N runs N processes, one per GPU (best when you have multiple GPUs).
  - Multi-core: --workers N uses N CPU workers (good when using 1 GPU or CPU only).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue
from pathlib import Path

from dataset import DEFAULT_SEED, generate_dataset, save_dataset

# --- Grid (one file per combination) ---
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]
DTYPES = ["float16", "bfloat16"]
KV_LENGTHS = [1, 500, 1000, 2000, 4000, 8000]
# q_length: 1, 3, 5
Q_LENGTHS: list[int | str] = [1, 3, 5]
HEAD_SIZES = [128, 64]

# Attention types: MHA and GQA (4:1 and 8:1).
# Each entry: (attn_type, num_heads, num_kv_heads). GQA 4:1 = 8q/2kv, GQA 8:1 = 8q/1kv.
ATTN_TYPES: list[tuple[str, int, int]] = [
    ("mha", 1, 1),
    ("gqa", 8, 2),   # GQA ratio 4:1 (8 query heads, 2 KV heads)
    ("gqa", 8, 1),   # GQA ratio 8:1 (8 query heads, 1 KV head)
]

# q_length value when grid says "prefill"
PREFILL_Q_LENGTH = 128

# KV length distributions. When kv_mean == 1 only "constant" is used; otherwise all 8.
# Each entry: (dist_type for filename, kv_dist for generate_dataset).
# m = mean kv length (from grid).
KV_DISTRIBUTIONS: list[tuple[str, object]] = [
    ("constant", "constant"),          # 1. constant = m
    ("uniform", "uniform"),            # 2. uniform on [1, 2m-1]
    ("normal", "normal"),              # 3. normal mean m, std = m/4
    ("exp_soft", "exp_soft"),          # 4. soft exponential tail around mean m
    ("exp_hard", "exp_hard"),          # 5. harder exponential mixture tail
    ("exp_soft_rev", "exp_soft_rev"),  # 6. reversed soft exponential (heavier near upper bound)
    ("exp_hard_rev", "exp_hard_rev"),  # 7. reversed hard exponential mixture
    ("lognormal", "lognormal"),        # 8. log-normal mean m (sigma fixed)
    ("poisson", "poisson"),            # 9. Poisson mean m
]


def _resolve_q_length(q: int | str) -> int:
    if q == "prefill":
        return PREFILL_Q_LENGTH
    return int(q)


def _filename(
    dtype: str,
    kv_dist_type: str,
    kv_mean: int | float,
    batch: int,
    q_len: int | str,
    num_kv_heads: int,
    num_heads: int,
    head: int,
    attn_type: str,
) -> str:
    q = q_len if q_len == "prefill" else str(q_len)
    return f"gravity_{dtype}_{kv_dist_type}_kv{kv_mean}_b{batch}_q{q}_{num_kv_heads}_{num_heads}_{head}_{attn_type}.safetensors"


def _build_jobs(
    out_dir: Path,
    seed: int | None = DEFAULT_SEED,
    device: str | None = None,
    deterministic: bool = False,
) -> list[tuple]:
    jobs = []
    job_idx = 0
    for batch in BATCH_SIZES:
        for dtype in DTYPES:
            for kv_mean in KV_LENGTHS:
                dists = KV_DISTRIBUTIONS if kv_mean > 1 else [KV_DISTRIBUTIONS[0]]
                for kv_dist_type, dist_key in dists:
                    kv_dist = (dist_key, kv_mean)
                    for q_len in Q_LENGTHS:
                        for head in HEAD_SIZES:
                            for attn_type, num_heads, num_kv_heads in ATTN_TYPES:
                                jobs.append((str(out_dir), batch, dtype, kv_mean, kv_dist, q_len, head, attn_type, num_heads, num_kv_heads, seed, job_idx, device, deterministic))
                                job_idx += 1
    return jobs


def _run_one(job: tuple, device_override: str | None = None) -> str:
    (out_dir, batch, dtype, kv_mean, kv_dist, q_len, head, attn_type, num_heads, num_kv_heads, base_seed, _job_idx, device, deterministic) = job
    if device_override is not None:
        device = device_override
    q_resolved = _resolve_q_length(q_len)
    kv_dist_type = kv_dist[0] if isinstance(kv_dist, (list, tuple)) else "fixed"
    q_phase = "prefill" if q_resolved > 2 else "causal"
    name = _filename(dtype, kv_dist_type, kv_mean, batch, q_len, num_kv_heads, num_heads, head, attn_type)
    path = Path(out_dir) / name
    seed = base_seed if base_seed is not None else DEFAULT_SEED
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
        seed=seed,
        device=device,
        deterministic=deterministic,
    )
    save_dataset(data, str(path))
    return str(path)


def _run_chunk_on_gpu(jobs_chunk: list[tuple], gpu_id: int) -> list[str]:
    """Run a chunk of jobs on a single GPU (cuda:gpu_id). Used by multi-GPU mode."""
    device = f"cuda:{gpu_id}"
    return [_run_one(job, device_override=device) for job in jobs_chunk]


def generate_all(
    out_dir: str | Path,
    *,
    seed: int | None = DEFAULT_SEED,
    workers: int = 1,
    num_gpus: int = 0,
    device: str | None = None,
    deterministic: bool = False,
) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Multi-GPU: each GPU gets a process and a chunk of jobs (round-robin)
    if num_gpus > 1:
        jobs = _build_jobs(out_dir, seed, device=None, deterministic=deterministic)
        chunks: list[list[tuple]] = [[] for _ in range(num_gpus)]
        for i, j in enumerate(jobs):
            chunks[i % num_gpus].append(j)
        result_queue: Queue = mp.Queue()
        procs = []
        for gpu_id in range(num_gpus):
            p = Process(target=_run_chunk_and_put, args=(chunks[gpu_id], gpu_id, result_queue))
            p.start()
            procs.append(p)
        saved = []
        for _ in range(num_gpus):
            paths = result_queue.get()
            saved.extend(paths)
        for p in procs:
            p.join()
        return saved
    # Single-GPU or CPU: use device from args (or default) and optional worker pool
    jobs = _build_jobs(out_dir, seed, device, deterministic=deterministic)
    if workers <= 1:
        saved = [_run_one(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            saved = pool.map(_run_one, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
    return saved


def _run_chunk_and_put(jobs_chunk: list[tuple], gpu_id: int, queue: Queue) -> None:
    """Run chunk on cuda:gpu_id and put list of paths on queue. (Used in multi-GPU Process.)"""
    paths = _run_chunk_on_gpu(jobs_chunk, gpu_id)
    queue.put(paths)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate safetensor files for all grid combinations")
    ap.add_argument("--output-dir", "-o", default="datasets", help="Output directory for .safetensors files")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    ap.add_argument("--workers", "-j", type=int, default=1, help="Parallel workers (single-GPU/CPU mode)")
    ap.add_argument("--num-gpus", "-G", type=int, default=0, help="Multi-GPU: N processes, one per GPU (overrides --workers when > 1)")
    ap.add_argument("--device", default=None, help="Device: cuda (default if available) or cpu")
    ap.add_argument("--deterministic", action="store_true", help="Enable strict deterministic settings")
    args = ap.parse_args()

    paths = generate_all(
        args.output_dir,
        seed=args.seed,
        workers=args.workers,
        num_gpus=args.num_gpus,
        device=args.device,
        deterministic=args.deterministic,
    )
    print(f"Generated {len(paths)} files in {args.output_dir}")
    for p in paths[:5]:
        print(f"  {p}")
    if len(paths) > 5:
        print(f"  ... and {len(paths) - 5} more")


if __name__ == "__main__":
    main()
