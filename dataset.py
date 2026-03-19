"""
dataset.py
Generate attention/KV-cache style datasets and save as safetensors.
Data values are drawn with no range restriction (standard normal).
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Callable, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

# KV cache size: "fixed" | int | ("uniform", min, max) | ("normal", mean, std) | ("poisson", lam) |
# ("constant", m) | ("uniform", m) [~0.5m..1.5m] | ("normal", m) [std=max(m/4,1)] |
# ("exp_soft", m) | ("exp_hard", m) | ("exp_soft_rev", m) | ("exp_hard_rev", m) |
# ("beta_soft", m) | ("beta_hard", m) | ("lognormal", m) | ("poisson", m) | [(len, weight), ...]
KVDist = Union[
    Literal["fixed"],
    int,
    tuple[Literal["uniform"], int, int],
    tuple[Literal["normal"], float, float],
    tuple[Literal["poisson"], float],  # (poisson, lam) or (poisson, m) for mean m
    tuple[Literal["constant"], float],
    tuple[Literal["uniform"], float],  # mean m -> uniform on [1, 2m-1]
    tuple[Literal["normal"], float],   # mean m -> std = m/4
    tuple[Literal["exp_soft"], float],
    tuple[Literal["exp_hard"], float],
    tuple[Literal["exp_soft_rev"], float],
    tuple[Literal["exp_hard_rev"], float],
    tuple[Literal["beta_soft"], float],   # Beta-based soft-edged distribution
    tuple[Literal["beta_hard"], float],   # Beta-based hard-edged distribution
    tuple[Literal["lognormal"], float],   # log-normal with mean m (sigma fixed)
    list[tuple[int, float]],
]


def _dtype_from_str(s: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[s]


# Attention types: MHA (multi-head), GQA (grouped query).
AttnType = Literal["mha", "gqa"]


DEFAULT_SEED = 42


def configure_reproducibility(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable dataset generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False


def _attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_mask: torch.Tensor,
    scale: float | None = None,
    num_kv_heads: int | None = None,
    q_phase: Literal["prefill", "causal"] = "prefill",
) -> torch.Tensor:
    """
    Scaled dot-product attention via PyTorch SDPA.
    q: (B, H_q, Q_len, head_size), k/v: (B, H_kv, Kv_len, head_size). For GQA, H_kv < H_q.
    If num_kv_heads is set and H_kv != H_q, K/V are repeated to match Q head count (GQA).
    q_phase:
        - "prefill": no causal restriction between Q positions and KV positions.
        - "causal": apply a lower-triangular restriction so Q index `i` can only
          attend to KV positions `<= i`.
    Returns (B, H_q, Q_len, head_size).
    """
    B, H_q, Q_len, _ = q.shape
    H_kv = k.shape[1]
    Kv_len = k.shape[2]
    if H_kv != H_q and num_kv_heads is not None:
        # GQA: expand K,V from num_kv_heads to num_heads by repeating each kv head
        repeat = H_q // H_kv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    allowed = kv_mask.expand(B, 1, Q_len, Kv_len) > 0.5
    if q_phase == "causal":
        # Causal/decode: Q tokens are the *next* tokens after the KV cache (not the last
        # Q_len of the sequence). So absolute Q position = seq_lens + q_pos.
        # 1. Get the actual length of each KV sequence in the batch (ignoring padding)
        seq_lens = kv_mask.sum(dim=-1, keepdim=True)

        # 2. Relative Q positions (0 to Q_len-1) -> absolute positions (after cache)
        q_pos = torch.arange(Q_len, device=q.device, dtype=seq_lens.dtype).view(1, 1, Q_len, 1)
        abs_q_pos = (seq_lens - Q_len) + q_pos

        # 3. Absolute KV positions (0 to Kv_len-1)
        kv_pos = torch.arange(Kv_len, device=q.device, dtype=seq_lens.dtype).view(1, 1, 1, Kv_len)

        # 4. Causal: allow KV_pos <= abs_q_pos (Q can attend to past and current)
        causal_allow = kv_pos <= abs_q_pos
        allowed = allowed & causal_allow

    attn_mask = torch.where(allowed, 0.0, float("-inf")).to(q.dtype)
    scale = scale or (q.shape[-1] ** -0.5)
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        scale=scale,
        dropout_p=0.0,
    )
    return out


def generate_dataset(
    *,
    batch_size: int,
    dtype: Union[torch.dtype, str],
    kv_cache: bool,
    kv_cache_size_dist: KVDist,
    q_length: int,
    head_size: int,
    num_heads: int = 1,
    num_kv_heads: int | None = None,
    attn_type: AttnType = "mha",
    compute_attn_out: bool = True,
    q_phase: Literal["prefill", "causal"] = "prefill",
    num_batches: int = 1,
    seed: int | None = DEFAULT_SEED,
    device: str | torch.device | None = None,
    deterministic: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Generate a dataset with tensors drawn from a standard normal (no range restriction) and optional KV cache.
    Supports MHA and GQA. Uses GPU if available.

    Args:
        batch_size: Number of samples per batch.
        dtype: torch.dtype or "float16"|"bfloat16".
        kv_cache: If True, include K and V cache tensors.
        attn_type: "mha" (multi-head), "gqa" (grouped query, num_heads Q / num_kv_heads KV).
        num_heads: Number of query heads. For GQA typically 8.
        num_kv_heads: Number of K/V heads. For MHA equals num_heads; for GQA < num_heads (e.g. 1).
        device: "cuda", "cpu", or torch.device. Default: cuda if available else cpu.
        kv_cache_size_dist: How to choose KV sequence length per batch:
            - "fixed" or int: fixed length (int = that length).
            - ("constant", m): always m.
            - ("uniform", m): roughly uniform on [0.5*m, 1.5*m].
            - ("normal", m): normal with mean m, std = max(m/4, 1).
            - ("exp_soft", m): exponential with softer tail around mean m.
            - ("exp_hard", m): exponential mixture with heavier tail.
            - ("exp_soft_rev", m): reversed soft exponential (heavier near upper bound).
            - ("exp_hard_rev", m): reversed hard exponential mixture.
            - ("beta_soft", m) / ("beta_hard", m): Beta-based soft/hard-edged shapes around the mean.
            - ("lognormal", m): log-normal with mean m (sigma fixed).
            - ("poisson", m): Poisson with mean m.
            - ("uniform", a, b), ("normal", mean, std), [(len, w), ...]: legacy forms.
        q_length: Query sequence length.
        head_size: Attention head dimension.
        num_batches: Number of batch groups to generate (each can have different kv_len if dist).
        compute_attn_out: If True, compute and return `attn_out`.
            If False, only generate `q`, `k`, `v`, `kv_lengths`, `kv_mask`, and scalar metadata.
        q_phase: How to mask attention across Q positions:
            - "prefill": current behavior (no causal Q-Q constraint).
            - "causal": apply KV_pos <= Q_pos triangular constraint.
        seed: Random seed for reproducibility.

    Returns:
        Dict of tensor names -> tensors, suitable for save_file().
    """
    if attn_type == "gqa":
        num_kv_heads = num_kv_heads if num_kv_heads is not None else 1
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads for GQA"
    else:
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

    seed = DEFAULT_SEED if seed is None else seed
    configure_reproducibility(seed, deterministic=deterministic)

    dtype = _dtype_from_str(dtype) if isinstance(dtype, str) else dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype)

    def _finalize_kv_values(values: torch.Tensor) -> torch.Tensor:
        """
        Round KV-length samples to integers and clamp to at least 1.
        """
        values = torch.clamp(values, min=1.0)
        return torch.round(values).to(torch.int64)

    def _match_target_mean(values: torch.Tensor, target_mean: float) -> torch.Tensor:
        """
        Adjust real-valued KV-length samples so that their integer-rounded version
        has mean close to target_mean (torch version of the numpy helper).
        """
        if values.numel() == 0:
            return values
        cur_mean = values.mean().clamp_min(1e-6)
        scaled = values * (target_mean / cur_mean)
        finalized = _finalize_kv_values(scaled).to(torch.float32)
        final_mean = finalized.mean().clamp_min(1e-6)
        corrected = finalized * (target_mean / final_mean)
        return _finalize_kv_values(corrected)

    def make_kv_sampler() -> Callable[[], int]:
        dist = kv_cache_size_dist
        if dist == "fixed":
            val = max(1, q_length)
            return lambda: val
        if isinstance(dist, int):
            val = max(1, dist)
            return lambda: val
        if isinstance(dist, (list, tuple)):
            n = len(dist)
            if n == 3 and dist[0] == "uniform":
                a, b = int(dist[1]), int(dist[2])
                return lambda: int(torch.randint(a, b + 1, (1,), device=device).item())
            if n == 2 and dist[0] == "uniform":
                m = float(dist[1])
                low = max(1, int(m * 0.5))
                high = max(low + 1, int(m * 1.5) + 1)
                return lambda: int(torch.randint(low, high, (1,), device=device).item())
            if n == 3 and dist[0] == "normal":
                mean, std = float(dist[1]), float(dist[2])
                return lambda: max(1, int(round(torch.normal(mean=mean, std=std, size=(1,), device=device).item())))
            if n == 2 and dist[0] == "normal":
                m = float(dist[1])
                std = max(m * 0.25, 1.0)
                return lambda: max(1, int(round(torch.normal(mean=m, std=std, size=(1,), device=device).item())))
            if n == 2 and dist[0] == "constant":
                val = max(1, int(round(float(dist[1]))))
                return lambda: val
            if n == 2 and dist[0] == "exp_soft":
                m = float(dist[1])
                scale = max(m * 0.8, 1.0)
                rate = 1.0 / scale
                return lambda: max(
                    1,
                    int(
                        round(
                            torch.distributions.Exponential(rate)
                            .sample((1,))
                            .item()
                        )
                    ),
                )
            if n == 2 and dist[0] == "exp_hard":
                m = float(dist[1])
                scale_small = max(m * 0.18, 1.0)
                scale_large = max(m * 4.2, 1.0)
                rate_small = 1.0 / scale_small
                rate_large = 1.0 / scale_large

                def _sample_exp_hard() -> int:
                    small = (
                        torch.distributions.Exponential(rate_small)
                        .sample((1,))
                        .item()
                    )
                    large = (
                        torch.distributions.Exponential(rate_large)
                        .sample((1,))
                        .item()
                    )
                    selector = torch.rand(1, device=device).item() < 0.8
                    val = small if selector else large
                    return max(1, int(round(val)))

                return _sample_exp_hard
            if n == 2 and dist[0] == "exp_soft_rev":
                m = float(dist[1])
                scale = max(m * 0.8, 1.0)
                rate = 1.0 / scale
                upper_bound = max(m * 3.0, 1.0)

                def _sample_exp_soft_rev() -> int:
                    base = (
                        torch.distributions.Exponential(rate)
                        .sample((1,))
                        .item()
                    )
                    val = max(1.0, upper_bound - base)
                    return max(1, int(round(val)))

                return _sample_exp_soft_rev
            if n == 2 and dist[0] == "exp_hard_rev":
                m = float(dist[1])
                scale_small = max(m * 0.18, 1.0)
                scale_large = max(m * 4.2, 1.0)
                rate_small = 1.0 / scale_small
                rate_large = 1.0 / scale_large
                upper_bound = max(m * 6.0, 1.0)

                def _sample_exp_hard_rev() -> int:
                    small = (
                        torch.distributions.Exponential(rate_small)
                        .sample((1,))
                        .item()
                    )
                    large = (
                        torch.distributions.Exponential(rate_large)
                        .sample((1,))
                        .item()
                    )
                    selector = torch.rand(1, device=device).item() < 0.8
                    base = small if selector else large
                    val = max(1.0, upper_bound - base)
                    return max(1, int(round(val)))

                return _sample_exp_hard_rev
            if n == 2 and dist[0] == "beta_soft":
                m = float(dist[1])
                scale = max(m * 2.0, 1.0)
                alpha, beta_param = 0.5, 0.5

                def _sample_beta_soft() -> int:
                    edge = (
                        torch.distributions.Beta(alpha, beta_param)
                        .sample((1,))
                        .item()
                    )
                    val = edge * scale
                    return max(1, int(round(val)))

                return _sample_beta_soft
            if n == 2 and dist[0] == "beta_hard":
                m = float(dist[1])
                base_scale = max(m * 4.5, 1.0)
                alpha, beta_param = 0.08, 0.08

                def _sample_beta_hard() -> int:
                    edge = (
                        torch.distributions.Beta(alpha, beta_param)
                        .sample((1,))
                        .item()
                    )
                    if edge < 0.5:
                        edge_val = 2.0 * ((edge / 2.0) ** 1.8)
                    else:
                        edge_val = 1.0 - 2.0 * (((1.0 - edge) / 2.0) ** 1.8)
                    val = edge_val * base_scale
                    if edge < 0.25:
                        val *= 0.35
                    elif edge > 0.75:
                        val *= 1.75
                    return max(1, int(round(val)))

                return _sample_beta_hard
            if n == 2 and dist[0] == "lognormal":
                m = float(dist[1])
                sigma = 0.5
                mu = math.log(max(m, 1.0)) - 0.5 * (sigma ** 2)
                return lambda: max(
                    1,
                    int(
                        round(
                            torch.distributions.LogNormal(mu, sigma)
                            .sample((1,))
                            .item()
                        )
                    ),
                )
            if n == 2 and dist[0] == "poisson":
                lam = float(dist[1])
                return lambda: max(1, int(torch.poisson(torch.tensor(lam, device=device).expand(1)).item()))
            if isinstance(dist, list) and dist and isinstance(dist[0], (list, tuple)):
                lengths = torch.tensor([x[0] for x in dist], dtype=torch.float32, device=device)
                weights = torch.tensor([x[1] for x in dist], dtype=torch.float32, device=device)
                weights = weights / weights.sum()
                return lambda: max(1, int(lengths[torch.multinomial(weights.unsqueeze(0), 1).item()].item()))
        return lambda: max(1, q_length)

    sample_kv_length = make_kv_sampler()

    out: dict[str, torch.Tensor] = {}

    # Q: (B, num_heads, q_length, head_size)
    q = rand(batch_size * num_batches, num_heads, q_length, head_size)
    out["q"] = q

    if kv_cache:
        kv_lengths = [sample_kv_length() for _ in range(batch_size * num_batches)]
        max_kv_len = max(kv_lengths)

        # K, V: (B, num_kv_heads, max_kv_len, head_size). For GQA, num_kv_heads < num_heads.
        k = rand(batch_size * num_batches, num_kv_heads, max_kv_len, head_size)
        v = rand(batch_size * num_batches, num_kv_heads, max_kv_len, head_size)

        # Mask: 1 where valid, 0 where padding [batch*num_batches, 1, 1, max_kv_len]
        mask = torch.zeros(
            batch_size * num_batches, 1, 1, max_kv_len,
            device=device, dtype=dtype,
        )
        for i, L in enumerate(kv_lengths):
            mask[i, :, :, :L] = 1

        out["k"] = k
        out["v"] = v
        out["kv_lengths"] = torch.tensor(kv_lengths, dtype=torch.int64, device=device)
        out["kv_mask"] = mask

        # Reference attention output (GQA: K/V expanded to num_heads inside _attention_forward)
        if compute_attn_out:
            out["attn_out"] = _attention_forward(
                q, k, v, mask, num_kv_heads=num_kv_heads, q_phase=q_phase
            )

    out["batch_size"] = torch.tensor([batch_size], dtype=torch.int64)
    out["q_length"] = torch.tensor([q_length], dtype=torch.int64)
    out["head_size"] = torch.tensor([head_size], dtype=torch.int64)
    out["num_heads"] = torch.tensor([num_heads], dtype=torch.int64)
    out["num_kv_heads"] = torch.tensor([num_kv_heads], dtype=torch.int64)
    # attn_type is in the filename (mha/gqa), not stored in the file

    # Move to CPU for saving (portable, safe for multiprocessing)
    if device.type != "cpu":
        out = {k: t.cpu() for k, t in out.items()}
    return out


def save_dataset(
    tensors: dict[str, torch.Tensor],
    path: str,
) -> None:
    """Save a dict of tensors to a safetensors file.

    safetensors requires tensors to be contiguous. Ensure this before saving
    so that non-contiguous outputs like `attn_out` do not raise errors.
    """
    contiguous_tensors = {name: t.contiguous() for name, t in tensors.items()}
    save_file(contiguous_tensors, path)


def _parse_kv_dist(s: str) -> KVDist:
    s = s.strip().lower()
    if s == "fixed":
        return "fixed"
    parts = [p.strip() for p in s.split(",")]
    if parts[0] == "uniform" and len(parts) == 3:
        return ("uniform", int(parts[1]), int(parts[2]))
    if parts[0] == "normal" and len(parts) == 3:
        return ("normal", float(parts[1]), float(parts[2]))
    if parts[0] == "poisson" and len(parts) == 2:
        return ("poisson", float(parts[1]))
    if ":" in s:
        # "64:0.5,128:0.3,256:0.2" discrete weights
        out: list[tuple[int, float]] = []
        for p in parts:
            len_str, w_str = p.split(":", 1)
            out.append((int(len_str.strip()), float(w_str.strip())))
        return out
    if len(parts) == 2:
        return ("uniform", int(parts[0]), int(parts[1]))
    raise ValueError(
        'kv-cache-size must be: "fixed", "uniform,min,max", "normal,mean,std", '
        '"poisson,lambda", "len1:w1,len2:w2,...", or "min,max"'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate safetensor dataset for attention/KV cache")
    ap.add_argument("--output", "-o", required=True, help="Output .safetensors path")
    ap.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--kv-cache", action="store_true", help="Include K and V cache tensors")
    ap.add_argument(
        "--kv-cache-size",
        default="fixed",
        help='KV length distribution: "fixed" | "uniform,min,max" | "normal,mean,std" | "poisson,lambda" | "len1:w1,len2:w2" | "min,max"',
    )
    ap.add_argument("--q-length", "-q", type=int, default=128, help="Query sequence length")
    ap.add_argument(
        "--q-phase",
        choices=["prefill", "causal"],
        default="prefill",
        help='Attention phase: "prefill" or "causal" (controls Q-V mask).',
    )
    ap.add_argument("--head-size", type=int, default=64, help="Head dimension")
    ap.add_argument("--num-heads", type=int, default=8, help="Number of query heads (MHA: same as KV; GQA: 8 typical)")
    ap.add_argument("--num-kv-heads", type=int, default=None, help="Number of K/V heads (GQA only; default 1 for gqa, num_heads for mha)")
    ap.add_argument("--attn-type", choices=["mha", "gqa"], default="mha", help="Attention type: mha or gqa")
    ap.add_argument("--num-batches", type=int, default=1, help="Number of batch groups (total samples = batch_size * num_batches)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)")
    ap.add_argument("--device", default=None, help="Device: cuda (default if available) or cpu")
    ap.add_argument("--deterministic", action="store_true", help="Enable strict deterministic PyTorch settings")
    args = ap.parse_args()

    kv_dist = _parse_kv_dist(args.kv_cache_size)
    num_kv = args.num_kv_heads
    if num_kv is None and args.attn_type == "gqa":
        num_kv = 1

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
        num_batches=args.num_batches,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
        q_phase=args.q_phase,
    )
    save_dataset(data, args.output)
    print(f"Saved to {args.output}: {list(data.keys())}")


if __name__ == "__main__":
    main()
