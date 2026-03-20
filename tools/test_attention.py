"""
Test that attn_out in generated datasets matches recomputed attention.
Currently runs 3 checks for MHA and 3 for GQA. MLA is a placeholder only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dataset import generate_dataset


def _recompute_attn_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_mask: torch.Tensor,
    scale: float | None = None,
    *,
    q_phase: str = "prefill",
) -> torch.Tensor:
    """Recompute MHA attention: same head count for Q,K,V. Used to verify stored attn_out."""
    B, _, Q_len, _ = q.shape
    Kv_len = k.shape[2]
    scale = scale or (q.shape[-1] ** -0.5)
    allowed = kv_mask.expand(B, 1, Q_len, Kv_len) > 0.5
    if q_phase == "causal":
        # Match dataset.py causal masking:
        # abs_q_pos = (seq_lens - Q_len) + q_pos; allow kv_pos <= abs_q_pos.
        seq_lens = kv_mask.sum(dim=-1, keepdim=True)
        q_pos = torch.arange(Q_len, device=q.device, dtype=seq_lens.dtype).view(1, 1, Q_len, 1)
        abs_q_pos = (seq_lens - Q_len) + q_pos
        kv_pos = torch.arange(Kv_len, device=q.device, dtype=seq_lens.dtype).view(1, 1, 1, Kv_len)
        causal_allow = kv_pos <= abs_q_pos
        allowed = allowed & causal_allow

    attn_mask = torch.where(allowed, 0.0, float("-inf")).to(q.dtype)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        scale=scale,
        dropout_p=0.0,
    )


def _recompute_attn_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_mask: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float | None = None,
    *,
    q_phase: str = "prefill",
) -> torch.Tensor | None:
    """
    Recompute GQA attention using the same masking semantics as dataset.py.
    """
    B, Hq, Q_len, _ = q.shape
    H_kv = k.shape[1]
    Kv_len = k.shape[2]
    scale = scale or (q.shape[-1] ** -0.5)

    # Expand K/V to Q head count (same approach dataset.py uses internally).
    if H_kv != Hq:
        repeat = Hq // H_kv
        k_exp = k.repeat_interleave(repeat, dim=1)
        v_exp = v.repeat_interleave(repeat, dim=1)
    else:
        k_exp = k
        v_exp = v

    allowed = kv_mask.expand(B, 1, Q_len, Kv_len) > 0.5
    if q_phase == "causal":
        seq_lens = kv_mask.sum(dim=-1, keepdim=True)
        q_pos = torch.arange(Q_len, device=q.device, dtype=seq_lens.dtype).view(1, 1, Q_len, 1)
        abs_q_pos = (seq_lens - Q_len) + q_pos
        kv_pos = torch.arange(Kv_len, device=q.device, dtype=seq_lens.dtype).view(1, 1, 1, Kv_len)
        causal_allow = kv_pos <= abs_q_pos
        allowed = allowed & causal_allow

    attn_mask = torch.where(allowed, 0.0, float("-inf")).to(q.dtype)
    return F.scaled_dot_product_attention(
        q, k_exp, v_exp,
        attn_mask=attn_mask,
        scale=scale,
        dropout_p=0.0,
    )


def _recompute_attn_mla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_mask: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float | None = None,
) -> torch.Tensor | None:
    """
    Placeholder for MLA attention recompute.
    TODO: Replace with your MLA kernel / custom calculation; then return the tensor
    so tests can compare to stored attn_out.
    """
    return None


def _check_close(
    stored: torch.Tensor,
    recomputed: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    label: str | None = None,
) -> bool:
    ok = bool(torch.allclose(stored.float(), recomputed.float(), atol=atol, rtol=rtol))
    if not ok:
        s = stored.float()
        r = recomputed.float()
        diff = (s - r).abs()
        max_abs = diff.max().item()
        rel = diff / s.abs().clamp_min(1e-6)
        max_rel = rel.max().item()
        tag = f"[{label}]" if label else ""
        print(f"{tag} max_abs_diff={max_abs:.4e}, max_rel_diff={max_rel:.4e}")
    return ok


def test_mha_1() -> None:
    data = generate_dataset(
        batch_size=2,
        dtype="float16",
        kv_cache=True,
        kv_cache_size_dist=("constant", 16),
        q_length=3,
        head_size=64,
        num_heads=2,
        num_kv_heads=2,
        attn_type="mha",
        seed=1,
        q_phase="causal",
        device="cpu",
    )
    out = _recompute_attn_mha(data["q"], data["k"], data["v"], data["kv_mask"], q_phase="causal")
    assert _check_close(data["attn_out"], out, label="MHA test 1"), "MHA test 1: attn_out mismatch"


def test_mha_2() -> None:
    data = generate_dataset(
        batch_size=1,
        dtype="bfloat16",
        kv_cache=True,
        kv_cache_size_dist=("constant", 64),
        q_length=5,
        head_size=128,
        num_heads=1,
        num_kv_heads=1,
        attn_type="mha",
        seed=42,
        q_phase="causal",
        device="cpu",
    )
    out = _recompute_attn_mha(data["q"], data["k"], data["v"], data["kv_mask"], q_phase="causal")
    assert _check_close(data["attn_out"], out, label="MHA test 2"), "MHA test 2: attn_out mismatch"


def test_mha_3() -> None:
    data = generate_dataset(
        batch_size=1,
        dtype="float16",
        kv_cache=True,
        kv_cache_size_dist=("uniform", 8, 32),
        q_length=1,
        head_size=64,
        num_heads=4,
        num_kv_heads=4,
        attn_type="mha",
        seed=123,
        q_phase="prefill",
        device="cpu",
    )
    out = _recompute_attn_mha(data["q"], data["k"], data["v"], data["kv_mask"], q_phase="prefill")
    assert _check_close(data["attn_out"], out, label="MHA test 3"), "MHA test 3: attn_out mismatch"


def test_gqa_1() -> None:
    data = generate_dataset(
        batch_size=2,
        dtype="float16",
        kv_cache=True,
        kv_cache_size_dist=("constant", 24),
        q_length=3,
        head_size=64,
        num_heads=8,
        num_kv_heads=1,
        attn_type="gqa",
        seed=10,
        q_phase="causal",
        device="cpu",
    )
    out = _recompute_attn_gqa(
        data["q"], data["k"], data["v"], data["kv_mask"],
        data["num_heads"].item(), data["num_kv_heads"].item(),
        q_phase="causal",
    )
    if out is not None:
        assert _check_close(data["attn_out"], out, label="GQA test 1"), "GQA test 1: attn_out mismatch"


def test_gqa_2() -> None:
    data = generate_dataset(
        batch_size=1,
        dtype="bfloat16",
        kv_cache=True,
        kv_cache_size_dist=("constant", 32),
        q_length=5,
        head_size=128,
        num_heads=8,
        num_kv_heads=2,
        attn_type="gqa",
        seed=20,
        q_phase="causal",
        device="cpu",
    )
    out = _recompute_attn_gqa(
        data["q"], data["k"], data["v"], data["kv_mask"],
        data["num_heads"].item(), data["num_kv_heads"].item(),
        q_phase="causal",
    )
    if out is not None:
        assert _check_close(data["attn_out"], out, label="GQA test 2"), "GQA test 2: attn_out mismatch"


def test_gqa_3() -> None:
    data = generate_dataset(
        batch_size=1,
        dtype="float16",
        kv_cache=True,
        kv_cache_size_dist=("normal", 50.0),
        q_length=1,
        head_size=64,
        num_heads=8,
        num_kv_heads=1,
        attn_type="gqa",
        seed=30,
        q_phase="prefill",
        device="cpu",
    )
    out = _recompute_attn_gqa(
        data["q"], data["k"], data["v"], data["kv_mask"],
        data["num_heads"].item(), data["num_kv_heads"].item(),
        q_phase="prefill",
    )
    if out is not None:
        assert _check_close(data["attn_out"], out, label="GQA test 3"), "GQA test 3: attn_out mismatch"


def test_mla_1() -> None:
    out = _recompute_attn_mla(
        torch.zeros(1, 1, 1, 64),
        torch.zeros(1, 1, 1, 64),
        torch.zeros(1, 1, 1, 64),
        torch.ones(1, 1, 1, 1),
        1, 1,
    )
    assert out is None, "MLA recompute should return None until implemented"


def test_mla_2() -> None:
    out = _recompute_attn_mla(
        torch.zeros(1, 2, 3, 64),
        torch.zeros(1, 2, 10, 64),
        torch.zeros(1, 2, 10, 64),
        torch.ones(1, 1, 1, 10),
        2, 2,
    )
    assert out is None


def test_mla_3() -> None:
    out = _recompute_attn_mla(
        torch.zeros(2, 8, 5, 128),
        torch.zeros(2, 1, 100, 128),
        torch.zeros(2, 1, 100, 128),
        torch.ones(2, 1, 1, 100),
        8, 1,
    )
    assert out is None


def run_all() -> None:
    tests = [
        test_mha_1,
        test_mha_2,
        test_mha_3,
        test_gqa_1,
        test_gqa_2,
        test_gqa_3,
    ]
    for t in tests:
        t()
        print(f"  OK {t.__name__}")
    print("All 6 checks passed (3 MHA, 3 GQA).")


if __name__ == "__main__":
    run_all()

