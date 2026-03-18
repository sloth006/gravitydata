"""
Verify that attn_out in a safetensor file matches recomputed MHA attention.
Usage: python verify_safetensors.py <file1.safetensors> [file2.safetensors ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def recompute_attn_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_mask: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    B, _, Q_len, _ = q.shape
    Kv_len = k.shape[2]
    scale = scale or (q.shape[-1] ** -0.5)
    attn_mask = torch.where(
        kv_mask.expand(B, 1, Q_len, Kv_len) > 0.5,
        0.0,
        float("-inf"),
    ).to(q.dtype)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        scale=scale,
        dropout_p=0.0,
    )


def check_close(stored: torch.Tensor, recomputed: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    ok = bool(torch.allclose(stored.float(), recomputed.float(), atol=atol, rtol=rtol))
    if not ok:
        diff = (stored.float() - recomputed.float()).abs()
        max_abs = diff.max().item()
        rel = diff / stored.float().abs().clamp_min(1e-6)
        max_rel = rel.max().item()
        print(f"  max_abs_diff={max_abs:.4e}, max_rel_diff={max_rel:.4e}")
    return ok


def verify(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        return False
    data = load_file(str(path))
    q = data["q"]
    k = data["k"]
    v = data["v"]
    kv_mask = data["kv_mask"]
    attn_out_stored = data["attn_out"]
    num_heads = data["num_heads"].item()
    num_kv_heads = data["num_kv_heads"].item()
    if num_heads != num_kv_heads:
        print(f"  Skipped (GQA): num_heads={num_heads}, num_kv_heads={num_kv_heads}; need MHA for this script.")
        return False
    recomputed = recompute_attn_mha(q, k, v, kv_mask)
    ok = check_close(attn_out_stored, recomputed)
    return ok


def main() -> None:
    paths = [
        r"C:\Users\woori\OneDrive\바탕 화면\mlsys\gravity\gravity_bfloat16_constant_kv1_b8_q3_1_1_128_mha.safetensors",
        r"C:\Users\woori\OneDrive\바탕 화면\mlsys\gravity\gravity_bfloat16_exp_hard_rev_kv4000_b32_q3_1_1_64_mha.safetensors",
    ]
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    all_ok = True
    for p in paths:
        print(f"Verifying: {Path(p).name}")
        ok = verify(p)
        print(f"  {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    print("All passed." if all_ok else "Some checks failed.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

