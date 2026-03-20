"""
Generate small test datasets using dataset.py: 3 MHA, 3 GQA.
Saves to test_mha_1.safetensors, test_mha_2.safetensors, ... test_gqa_1..3.

MLA is not implemented yet in dataset.py, so MLA test files are not generated.

Seed: one constant seed (SEED below) is used for all files. Not stored in .safetensors.
Reproducible data, no meaningful speed cost.
"""

from __future__ import annotations

SEED = 42

from pathlib import Path

from dataset import generate_dataset, save_dataset


def main() -> None:
    out_dir = Path(__file__).parent
    device = "cpu"

    configs = [
        # --- 3 MHA ---
        {"name": "test_mha_1", "attn_type": "mha", "batch_size": 1, "dtype": "float16", "kv_cache_size_dist": ("constant", 32), "q_length": 3, "head_size": 64, "num_heads": 1, "num_kv_heads": 1},
        {"name": "test_mha_2", "attn_type": "mha", "batch_size": 2, "dtype": "bfloat16", "kv_cache_size_dist": ("constant", 16), "q_length": 5, "head_size": 128, "num_heads": 2, "num_kv_heads": 2},
        {"name": "test_mha_3", "attn_type": "mha", "batch_size": 1, "dtype": "float16", "kv_cache_size_dist": ("uniform", 8, 24), "q_length": 1, "head_size": 64, "num_heads": 4, "num_kv_heads": 4},
        # --- 3 GQA (covers 8:1 and 4:1) ---
        {"name": "test_gqa_1", "attn_type": "gqa", "batch_size": 1, "dtype": "float16", "kv_cache_size_dist": ("constant", 32), "q_length": 3, "head_size": 64, "num_heads": 8, "num_kv_heads": 1},  # 8:1
        {"name": "test_gqa_2", "attn_type": "gqa", "batch_size": 2, "dtype": "bfloat16", "kv_cache_size_dist": ("constant", 24), "q_length": 5, "head_size": 128, "num_heads": 8, "num_kv_heads": 2},  # 4:1
        {"name": "test_gqa_3", "attn_type": "gqa", "batch_size": 1, "dtype": "float16", "kv_cache_size_dist": ("normal", 20.0), "q_length": 1, "head_size": 64, "num_heads": 8, "num_kv_heads": 1},  # 8:1, normal KV
    ]

    for cfg in configs:
        name = cfg["name"]
        attn_type = cfg["attn_type"]
        path = out_dir / f"{name}.safetensors"
        kwargs = {k: v for k, v in cfg.items() if k not in ("name", "attn_type")}
        # Match benchmark semantics: q_len > 2 is an autoregressive chunk => causal (triangular) masking.
        kwargs["q_phase"] = "causal" if int(kwargs["q_length"]) > 2 else "prefill"
        kwargs["seed"] = SEED
        data = generate_dataset(
            kv_cache=True,
            device=device,
            attn_type=attn_type,
            **kwargs,
        )
        save_dataset(data, str(path))
        print(f"  {path.name}")

    print("Done. 6 files written (3 MHA, 3 GQA).")


if __name__ == "__main__":
    main()

