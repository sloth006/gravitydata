from __future__ import annotations

import argparse
from pathlib import Path

from tools.index_status import build_index


def setup_default_layout(base_dir: str | Path) -> None:
    """
    Ensure the default dataset folders exist and build status indices for them.

    - mha_fp16: MHA, float16
    - mha_bf16: MHA, bfloat16
    - gqa_fp16: GQA, float16
    """
    base = Path(base_dir)

    configs = [
        ("mha_fp16", "mha", "float16"),
        ("mha_bf16", "mha", "bfloat16"),
        ("gqa_fp16", "gqa", "float16"),
    ]

    for rel_dir, attn, dtype in configs:
        out_dir = base / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        index_path = build_index(out_dir, attn=attn, dtype_filter=dtype)
        print(f"[{attn} {dtype}] index written to {index_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create standard dataset folders and build index.csv files."
    )
    ap.add_argument(
        "--base-dir",
        default=".",
        help="Project root directory (default: current directory).",
    )
    args = ap.parse_args()

    setup_default_layout(args.base_dir)


if __name__ == "__main__":
    main()

