"""
Dump a .safetensors file contents to a .txt file (all keys, shapes, dtypes, values).
Usage: python safetensor_to_txt.py <input.safetensors> [output.txt]
  If output.txt is omitted, writes to <input>.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors.torch import load_file


def tensor_to_str(t) -> str:
    flat = t.flatten().tolist()
    if len(flat) <= 200:
        return str(flat)
    return str(flat[:100]) + " ... " + str(flat[-100:])


def dump_to_txt(safetensor_path: str | Path, txt_path: str | Path | None = None) -> None:
    path = Path(safetensor_path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = Path(txt_path) if txt_path else path.with_name(path.stem + ".txt")
    data = load_file(str(path))
    lines = [f"=== {path.name} FULL contents ===", ""]
    for key in sorted(data.keys()):
        t = data[key]
        lines.append(f"Key: {key}")
        lines.append(f"  shape: {tuple(t.shape)}")
        lines.append(f"  dtype: {t.dtype}")
        lines.append("  values:")
        lines.append(tensor_to_str(t))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python safetensor_to_txt.py <input.safetensors> [output.txt]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    dump_to_txt(inp, out)


if __name__ == "__main__":
    main()

