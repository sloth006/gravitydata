"""
Get desired data from Google Drive and open it.

Use this when your datasets were generated and uploaded to a Drive folder
(e.g. by tools/drive_generate.py run). This script downloads .safetensors
files and loads them so you can use the tensors (e.g. on GPU for benchmarking).

Run from project root:
  python -m examples.get_data_from_cloud --drive-folder-id ID --auth oauth --credentials-json path/to/client_secret.json --out-dir ./downloaded

Or with an index CSV (only download files listed there):
  python -m examples.get_data_from_cloud --index-csv mha_fp16/index.csv --auth oauth --credentials-json path/to/client_secret.json --out-dir ./downloaded
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running from repo root or from examples/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetensors.torch import load_file

from tools.drive_utils import DriveUploader


def _load_index_rows(index_csv: Path) -> list[dict]:
    if not index_csv.exists():
        return []
    with index_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download .safetensors from Google Drive and open them."
    )
    ap.add_argument("--drive-folder-id", help="Drive folder ID to list and download from")
    ap.add_argument(
        "--index-csv",
        type=Path,
        help="Index CSV (e.g. mha_fp16/index.csv) with path, drive_file_id. If set, only these files are downloaded.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("downloaded"), help="Local directory to save files")
    ap.add_argument("--auth", choices=["oauth", "service-account"], default="oauth")
    ap.add_argument("--credentials-json", type=Path, help="OAuth client secret or service account JSON")
    ap.add_argument("--token-json", type=Path, default=None, help="OAuth token cache (optional)")
    ap.add_argument(
        "--open-first",
        action="store_true",
        help="After download, load the first .safetensors and print keys/shapes and move to device example",
    )
    ap.add_argument("--device", default="cuda", help="Device to load tensors to when using --open-first")
    args = ap.parse_args()

    folder_id = args.drive_folder_id or ""
    if not args.index_csv and not folder_id:
        ap.error("Provide either --drive-folder-id or --index-csv (with drive_file_id column).")
    # DriveUploader needs a folder_id string; only used when listing. For index-csv we only download by file_id.
    if not folder_id:
        folder_id = "placeholder"

    if args.auth == "oauth" and not args.credentials_json:
        ap.error("--credentials-json is required for oauth.")

    uploader = DriveUploader(
        folder_id or "",
        auth_mode=args.auth,
        credentials_json=str(args.credentials_json) if args.credentials_json else None,
        token_json=args.token_json,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    to_download: list[tuple[str, str]] = []  # (file_id, local_name)

    if args.index_csv:
        rows = _load_index_rows(args.index_csv)
        for row in rows:
            file_id = (row.get("drive_file_id") or "").strip()
            if not file_id:
                continue
            # Use path column as subpath under out_dir, or just the filename
            path_val = row.get("path", "")
            if path_val:
                local_name = Path(path_val).name
            else:
                local_name = f"file_{file_id}.safetensors"
            to_download.append((file_id, local_name))
    else:
        files = uploader.list_files_in_folder(folder_id)
        for f in files:
            if f["name"].endswith(".safetensors"):
                to_download.append((f["id"], f["name"]))

    if not to_download:
        print("No .safetensors files to download.")
        return

    print(f"Downloading {len(to_download)} file(s) to {out_dir}")
    for file_id, name in to_download:
        path = out_dir / name
        if path.exists():
            print(f"  skip (exists): {name}")
            continue
        try:
            uploader.download_file(file_id, path)
            print(f"  ok: {name}")
        except Exception as e:
            print(f"  error {name}: {e}")

    if args.open_first:
        # Find first .safetensors in out_dir (prefer one we just downloaded)
        first = None
        for p in sorted(out_dir.glob("*.safetensors")):
            first = p
            break
        if not first:
            print("No .safetensors in out-dir to open.")
            return
        print(f"\nOpening first file: {first.name}")
        data = load_file(str(first))
        print("Keys and shapes:")
        for k, v in data.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")
        # Example: move to device for use (e.g. benchmarking)
        import torch
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        data_on_device = {k: t.to(device) if hasattr(t, "to") else t for k, t in data.items()}
        print(f"\nTensors moved to {device}. Use data_on_device['q'], data_on_device['k'], etc. for attention.")


if __name__ == "__main__":
    main()
