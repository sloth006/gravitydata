## Gravity — Attention benchmark datasets

Generate `.safetensors` datasets for benchmarking attention kernels (MHA and GQA).  
Each file contains inputs (Q, K, V, masks) and a reference output `attn_out` so you can measure speed and check correctness.

---

## Requirements

- **Python** 3.9+
- **PyTorch** 2.0+
- **safetensors**

Install Python packages only inside a virtual environment (PEP 668–friendly).

---

## Setup (virtual environment)

Create and activate a venv, then install dependencies.

**Linux / macOS / WSL** (venv uses `bin/`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (Git Bash)** (venv uses `Scripts/`):

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python3 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt):**

```cmd
python3 -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

After activation your prompt shows `(venv)`. Run all project commands from this environment.

---

## If you see “externally-managed-environment”

On Debian/Ubuntu‑style systems you may see:

> error: externally-managed-environment  
> This environment is externally managed …

This means you are trying to use `pip` in the **system Python**, which is protected by PEP 668. The fix is:

```bash
cd /path/to/gravity

python3 -m venv venv
source venv/bin/activate          # on Linux/macOS/WSL

pip install -r requirements.txt   # now installs into venv, not system Python
```

After that:

- Always activate the venv (`source venv/bin/activate` or the Windows variants above) **before** running `pip` or any of the project scripts.
- Do **not** use `--break-system-packages` unless you know exactly what you are doing; the venv approach is the safe, recommended solution.

---

## Project Layout

- Root: main generation scripts and core docs
- `tools/`: Drive upload, index management, verification, and test utilities
- `examples/`: sample outputs and reference files

---

## Dataset contents

Each `.safetensors` file includes:

| Key | Description |
|-----|-------------|
| `q` | Query tensor `(batch, num_heads, q_length, head_size)` |
| `k` | Key cache `(batch, num_kv_heads, kv_len, head_size)` |
| `v` | Value cache (same shape as `k`) |
| `kv_mask` | Validity mask, 1 = valid, 0 = padding |
| `kv_lengths` | Actual KV length per batch (for variable-length KV) |
| `attn_out` | Reference attention output (ground truth) |
| `batch_size`, `q_length`, `head_size`, `num_heads`, `num_kv_heads` | Scalar metadata |

All tensors are drawn from a **standard normal** distribution (no manual clipping).

---

## Quick start

### 1. Sanity check: small test datasets

```bash
# Bash (Git Bash / WSL)
bash run_tests.sh
```

This runs `tools/generate_test_data.py` (6 small files: 3 MHA, 3 GQA) and `tools/test_attention.py` (verifies `attn_out` matches recomputed attention).

### 2. Generate a single file (interactive)

```bash
bash run_dataset.sh
```

You choose dtype, KV distribution, and output path; the script can create one file or the full grid.

### 3. Full grid (mixed MHA + GQA)

```bash
python generate_all.py -o datasets

# Multi‑GPU example (4 GPUs)
python generate_all.py -o datasets --num-gpus 4

# Single GPU with 4 CPU workers
python generate_all.py -o datasets -j 4
```

File naming:

`gravity_{dtype}_{kv_dist_type}_kv{kv_mean}_b{batch}_q{q_len}_{kv_heads}_{query_heads}_{head_dim}_{attn_type}.safetensors`

Example:  
`gravity_float16_constant_kv500_b8_q3_2_8_64_gqa.safetensors`

---

## Separate sweeps: MHA and GQA

You can also generate MHA-only and GQA-only datasets with more control.

### `generate_all_mha.py` — MHA only

```bash
# fp16 and bf16
python generate_all_mha.py --output-dir datasets_mha

# Only fp16
python generate_all_mha.py --output-dir mha_fp16 --dtype float16

# Resume from a job index (skip first 1850 jobs)
python generate_all_mha.py --output-dir mha_fp16 --dtype float16 --start 1850

# Strict repeatability
python generate_all_mha.py --output-dir mha_fp16 --dtype float16 --deterministic
```

- Progress is based on a fixed job ordering over:
  - batch size, dtype, KV mean, KV distribution, query length, head size.
- If a file already exists at a path, the job is **skipped** and counted as done.
- OOM jobs are appended to `oom_jobs.csv` in the same output directory.

### `generate_all_gqa.py` — GQA only

```bash
# Both 4:1 and 8:1 GQA variants, fp16 + bf16
python generate_all_gqa.py --output-dir datasets_gqa

# Only fp16 GQA, starting from job 0
python generate_all_gqa.py --output-dir gqa_fp16 --dtype float16 --start 0

# Run 50% of the jobs at a time
python generate_all_gqa.py --output-dir gqa_fp16 --dtype float16 --split-parts 2 --split-index 0
python generate_all_gqa.py --output-dir gqa_fp16 --dtype float16 --split-parts 2 --split-index 1

# Strict repeatability
python generate_all_gqa.py --output-dir gqa_fp16 --dtype float16 --deterministic
```
python generate_all_gqa.py --output-dir gqa_bf16 --dtype bfloat16 --split-parts 6 --split-index 0

Options:

- `--start N` skips the first N jobs (0‑based) in the fixed grid order.
- `--split-parts K` and `--split-index I` (0 ≤ I < K) let you run only a fraction of the jobs per run (e.g. 50% with K=2).
- Existing files are skipped; OOM jobs are recorded in `oom_jobs.csv`.

---

## Managing OOM jobs and incomplete grids

### `tools/run_oom_jobs.py` — retry failed jobs

If a sweep produced an `oom_jobs.csv`, you can retry those jobs on a different device:

```bash
# Example: retry MHA fp16 OOMs on GPU 1
python tools/run_oom_jobs.py mha_fp16/oom_jobs.csv --device cuda:1

# Example: retry GQA fp16 OOMs on GPU 1
python tools/run_oom_jobs.py gqa_fp16/oom_jobs.csv --device cuda:1
```

This regenerates only the jobs listed in the CSV and overwrites the corresponding `.safetensors` files.

### `tools/index_status.py` — status‑aware index + “pending only” mode

`tools/index_status.py` can build a full index for a directory and then run only the jobs that are still missing.

**Build a status index:**

```bash
# MHA, fp16
python tools/index_status.py build --out-dir mha_fp16 --attn mha --dtype float16

# MHA, bf16
python tools/index_status.py build --out-dir mha_bf16 --attn mha --dtype bfloat16

# GQA, fp16
python tools/index_status.py build --out-dir gqa_fp16 --attn gqa --dtype float16
```

This writes `<out-dir>/index.csv` with columns:

- `path, dtype, kv_dist_type, kv_mean, batch, q_len, num_kv_heads, num_heads, head_dim, attn_type, status`
- `status = "1"` → file exists  
- `status = "oom"` → filename appears in `<out-dir>/oom_jobs.csv`  
- `status = "0"` → not generated yet

**Run only pending jobs (`status == "0"`):**

```bash
python tools/index_status.py run mha_fp16/index.csv --device cuda
python tools/index_status.py run gqa_fp16/index.csv --device cuda:1
```

The script:

- Regenerates each `status == "0"` row using `generate_dataset` and `save_dataset`.
- Updates `status` to `"1"` on success, or `"oom"` if it OOMs.
- Rewrites `index.csv` in place with updated statuses.

### `tools/setup_layout.py` — create folders and indices in one step

To create the standard directories and put `index.csv` in the right place for each:

```bash
python tools/setup_layout.py
```

This does the following under the project root:

- Ensures `mha_fp16/`, `mha_bf16/`, and `gqa_fp16/` exist.
- For each of those, calls `index_status.build(...)` to write:
  - `mha_fp16/index.csv`   (MHA, float16)
  - `mha_bf16/index.csv`   (MHA, bfloat16)
  - `gqa_fp16/index.csv`   (GQA, float16)

You can pass a custom base directory (if you run from somewhere else) with:

```bash
python tools/setup_layout.py --base-dir /path/to/gravity
```

After running this, use `tools/index_status.py run <dir>/index.csv` to generate only the missing jobs.

### `tools/drive_generate.py` — upload to Google Drive and delete local files

This is the Drive-backed workflow. It works for both MHA and GQA, can build filtered subsets, checks memory before generation, uploads each finished file to Google Drive, and deletes the local `.safetensors` after upload.

**1. Build a queue CSV**

```bash
# MHA, fp16
python tools/drive_generate.py build --attn mha --dtype float16 --out-dir mha_fp16

# GQA, fp16
python tools/drive_generate.py build --attn gqa --dtype float16 --out-dir gqa_fp16

# Example subset
python tools/drive_generate.py build --attn mha --dtype float16 --out-dir mha_fp16_subset --batch 32,64 --kv-mean 4000,8000 --q-len 1,3 --head-dim 64,128
```

Useful build filters:

- `--batch`
- `--kv-mean`
- `--q-len`
- `--head-dim`
- `--kv-dist`
- `--start`
- `--split-parts`
- `--split-index`

**2. Run and upload to Drive**

```bash
python tools/drive_generate.py run \
  --index-csv mha_fp16/index.csv \
  --auth oauth \
  --credentials-json /path/to/oauth-client-secrets.json \
  --drive-folder-id YOUR_FOLDER_ID \
  --device cpu
```

What it does:

- Generates any row with `status = 0`.
- Uploads the finished `.safetensors` to Google Drive.
- Deletes the local `.safetensors` after upload.
- Writes the CSV after every important step so progress is preserved.
- Uses `status` values like `0`, `generated`, `uploaded`, and `oom` so you can track the job life cycle.
- For personal Drive use, the first successful login creates `drive_token.json` and reuses it next time.

**Interrupt handling**

If you press `Ctrl+C`, the script will:

- Upload any file that has already been generated locally.
- Update the CSV.
- Delete the local `.safetensors` files that were posted.

**Google Drive setup for a personal Drive**

- Create an OAuth client in Google Cloud Console.
- Download the OAuth client secrets JSON file.
- The first time you run the uploader, it will open an authorization flow and save a token file (default: `drive_token.json`).
- Pass the OAuth client secrets path with `--credentials-json`.
- Pass the folder ID with `--drive-folder-id`.
- If you want the tool to remember your login, keep the saved token file in place.

**Optional service-account mode**

- If you still want to use a service account, pass `--auth service-account`.
- In that mode, `--credentials-json` should point to the service-account JSON.

**Memory safety**

- The Drive workflow checks free CPU RAM before each job.
- If you run on GPU, it also clears CUDA cache and checks free VRAM.
- You can tune the thresholds with `--min-cpu-free-gb` and `--min-gpu-free-gb`.

---

## Attention types

- **MHA** — classical multi‑head attention; same number of heads for Q, K, and V.
- **GQA** — grouped‑query attention:
  - 4:1 (8 query heads, 2 KV heads)
  - 8:1 (8 query heads, 1 KV head)

The attention type is encoded in the filename suffix (`..._mha.safetensors` or `..._gqa.safetensors`).

---

## KV length distributions

When generating with variable KV length, each sample’s KV length is drawn from one of:

- `constant` — fixed length (the grid’s `kv_mean`)
- `uniform` — roughly uniform on \([1, 2m-1]\) where \(m\) is `kv_mean`
- `normal` — normal with mean \(m\), std \(\approx m/4\)
- `exp_soft`, `exp_hard`, `exp_soft_rev`, `exp_hard_rev` — exponential‑style tails and their reversed variants
- `beta_soft`, `beta_hard` — Beta‑based U‑shaped families on \([1, 2m-1]\)
- `lognormal` — log‑normal with mean \(m\)
- `poisson` — Poisson with mean \(m\)

For `kv_mean == 1`, only the `constant` distribution is used.

---

## Device and performance notes

- **Device selection:** by default we use CUDA if available, else CPU. Override with `--device cuda`, `--device cuda:1`, or `--device cpu` in `dataset.py`, `generate_all_mha.py`, or `generate_all_gqa.py`.
- **Multi‑GPU:** `generate_all.py --num-gpus N` launches N processes, one per GPU (round‑robin over jobs).
- **Seeds:** `tools/generate_test_data.py` and the generation scripts default to a fixed seed of 42; seeds are not stored in the `.safetensors` files.
- **Determinism:** pass `--deterministic` to the generation scripts if you want stricter PyTorch deterministic settings. Same seed plus same PyTorch/device setup should then reproduce the same files.

---

## Example: load and use a dataset

```python
from safetensors.torch import load_file
import torch.nn.functional as F

data = load_file("mha_fp16/fp16/gravity_float16_constant_kv1000_b8_q3_1_1_64_mha.safetensors")
q, k, v, mask = data["q"], data["k"], data["v"], data["kv_mask"]
ref_out = data["attn_out"]

# Run your kernel
# out = my_attn(q, k, v, mask)
# assert torch.allclose(out.float(), ref_out.float(), atol=1e-2)
```

Use the file naming convention and `index.csv` to select the subset of attention problems you want to benchmark. 