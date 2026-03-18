# How to use data for each job

Three ways to get or use the attention/KV-cache datasets:

---

## 1. Load a safetensor and get the data out

**Script:** `load_safetensor.py`

Use when you have a local `.safetensors` file and want to load it and access Q, K, V, and metadata (batch size, head counts, etc.).

**Run (from project root):**
```bash
python -m examples.load_safetensor path/to/gravity_float16_constant_kv500_b8_q3_1_1_64_gqa.safetensors
```

**Load tensors onto GPU:**
```bash
python -m examples.load_safetensor path/to/file.safetensors --device cuda
```

**Only list keys and shapes:**
```bash
python -m examples.load_safetensor path/to/file.safetensors --list-only
```

**How to get the data out in your own code:**
- `data = load_file("file.safetensors")` → dict of tensors
- **Query:** `q = data["q"]` — shape `(batch, num_heads, q_length, head_size)`
- **Key / Value:** `k = data["k"]`, `v = data["v"]` — shape `(batch, num_kv_heads, kv_len, head_size)`
- **Metadata (scalars):** `batch_size = data["batch_size"].item()`, `num_heads = data["num_heads"].item()`, `num_kv_heads = data["num_kv_heads"].item()`, `q_length`, `head_size`
- **KV cache (if present):** `kv_lengths`, `kv_mask`, `attn_out` (reference output)

---

## 2. Get data from Google Cloud and open it

**Script:** `get_data_from_cloud.py`

Use when your datasets are already in a Google Drive folder (e.g. after `tools/drive_generate.py run`). This script downloads the `.safetensors` files and can load them so you can use the tensors on GPU.

**By Drive folder (download all .safetensors in the folder):**
```bash
python -m examples.get_data_from_cloud \
  --drive-folder-id YOUR_FOLDER_ID \
  --auth oauth \
  --credentials-json /path/to/client_secret.json \
  --out-dir ./downloaded
```

**By index CSV (download only files listed in the index):**
```bash
python -m examples.get_data_from_cloud \
  --index-csv mha_fp16/index.csv \
  --auth oauth \
  --credentials-json /path/to/client_secret.json \
  --out-dir ./downloaded
```

**Download and open the first file on device:**
```bash
python -m examples.get_data_from_cloud ... --open-first --device cuda
```

---

## 3. Make data on device with custom settings

**Script:** `make_data_on_device.py`

Use when you want to generate one or more `.safetensors` locally with your own batch size, dtype, KV distribution, sequence lengths, etc., without running the full grid or Drive pipeline.

**Example:**
```bash
python -m examples.make_data_on_device \
  --output my_data.safetensors \
  --batch-size 16 \
  --dtype float16 \
  --kv-cache-size constant,2000 \
  --q-length 128 \
  --head-size 64 \
  --num-heads 8 \
  --attn-type mha \
  --device cuda
```

**GQA example:**
```bash
python -m examples.make_data_on_device \
  --output gqa_custom.safetensors \
  --batch-size 8 \
  --num-heads 8 \
  --num-kv-heads 1 \
  --attn-type gqa \
  --q-length 3 \
  --kv-cache-size normal,1000 \
  --device cuda
```

Run from the **project root** (where `dataset.py` and `tools/` live).
