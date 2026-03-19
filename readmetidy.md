# 🚀 How to Use Attention/KV-Cache Datasets

This guide covers three primary workflows for acquiring and interacting with the attention/KV-cache datasets, depending on whether your data is local, in the cloud, or needs to be generated on the fly. 

> **Important:** All scripts should be run from the **project root** (where `dataset.py` and the `tools/` directory are located).

---

## 1. Read Local Data (`load_safetensor.py`)

**Use Case:** You already have a `.safetensors` file downloaded locally and want to access the tensors (Q, K, V) and metadata (batch size, head counts, etc.).

### Command Line Usage

**Basic Run:**
```bash
python -m examples.load_safetensor path/to/gravity_float16_constant_kv500_b8_q3_1_1_64_gqa.safetensors
```

**Load directly onto a GPU:**
```bash
python -m examples.load_safetensor path/to/file.safetensors --device cuda
```

**Inspect file contents (list keys and shapes only):**
```bash
python -m examples.load_safetensor path/to/file.safetensors --list-only
```

### Accessing Data in Your Python Code
When integrating these files into your own scripts, the loaded data acts as a dictionary of tensors and scalar values:

```python
# Load the file into a dictionary
data = load_file("file.safetensors") 

# Attention Tensors
q = data["q"] # Shape: (batch, num_heads, q_length, head_size)
k = data["k"] # Shape: (batch, num_kv_heads, kv_len, head_size)
v = data["v"] # Shape: (batch, num_kv_heads, kv_len, head_size)

# Metadata (Scalars)
batch_size = data["batch_size"].item()
num_heads = data["num_heads"].item()
num_kv_heads = data["num_kv_heads"].item()
# Additional metadata: 'q_length', 'head_size'

# Optional KV Cache (if present in the file)
kv_lengths = data.get("kv_lengths")
kv_mask = data.get("kv_mask")
attn_out = data.get("attn_out") # Reference output
```

---

## 2. Download Data from Google Cloud (`get_data_from_cloud.py`)

**Use Case:** Your datasets are stored in Google Drive (e.g., generated via `tools/drive_generate.py`). This script handles authentication, downloads the `.safetensors` files, and can optionally load them directly onto a GPU.

### By Google Drive Folder
Downloads *all* `.safetensors` files located within the specified folder ID using standard OAuth.
```bash
python -m examples.get_data_from_cloud \
  --drive-folder-id YOUR_FOLDER_ID \
  --auth oauth \
  --credentials-json /path/to/client_secret.json \
  --token-json token.json \
  --out-dir ./downloaded
```

### By Index CSV
Downloads *only* the files explicitly listed in a CSV file. 
> **Note:** Your CSV must contain a `drive_file_id` column. Rows missing this column will be skipped.

```bash
# Using interactive OAuth
python -m examples.get_data_from_cloud \
  --index-csv mha_fp16/index.csv \
  --auth oauth \
  --credentials-json /path/to/client_secret.json \
  --out-dir ./downloaded

# Using a Service Account (ideal for automated environments)
python -m examples.get_data_from_cloud \
  --index-csv mha_fp16/index.csv \
  --auth service-account \
  --credentials-json /path/to/service_account.json \
  --out-dir ./downloaded
```

### Download & Test
You can append `--open-first` and `--device cuda` to immediately test-load the first downloaded file onto your GPU:
```bash
python -m examples.get_data_from_cloud ... --open-first --device cuda
```

---

## 3. Generate Custom Data Locally (`make_data_on_device.py`)

**Use Case:** You need to rapidly create one or more `.safetensors` files with highly specific parameters (custom batch sizes, dtypes, sequence lengths, KV distributions) without running the full dataset generation pipeline.

### Multi-Head Attention (MHA) Example
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

### Grouped-Query Attention (GQA) Example
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
