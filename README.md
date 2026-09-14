# MoGe v3 Standalone Panorama CLI (Dockerized)

A GPU-accelerated, self-contained **MoGe v3 Panorama Inference CLI** packaged into a standalone Docker container. Allows running MoGe panorama depth estimation directly from the command line on single images or entire folders with zero dependency on the root `moge` repo.

---

## 📁 Directory Structure

```text
Moge_CLI/
├── standalone_moge/               # 100% independent MoGe-3 core module
│   ├── custom_deps/               # Bundled dependencies (utils3d_moge, flex_gemm, DEPENDENCIES.md)
│   ├── model/                     # MoGe-3 ViT-L & ViT-G architectures, Sparse 3D UNet
│   ├── panorama/                  # Spherical geometry, multi-view splitter & merger
│   ├── utils/                     # Exporters, visualizers & download_weights.py
│   └── infer_panorama.py          # Canonical MoGe-3 Panorama CLI script
├── app.py                         # Cross-platform CLI entrypoint
├── Dockerfile                     # Multi-stage build (builder -> slim runtime)
├── docker-compose.yml             # Compose service definitions
├── requirements.txt               # Standalone dependencies
├── .dockerignore                  # Build context exclusions
├── .gitignore                     # Git tracking exclusions
└── README.md
```


---

## 🐳 Quick Start with Docker

### Step 1: Build Docker Image
```bash
cd Moge_CLI
docker build -t moge-panorama-cli .
```

### Step 2: Run Inference via CLI

#### 1. Single Image Inference (Output `depth.npy`)
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  moge-panorama-cli \
  -i /data/input_panorama.jpg \
  -o /data/output
```

#### 2. Visual Maps & Raw EXR Outputs
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  moge-panorama-cli \
  -i /data/input_panorama.jpg \
  -o /data/output \
  --maps
```

#### 3. Batch Folder Processing with High Precision (Refinement Steps = 5)
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  moge-panorama-cli \
  -i /data/input_folder \
  -o /data/output_folder \
  --resolution_level 9 \
  --refine_steps 5 \
  --maps
```

#### 4. Debug Mode (Save Splitted Perspective Views & Camera JSON Metadata)
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  moge-panorama-cli \
  -i /data/input_panorama.jpg \
  -o /data/output \
  --debug
```

---

## 📋 Full CLI Argument & Environment Variable Reference

All options can be configured via **CLI flags**, **Docker environment variables (`-e`)**, or inside `docker-compose.yml`. CLI flags always take highest precedence and override environment variables.

| Option | Shorthand | Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--input` | `-i` | *None (CLI Required)* | `Path` | **Mandatory** | Input panorama image or folder path. |
| `--output` | `-o` | `MOGE_OUTPUT` | `Path` | `./output` / `/data/outputs` | Output destination directory. |
| `--model` | `-m` | `MOGE_MODEL` | `Choice` | `vitl` | Model variant: `vitl` (370M) or `vitg` (1.25B). |
| `--pretrained` | | `MOGE_PRETRAINED` | `String` | `None` | Custom checkpoint path or HF repo ID. |
| `--checkpoint_dir` | | `CHECKPOINT_DIR` | `Path` | `/checkpoints` or `checkpoints` | Checkpoints storage directory. Automatically downloads the configured model if not present. |
| `--device` | | `MOGE_DEVICE` | `String` | `cuda` | Execution device (`cuda`, `cuda:0`, `cpu`). |
| `--fp16` | | `MOGE_FP16` | `Flag` | `False` | Enables FP16 precision. |
| `--resize` | | `MOGE_RESIZE` | `Int` | `None` | Max dimension ceiling (default: keep original). |
| `--resolution_level` | | `MOGE_RESOLUTION_LEVEL` | `Int [0-9]` | `9` | Inference resolution level (0-9). |
| `--num_tokens` | | `MOGE_NUM_TOKENS` | `Int` | `None` | Number of tokens (overrides resolution level). |
| `--refine_steps` | | `MOGE_REFINE_STEPS` | `Int` | `3` | Sparse 3D refinement steps for v3. |
| `--split_resolution` | | `MOGE_SPLIT_RESOLUTION` | `Int` | `512` | Perspective view tile resolution. |
| `--batch_size` | | `MOGE_BATCH_SIZE` | `Int` | `4` | Perspective views inference batch size. |
| `--debug` | | `MOGE_DEBUG` | `Flag` | `False` | Saves debug views, distance maps, cameras.json. |
| `--maps` | | `MOGE_MAPS` | `Flag` | `False` | Saves `depth_vis.png`, `normal_vis.png`, `.exr` files. |
| `--depth_npy / --no-depth_npy` | | `MOGE_DEPTH_NPY` | `Bool` | `True` | Saves primary `depth.npy` float32 array. |
| `--points_npy` | | `MOGE_POINTS_NPY` | `Flag` | `False` | Saves 3D coordinates `points.npy` array. |
