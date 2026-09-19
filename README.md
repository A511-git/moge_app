# MoGe v3 Standalone Panorama CLI (Dockerized)

A GPU-accelerated, self-contained **MoGe v3 Panorama Inference CLI** packaged into a standalone Docker container. Allows running MoGe panorama depth estimation directly from the command line on single images or entire folders with zero dependency on the root `moge` repo.

---

## 🌟 Key Features

- **🚀 GPU-Accelerated Pipeline**: Matrix-free CUDA conjugate gradient Poisson solver for ultra-fast spherical depth merging and GPU-accelerated 3D Gaussian Splatting.
- **🌐 3D Gaussian Splatting (`splat.ply`)**: Direct export of panoramic Gaussian splat point clouds compatible with WebGL viewers (SuperSplat, PlayCanvas, Luma, GSplat).
- **📦 100% Self-Contained**: Zero external reliance on the original root `moge` repo; bundled with CUDA-compatible `utils3d_moge` and `flex_gemm`.
- **⚙️ Configurable Presets**: Specialized indoor (15m cutoff) and outdoor (80m cutoff with sky removal) splat presets, customizable sampling strides, scales, and thickness.

---

## 📁 Directory Structure

```text
Moge_CLI/
├── standalone_moge/               # 100% independent MoGe-3 core module
│   ├── custom_deps/               # Bundled dependencies (utils3d_moge, flex_gemm, DEPENDENCIES.md)
│   ├── model/                     # MoGe-3 ViT-L & ViT-G architectures, Sparse 3D UNet
│   ├── panorama/                  # Spherical geometry, multi-view splitter & merger
│   ├── utils/                     # Exporters, visualizers, splat generators & download_weights.py
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

#### 2. Generate 3D Gaussian Splat (`splat.ply`)
```bash
# Generate full-resolution 3D Gaussian Splat for WebGL viewers
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  moge-panorama-cli \
  -i /data/input_panorama.jpg \
  -o /data/output \
  --ply

# Outdoor panorama with custom sampling stride (lighter splat for web viewer)
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  moge-panorama-cli \
  -i /data/outdoor_panorama.jpg \
  -o /data/output \
  --ply \
  --no-ply_is_indoor \
  --ply_stride 2 \
  --ply_max_depth 80.0
```

#### 3. Visual Maps & Raw EXR Outputs
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  moge-panorama-cli \
  -i /data/input_panorama.jpg \
  -o /data/output \
  --maps
```

#### 4. Batch Folder Processing with High Precision (Refinement Steps = 5)
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  moge-panorama-cli \
  -i /data/input_folder \
  -o /data/output_folder \
  --resolution_level 9 \
  --refine_steps 5 \
  --maps \
  --ply
```

#### 5. Debug Mode (Save Splitted Perspective Views & Camera JSON Metadata)
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
| `--ply` | | `MOGE_PLY` | `Flag` | `False` | Save 3D Gaussian Splatting `splat.ply` file. |
| `--ply_is_indoor / --no-ply_is_indoor` | | `MOGE_PLY_IS_INDOOR` | `Bool` | `True` | Scene preset (indoor: max 15m cutoff; outdoor: max 80m cutoff with sky removal). |
| `--ply_stride` | | `MOGE_PLY_STRIDE` | `Int` | `1` | Pixel sampling stride (1 = full res, 2 = half res for lightweight WebGL viewing). |
| `--ply_scale` | | `MOGE_PLY_SCALE` | `Float` | `1.2` | Global splat radius scale multiplier. |
| `--ply_thickness` | | `MOGE_PLY_THICKNESS` | `Float` | `0.2` | Splat disc thickness ratio. |
| `--ply_min_depth` | | `MOGE_PLY_MIN_DEPTH` | `Float` | `0.1` | Minimum distance threshold in meters. |
| `--ply_max_depth` | | `MOGE_PLY_MAX_DEPTH` | `Float` | `None` | Maximum distance cutoff in meters (`15.0m` indoor, `80.0m` outdoor by default). |
