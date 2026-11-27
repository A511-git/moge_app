# Feature: Multi-View ARKit Perspective Depth Fusion (Near/Far Wall Scale Align)

> Status: **spec / desired capability** — not implemented as a first-class path yet.  
> Related research notes: [moge_arkit_world_axis_handoff.md](moge_arkit_world_axis_handoff.md), plan report `panorama_pipeline_report`.  
> Concrete sample data: `C:\Users\A-511\Downloads\single_face`

---

## 1. What I actually want

I have **two (or more) perspective photos** of the **same real-world surface** (e.g. the same wall segment), captured in a **single ARKit world**, with known:

- RGB images
- Per-image intrinsics (`fx`, `fy`, `cx`, `cy`)
- Per-image extrinsics / poses (`cameraToWorld` in OpenCV RDF → ARKit world)

Typical hard case:

- Photo A taken **near** the wall
- Photo B taken **farther** from the wall (same wall, different camera origin / FOV)
- MoGe will emit depth for each independently, possibly at **different relative scales**
- After placing both into ARKit world with poses, camera baselines stay correct, but the reconstructed wall surfaces can sit **in front / behind each other** instead of on one plane

### Desired outcome

1. Run MoGe on each perspective image using **that image’s true FOV / intrinsics** (not hardcoded 90° cube faces).
2. Unproject each depth/point map into **one ARKit world** using the provided poses.
3. **Match relative scale** across overlapping geometry so shared wall segments land on the **same surface**.
4. Fuse into one point cloud / mesh (concat + voxel, or TSDF later).
5. Export a debug PLY that makes double-wall vs aligned-wall visually obvious.

This is **not** “split one equirect into 12 faces and Poisson-merge on a shared sphere.” That existing path solves a different problem (same optical center). This feature is for **translated cameras** with known ARKit poses.

---

## 2. Motivating sample: `single_face`

Folder: `C:\Users\A-511\Downloads\single_face`

| File | Role |
|------|------|
| `9oPhlqgwrM_11.jpg` | Farther view of warehouse wall (1440×1920) |
| `sAHiABwkaK_11.jpg` | Nearer view of same wall region (1440×1920) |
| `9oPhlqgwrM_poses_world_opencv_rdf.json` | 106 ARKit frames; use `index == 11` |
| `sAHiABwkaK_poses_world_opencv_rdf.json` | 106 ARKit frames; use `index == 11` |

### Measured geometry (frame 11)

| | `9oPhlqgwrM` (far) | `sAHiABwkaK` (near) |
|--|--|--|
| Origin (ARKit m) | `[0.82, 0.26, 0.70]` | `[4.13, -0.01, 7.44]` |
| Intrinsics | `fx=fy≈1336`, `cx≈715`, `cy≈964` | same ballpark |
| FOV | **≈56.64° × 71.40°** | **≈56.62° × 71.38°** |
| Look-angle delta | — | **≈14.8°** |
| Baseline between cameras | **≈7.51 m** | |
| Distance along look to shared region | **≈8.85 m** | **≈1.47 m** |
| Closest-approach ray separation | **≈1.10 m** | |

Pose JSON convention (already converted):

- World: ARKit (`+Y` up, meters)
- Camera: OpenCV RDF (`+X` right, `+Y` down, `+Z` forward)
- Matrices: column-major `cameraToWorld` / `worldToCamera`
- Per-frame `intrinsicsImage`: `{fx, fy, cx, cy}`

References:

- Sample analysis captured in plan section “Concrete case study: single_face”
- Axis / RDF conversion spec: [moge_arkit_world_axis_handoff.md](moge_arkit_world_axis_handoff.md)

---

## 3. What exists today (and why it is not enough)

### 3.1 MoGe is single-image

`model.infer(image, fov_x=...)` estimates geometry for **one** image. It does **not** take poses or fuse multi-view.

References:

- Original: [original_moge/MoGe/moge/model/v1.py](original_moge/MoGe/moge/model/v1.py), [v2.py](original_moge/MoGe/moge/model/v2.py), [v3.py](original_moge/MoGe/moge/model/v3.py)
- Standalone: [standalone_moge/model/v3.py](standalone_moge/model/v3.py) (`infer`, optional `fov_x`, metric `metric_scale`)
- Prior note: [moge_arkit_world_axis_handoff.md](moge_arkit_world_axis_handoff.md) (“Not directly out of the box”)

### 3.2 Panorama path: 12 faces, hardcoded 90°, shared origin

```text
equirect → get_panorama_cameras (12×90°) → split_panorama_image
        → MoGe.infer(..., fov_x=90)
        → merge_panorama_depth (Poisson on ∇log(distance))
        → spherical unproject from ONE center
```

References:

- Cameras / split / merge: [standalone_moge/utils/panorama.py](standalone_moge/utils/panorama.py)
  - `get_panorama_cameras()` — FOV **hardcoded 90°**, shared eye `[0,0,0]`
  - `split_panorama_image(...)`
  - `merge_panorama_depth(...)` / `merge_panorama_depth_gpu(...)` / `solve_poisson_cg_torch(...)`
- Original merge (SciPy `lsmr`): [original_moge/MoGe/moge/utils/panorama.py](original_moge/MoGe/moge/utils/panorama.py) `merge_panorama_depth`
- Scripts: [standalone_moge/infer_panorama.py](standalone_moge/infer_panorama.py), [original_moge/MoGe/moge/scripts/infer_panorama.py](original_moge/MoGe/moge/scripts/infer_panorama.py)

**Scale behavior of Poisson merge (important):**

- Works on **gradients / Laplacian of** `log(distance)`, not absolute meters.
- A **constant** per-face scale cancels in `∇log(d)`, so shape can stitch without an explicit scale solve.
- There is **no** per-face RANSAC / median-ratio scale match.
- Absolute meters of the merged pano are only weakly set (coarse init / solver nullspace).
- Optional external scale is **per panorama jobId**, not per face: `moge_depth_scales.json` via [standalone_moge/utils/moge_world_poses.py](standalone_moge/utils/moge_world_poses.py) `load_depth_scales` / `get_depth_scale`.

### 3.3 Multi-pano ARKit placement (still not near/far wall scale align)

[standalone_moge/infer_multi_panorama.py](standalone_moge/infer_multi_panorama.py) can:

- Split each equirect into 12 faces
- Pose stations into MoGe/ARKit-related world via [moge_world_poses.py](standalone_moge/utils/moge_world_poses.py)
- Apply per-`jobId` `depth_scale`
- Concat / voxel (`all_faces_world`) or Poisson on a **shared** sphere (`all_faces_sphere` — wrong when stations translate)

Helpers already useful for this feature:

- `transform_points_to_world(points_cam, T_c2w)`
- `voxel_downsample_points(...)`
- `write_rgb_ply(...)`
- `compute_panorama_moge_pose` / `arkit_to_moge_world_matrix` (pano stations; perspective frames may use JSON `cameraToWorld` directly)

### 3.4 What does **not** solve the two-image case

| Idea | Why it fails |
|------|----------------|
| Feed both images into `merge_panorama_depth` | Requires **shared optical center**; these cameras are ~7.5 m apart |
| Invent arbitrary center `C` and rewrite ARKit origins as `t - C` | Coordinate shift only; cameras still have different translations; depths remain “from camera A/B”, not “from C” |
| Force both translations to zero so Poisson accepts them | Destroys parallax; geometrically false for near vs far |
| Voxel downsample alone | Averages colocated points; does not pull a wrongly scaled wall onto the other |
| Hardcoded FOV 90° | Sample FOV is ~56.6° H; wrong focal → worse metric geometry |

---

## 4. Target pipeline (desired feature)

```text
Input:
  image_i, K_i (or fov_x_i), cameraToWorld_i   for i in {A, B, ...}

1) fov_x_i = 2 * atan(W_i / (2 * fx_i)) * rad2deg
2) MoGe.infer(image_i, fov_x=fov_x_i) → depth_i / points_i / mask_i
3) Unproject to camera RDF points (MoGe points already RDF if used as-is)
4) P_world_i = P_cam_i @ R_i.T + t_i     # column-major cameraToWorld from JSON
5) Relative scale align on overlap:
     - Project world points of A into camera B (or vice versa)
     - On valid overlap: s = median( depth_B_moge / depth_A_projected )
     - Or plane-fit shared wall / ICP with scale
     - Apply s to one view’s points
6) Concat (+ optional voxel_downsample_points)
7) Write PLY + debug stats (baseline, scale s, wall residual)
```

### Acceptance criteria (for `single_face`)

- Both frame-11 images run with `fov_x ≈ 56.6°` derived from JSON intrinsics + 1440×1920 size.
- Combined cloud uses each frame’s `cameraToWorld` (OpenCV RDF → ARKit world).
- After scale align, shared wall / pillar region residual (median point-to-plane or depth disagreement) drops vs unaligned baseline.
- Output PLY viewable; optional colored-by-source (A vs B) to verify overlap.

### Non-goals (v1)

- Replacing the existing 12-face panorama Poisson path
- Joint multi-view neural fusion inside MoGe weights
- Assuming shared optical center / arbitrary-center Poisson shortcut
- Full TSDF meshing (optional later)

---

## 5. Implementation sketch (when building)

Suggested new entrypoint (name TBD), e.g. `standalone_moge/infer_posed_perspectives.py`:

1. CLI inputs:
   - `--image` repeatable, or a folder like `single_face`
   - `--poses` matching `*_poses_world_opencv_rdf.json`
   - `--frame-index` (default 11 for the sample)
   - `--model` MoGe-3 metric weights
   - `--align {none,median_depth_ratio,plane}` (default `median_depth_ratio`)
   - `--voxel-size`, `--out`
2. Reuse:
   - MoGe model load/infer from existing infer scripts
   - `transform_points_to_world`, `voxel_downsample_points`, `write_rgb_ply` from [moge_world_poses.py](standalone_moge/utils/moge_world_poses.py)
   - Axis notes from [moge_arkit_world_axis_handoff.md](moge_arkit_world_axis_handoff.md)
3. New logic to add:
   - Parse perspective pose JSON frames (not panorama `jobdata` heading/origin only)
   - `fov_x` from `intrinsicsImage` + image size
   - Overlap scale estimation between posed clouds
   - Debug dumps: per-view PLY before/after scale, `scales.json`

Closest prior art in handoff notes (MapAnything `scale_align` median prior/pred ratio) — same *idea*, not currently wired for free MoGe two-view perspectives.

---

## 6. Reference index

### Code — panorama / merge

| Symbol | Path |
|--------|------|
| `get_panorama_cameras` | [standalone_moge/utils/panorama.py](standalone_moge/utils/panorama.py) |
| `split_panorama_image` | same |
| `merge_panorama_depth` / GPU CG | same |
| Original `merge_panorama_depth` | [original_moge/MoGe/moge/utils/panorama.py](original_moge/MoGe/moge/utils/panorama.py) |
| Single pano CLI | [standalone_moge/infer_panorama.py](standalone_moge/infer_panorama.py) |
| Multi pano CLI | [standalone_moge/infer_multi_panorama.py](standalone_moge/infer_multi_panorama.py) |

### Code — poses / world / scale file

| Symbol | Path |
|--------|------|
| `transform_points_to_world` | [standalone_moge/utils/moge_world_poses.py](standalone_moge/utils/moge_world_poses.py) |
| `get_depth_scale` / `load_depth_scales` | same |
| `voxel_downsample_points` / `write_rgb_ply` | same |
| Pose unit tests | [test_moge_world_poses.py](test_moge_world_poses.py) |

### Docs / data

| Item | Path |
|------|------|
| ARKit ↔ OpenCV RDF handoff | [moge_arkit_world_axis_handoff.md](moge_arkit_world_axis_handoff.md) |
| Near/far sample | `C:\Users\A-511\Downloads\single_face` |

---

## 7. One-line summary

**Wanted:** posed multi-perspective MoGe fusion with **explicit overlap scale alignment** in ARKit world (near/far same wall).  
**Not wanted as the solution:** panorama Poisson `merge_panorama_depth`, hardcoded 90° faces, or “pick an arbitrary center and pretend cameras share it.”
