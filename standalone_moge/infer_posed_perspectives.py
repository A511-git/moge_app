#!/usr/bin/env python3
"""Fuse posed perspective photos into one world point cloud.

``cameras_auto.json`` supplies each photo's intrinsics, ``cameraToWorld``, and
``metricScale``. MoGe predicts the depth. That scale is applied to the points,
then ``cameraToWorld`` places them.

This is not the panorama path: cameras are translated, and ``merge_panorama_depth``
is not used.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import click
import cv2
import numpy as np
import torch

try:
    from standalone_moge.model import MoGeModel
    from standalone_moge.utils.download_weights import MODELS, download_single, normalize_model_name
    from standalone_moge.utils.moge_world_poses import transform_points_to_world, voxel_downsample_points, write_rgb_ply
    from standalone_moge.utils.posed_perspectives import (
        DEPTH_MAX_M,
        DEPTH_MIN_M,
        CamerasAuto,
        load_cameras_auto,
        match_cameras_to_images,
        median_valid_depth,
    )
except ImportError:
    from .model import MoGeModel
    from .utils.download_weights import MODELS, download_single, normalize_model_name
    from .utils.moge_world_poses import transform_points_to_world, voxel_downsample_points, write_rgb_ply
    from .utils.posed_perspectives import (
        DEPTH_MAX_M,
        DEPTH_MIN_M,
        CamerasAuto,
        load_cameras_auto,
        match_cameras_to_images,
        median_valid_depth,
    )


# Fixed per-view colors so a double wall is obvious in the debug PLY.
SOURCE_TINTS = np.array(
    [
        [220, 40, 40],
        [40, 110, 220],
        [40, 180, 70],
        [230, 150, 30],
        [160, 60, 200],
        [40, 190, 190],
    ],
    dtype=np.uint8,
)


def _load_rgb(image_path: Path, resize_to: Optional[int]) -> Tuple[np.ndarray, int, int]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Unreadable image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = rgb.shape[:2]
    if resize_to is not None and max(orig_height, orig_width) > int(resize_to):
        scale = float(resize_to) / float(max(orig_height, orig_width))
        target_width = max(1, int(round(orig_width * scale)))
        target_height = max(1, int(round(orig_height * scale)))
        rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return rgb, orig_width, orig_height


def _infer_maps(
    model: MoGeModel,
    rgb: np.ndarray,
    fov_x_deg: float,
    device: torch.device,
    resolution_level: int,
    num_tokens: Optional[int],
    refine_steps: int,
    use_fp16: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.tensor(rgb.astype(np.float32) / 255.0, device=device).permute(2, 0, 1)
    kwargs = {
        "fov_x": float(fov_x_deg),
        "resolution_level": resolution_level,
        "apply_mask": True,
        "refine_steps": refine_steps,
        "use_fp16": use_fp16,
    }
    if num_tokens is not None:
        kwargs["num_tokens"] = num_tokens
    with torch.inference_mode():
        output = model.infer(tensor, **kwargs)
    points = output["points"].detach().float().cpu().numpy()
    depth = output["depth"].detach().float().cpu().numpy()
    mask = output["mask"].detach().cpu().numpy()
    if points.ndim == 4:
        points = points[0]
    if depth.ndim == 3:
        depth = depth[0]
    if mask.ndim == 3:
        mask = mask[0]
    mask_bool = mask.astype(bool) if mask.dtype == np.bool_ else mask > 0.5
    return points, depth, mask_bool


def _valid_pixels(
    points: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    depth_max: float = DEPTH_MAX_M,
) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(depth)
    return mask & finite & (depth >= DEPTH_MIN_M) & (depth <= depth_max)


def _cloud_from_view(
    points_cam: np.ndarray,
    colors: np.ndarray,
    valid: np.ndarray,
    camera_to_world: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    scaled_cam = np.asarray(points_cam[valid], dtype=np.float64) * float(scale)
    world = transform_points_to_world(scaled_cam, camera_to_world)
    return world, np.asarray(colors[valid], dtype=np.uint8)


def _tint(count: int, view_index: int) -> np.ndarray:
    color = SOURCE_TINTS[view_index % len(SOURCE_TINTS)]
    return np.repeat(color.reshape(1, 3), count, axis=0)


def _json_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _write_clouds(
    out_dir: Path,
    unaligned_points: List[np.ndarray],
    unaligned_colors: List[np.ndarray],
    aligned_points: List[np.ndarray],
    aligned_colors: List[np.ndarray],
    rgb_points: List[np.ndarray],
    rgb_colors: List[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    def _concat(chunks: List[np.ndarray], width: int, dtype) -> np.ndarray:
        if not chunks:
            return np.zeros((0, width), dtype=dtype)
        return np.concatenate(chunks, axis=0)

    write_rgb_ply(
        out_dir / "unaligned_by_source.ply",
        _concat(unaligned_points, 3, np.float32),
        _concat(unaligned_colors, 3, np.uint8),
    )
    write_rgb_ply(
        out_dir / "aligned_by_source.ply",
        _concat(aligned_points, 3, np.float32),
        _concat(aligned_colors, 3, np.uint8),
    )
    rgb_pts = _concat(rgb_points, 3, np.float32)
    rgb_cols = _concat(rgb_colors, 3, np.uint8)
    if voxel_size > 0.0:
        rgb_pts, rgb_cols = voxel_downsample_points(rgb_pts, rgb_cols, voxel_size)
    write_rgb_ply(out_dir / "aligned_rgb.ply", rgb_pts, rgb_cols)
    return rgb_pts, rgb_cols


def fuse_cameras_auto(
    cameras_path: Path,
    image_dir: Path,
    output_path: Path,
    voxel_size: float,
    model: MoGeModel,
    device: torch.device,
    resolution_level: int,
    num_tokens: Optional[int],
    refine_steps: int,
    use_fp16: bool,
    resize_to: Optional[int],
    max_depth: Optional[float] = None,
) -> dict:
    """Run MoGe on each photo that has a cameras_auto entry, matched by filename."""
    document: CamerasAuto = load_cameras_auto(cameras_path)
    depth_max = document.max_depth_m if max_depth is None else float(max_depth)
    if depth_max <= 0:
        raise click.BadParameter("max-depth must be positive")
    views, skipped = match_cameras_to_images(document, image_dir)
    for name in skipped:
        print(f"[posed] skip {name}: no cameras_auto entry for that image, or the image file is missing")

    print(f"[posed] images {image_dir}")
    print(
        f"[posed] cameras_auto {document.world_frame}: {len(views)} views, "
        f"shared metricScale {document.metric_scale:.6f}, "
        f"max depth {depth_max:.1f} m"
    )
    print("[posed] MoGe predicts depth. JSON intrinsics set FOV, cameraToWorld places points, metricScale scales them.")

    unaligned_points: List[np.ndarray] = []
    unaligned_colors: List[np.ndarray] = []
    aligned_points: List[np.ndarray] = []
    aligned_colors: List[np.ndarray] = []
    rgb_points: List[np.ndarray] = []
    rgb_colors: List[np.ndarray] = []
    view_records = []

    for index, view in enumerate(views):
        rgb, orig_width, orig_height = _load_rgb(view.image_path, resize_to)
        photo = view.intrinsics_image
        if photo.width != orig_width or photo.height != orig_height:
            raise ValueError(
                f"{view.name}: intrinsicsImage is {photo.width}x{photo.height}, "
                f"image is {orig_width}x{orig_height}"
            )
        print(
            f"[posed] {view.name}: fov_x={photo.fov_x_deg:.2f} deg, "
            f"metricScale={view.metric_scale:.4f}, "
            f"origin={np.round(view.origin, 3).tolist()}, infer {rgb.shape[1]}x{rgb.shape[0]}"
        )
        points, depth, mask = _infer_maps(
            model,
            rgb,
            photo.fov_x_deg,
            device,
            resolution_level,
            num_tokens,
            refine_steps,
            use_fp16,
        )
        if depth.shape[:2] != rgb.shape[:2]:
            raise RuntimeError(
                f"{view.name}: MoGe depth {depth.shape} != image {rgb.shape[0]}x{rgb.shape[1]}"
            )
        scaled_depth = np.asarray(depth, dtype=np.float64) * float(view.metric_scale)
        valid = _valid_pixels(points, scaled_depth, mask, depth_max=depth_max)
        world_raw, _ = _cloud_from_view(points, rgb, valid, view.camera_to_world, 1.0)
        world_scaled, point_colors = _cloud_from_view(
            points, rgb, valid, view.camera_to_world, view.metric_scale
        )
        median_depth = median_valid_depth(
            scaled_depth,
            valid,
            depth_max=depth_max,
        )
        print(
            f"[posed] {view.name}: median scaled depth {median_depth:.3f} m, points {len(world_scaled)}"
        )
        if len(world_raw):
            unaligned_points.append(world_raw.astype(np.float32))
            unaligned_colors.append(_tint(len(world_raw), index))
        if len(world_scaled):
            aligned_points.append(world_scaled.astype(np.float32))
            aligned_colors.append(_tint(len(world_scaled), index))
            rgb_points.append(world_scaled.astype(np.float32))
            rgb_colors.append(point_colors)

        view_records.append(
            {
                "name": view.name,
                "scene": view.scene,
                "image": str(view.image_path),
                "fov_x_deg": float(photo.fov_x_deg),
                "fov_y_deg": float(photo.fov_y_deg),
                "fx": float(photo.fx),
                "fy": float(photo.fy),
                "cx": float(photo.cx),
                "cy": float(photo.cy),
                "width": int(photo.width),
                "height": int(photo.height),
                "origin": [float(v) for v in view.origin],
                "metric_scale": float(view.metric_scale),
                "median_depth_m": _json_float(median_depth),
                "scale": float(view.metric_scale),
            }
        )

    output_path.mkdir(parents=True, exist_ok=True)
    rgb_pts, _rgb_cols = _write_clouds(
        output_path,
        unaligned_points,
        unaligned_colors,
        aligned_points,
        aligned_colors,
        rgb_points,
        rgb_colors,
        voxel_size,
    )

    report = {
        "source": "cameras_auto",
        "cameras": str(document.path),
        "world": document.world_frame,
        "camera_axes": document.camera_axes,
        "units": document.units,
        "max_depth_m": float(depth_max),
        "metric_scale": float(document.metric_scale),
        "align": "metric_scale",
        "depth": "moge",
        "views": view_records,
    }
    if len(views) == 2:
        report["baseline_m"] = float(np.linalg.norm(views[0].origin - views[1].origin))

    ingest_path = output_path / "ingest.json"
    with ingest_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"[posed] Wrote {output_path / 'unaligned_by_source.ply'}")
    print(f"[posed] Wrote {output_path / 'aligned_by_source.ply'}")
    print(f"[posed] Wrote {output_path / 'aligned_rgb.ply'} ({len(rgb_pts)} points)")
    print(f"[posed] Wrote {ingest_path}")
    return report


@click.command(help="Fuse posed perspective images into one world point cloud.")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True, file_okay=False), help="Folder of the perspective images named in cameras_auto.json.")
@click.option("--cameras", "cameras_path", required=True, type=click.Path(exists=True, dir_okay=False), help="cameras_auto.json. MoGe predicts depth; JSON supplies K, poses, and metricScale.")
@click.option("--max-depth", "max_depth", type=float, default=None, help="Drop points farther than this many meters. Default: source.maxDepthMeters from cameras_auto.json.")
@click.option("--voxel-size", default=0.0, show_default=True, type=float, help="Voxel size in meters for aligned_rgb.ply. 0 keeps every point.")
@click.option("--output", "-o", "output_path", type=click.Path(), default="./output_posed", show_default=True)
@click.option("--model", "-m", "model_name", type=click.Choice(["vitl", "vitg", "moge-3-vitl", "moge-3-vitg"]), default="vitl", show_default=True)
@click.option("--pretrained", "pretrained_model_name_or_path", type=str, default=None, help="Checkpoint path or Hugging Face repo.")
@click.option("--checkpoint_dir", type=click.Path(), default=None, help="Directory for auto-downloaded MoGe-3 weights.")
@click.option("--device", "device_name", type=str, default="cuda", show_default=True)
@click.option("--fp16", "use_fp16", is_flag=True, help="Run inference in FP16 on CUDA.")
@click.option("--resize", "resize_to", type=int, default=None, help="Cap the long image side before inference. FOV is unchanged.")
@click.option("--resolution_level", type=int, default=9, show_default=True)
@click.option("--num_tokens", type=int, default=None)
@click.option("--refine_steps", type=int, default=3, show_default=True)
def main(
    input_path: str,
    cameras_path: str,
    max_depth: Optional[float],
    voxel_size: float,
    output_path: str,
    model_name: str,
    pretrained_model_name_or_path: Optional[str],
    checkpoint_dir: Optional[str],
    device_name: str,
    use_fp16: bool,
    resize_to: Optional[int],
    resolution_level: int,
    num_tokens: Optional[int],
    refine_steps: int,
) -> None:
    if not (0 <= resolution_level <= 9):
        raise click.BadParameter("resolution_level must be between 0 and 9")
    if voxel_size < 0:
        raise click.BadParameter("voxel-size must be >= 0")

    if device_name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")
    if use_fp16 and device.type != "cuda":
        use_fp16 = False
        print("[posed] FP16 needs CUDA. Running in FP32.")

    model_key = normalize_model_name(model_name)
    model_params = MODELS.get(model_key, (None, None, "unknown"))[2]
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
    elif os.environ.get("CHECKPOINT_DIR"):
        ckpt_dir = Path(os.environ["CHECKPOINT_DIR"])
    elif Path("/checkpoints").is_dir():
        ckpt_dir = Path("/checkpoints")
    else:
        ckpt_dir = Path("checkpoints")
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = download_single(model_key, ckpt_dir)

    print(f"[posed] Loading MoGe-3 ({model_key}, {model_params}) from '{pretrained_model_name_or_path}' on {device}")
    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
    fuse_cameras_auto(
        Path(cameras_path),
        Path(input_path),
        Path(output_path),
        voxel_size,
        model,
        device,
        resolution_level,
        num_tokens,
        refine_steps,
        use_fp16,
        resize_to,
        max_depth,
    )


if __name__ == "__main__":
    main()
