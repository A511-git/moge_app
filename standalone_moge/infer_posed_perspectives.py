#!/usr/bin/env python3
"""Fuse posed perspective photos in one ARKit world, with overlap scale alignment.

Each image is run through MoGe at the horizontal FOV implied by its JSON
intrinsics. Camera-space points are scaled about that camera, then placed with
the frame's column-major ``cameraToWorld`` (OpenCV RDF to ARKit world).

This is not the panorama path: cameras are translated, and ``merge_panorama_depth``
is not used.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
        PerspectiveFrame,
        ScaleResult,
        choose_reference_index,
        discover_perspective_pairs,
        estimate_scale,
        load_perspective_frame,
        median_valid_depth,
    )
except ImportError:
    from .model import MoGeModel
    from .utils.download_weights import MODELS, download_single, normalize_model_name
    from .utils.moge_world_poses import transform_points_to_world, voxel_downsample_points, write_rgb_ply
    from .utils.posed_perspectives import (
        DEPTH_MAX_M,
        DEPTH_MIN_M,
        PerspectiveFrame,
        ScaleResult,
        choose_reference_index,
        discover_perspective_pairs,
        estimate_scale,
        load_perspective_frame,
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


def _resolve_inputs(
    input_path: Optional[str],
    images: Sequence[str],
    poses: Sequence[str],
) -> List[Tuple[Path, Path]]:
    if input_path and (images or poses):
        raise click.UsageError("Pass either --input or --image/--poses, not both.")
    if input_path:
        return discover_perspective_pairs(input_path)
    if len(images) == 0 and len(poses) == 0:
        raise click.UsageError("Pass --input or at least one --image/--poses pair.")
    if len(images) != len(poses):
        raise click.UsageError(
            f"--image ({len(images)}) and --poses ({len(poses)}) must be paired one to one."
        )
    return [(Path(image), Path(pose)) for image, pose in zip(images, poses)]


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


def _valid_pixels(points: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(depth)
    return mask & finite & (depth >= DEPTH_MIN_M) & (depth <= DEPTH_MAX_M)


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


def _view_record(
    frame: PerspectiveFrame,
    median_depth: float,
    *,
    is_reference: bool,
    applied_scale: float,
    result: Optional[ScaleResult],
    baseline_to_reference_m: Optional[float],
) -> dict:
    record = {
        "name": frame.name,
        "image": str(frame.image_path) if frame.image_path is not None else None,
        "poses": str(frame.poses_path) if frame.poses_path is not None else None,
        "origin": [float(v) for v in frame.origin],
        "fov_x_deg": float(frame.intrinsics.fov_x_deg),
        "fov_y_deg": float(frame.intrinsics.fov_y_deg),
        "median_depth_m": _json_float(median_depth),
        "is_reference": is_reference,
        "scale": float(applied_scale),
        "estimated_scale": 1.0 if result is None else float(result.estimated_scale),
        "inlier_count": 0 if result is None else int(result.inlier_count),
        "residual_before_m": None if result is None else _json_float(result.residual_before_m),
        "residual_after_m": None if result is None else _json_float(result.residual_after_m),
        "naive_depth_ratio": None if result is None else _json_float(result.naive_depth_ratio),
        "sufficient_overlap": True if result is None else bool(result.sufficient_overlap),
        "baseline_to_reference_m": _json_float(baseline_to_reference_m),
    }
    return record


@click.command(help="Fuse posed perspective images into one ARKit-world point cloud.")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True, file_okay=False), default=None, help="Folder of images plus *_poses_world_opencv_rdf.json.")
@click.option("--image", "images", multiple=True, type=click.Path(exists=True, dir_okay=False), help="Perspective image. Repeat, paired in order with --poses.")
@click.option("--poses", "poses", multiple=True, type=click.Path(exists=True, dir_okay=False), help="Pose JSON for the matching --image.")
@click.option("--frame-index", default=11, show_default=True, type=int, help="Frame index inside each pose JSON.")
@click.option("--align", type=click.Choice(["none", "median_depth_ratio", "plane"]), default="median_depth_ratio", show_default=True)
@click.option("--reference", "reference_index", type=int, default=None, help="View index to lock as metric. Default: smallest median depth.")
@click.option("--stride", default=8, show_default=True, type=int, help="Pixel stride used while estimating overlap scale.")
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
    input_path: Optional[str],
    images: Tuple[str, ...],
    poses: Tuple[str, ...],
    frame_index: int,
    align: str,
    reference_index: Optional[int],
    stride: int,
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
    if stride < 1:
        raise click.BadParameter("stride must be >= 1")
    if voxel_size < 0:
        raise click.BadParameter("voxel-size must be >= 0")

    pairs = _resolve_inputs(input_path, images, poses)
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

    inferred: List[dict] = []
    for image_path, poses_path in pairs:
        rgb, orig_width, orig_height = _load_rgb(image_path, resize_to)
        frame = load_perspective_frame(
            poses_path,
            frame_index,
            orig_width,
            orig_height,
            image_path=image_path,
            name=image_path.stem,
        )
        intrinsics_infer = frame.intrinsics.scaled_to(rgb.shape[1], rgb.shape[0])
        print(
            f"[posed] {frame.name}: fov_x={frame.intrinsics.fov_x_deg:.2f} deg, "
            f"origin={np.round(frame.origin, 3).tolist()}, infer {rgb.shape[1]}x{rgb.shape[0]}"
        )
        points, depth, mask = _infer_maps(
            model,
            rgb,
            frame.intrinsics.fov_x_deg,
            device,
            resolution_level,
            num_tokens,
            refine_steps,
            use_fp16,
        )
        if depth.shape[:2] != (intrinsics_infer.height, intrinsics_infer.width):
            raise RuntimeError(
                f"{frame.name}: MoGe depth {depth.shape} != image {intrinsics_infer.height}x{intrinsics_infer.width}"
            )
        inferred.append(
            {
                "frame": frame,
                "intrinsics": intrinsics_infer,
                "points": points,
                "depth": depth,
                "mask": mask,
                "colors": rgb,
                "median_depth": median_valid_depth(depth, mask),
            }
        )

    reference = choose_reference_index(
        [item["median_depth"] for item in inferred],
        override=reference_index,
    )
    ref_item = inferred[reference]
    ref_frame: PerspectiveFrame = ref_item["frame"]
    print(
        f"[posed] Reference view {reference} ({ref_frame.name}), "
        f"median depth {ref_item['median_depth']:.3f} m, align={align}"
    )

    align_method = "median_depth_ratio" if align == "none" else align
    applied_scales = [1.0] * len(inferred)
    results: List[Optional[ScaleResult]] = [None] * len(inferred)
    for index, item in enumerate(inferred):
        if index == reference:
            continue
        src_frame: PerspectiveFrame = item["frame"]
        result = estimate_scale(
            ref_item["depth"],
            ref_item["mask"],
            ref_item["intrinsics"],
            ref_frame.camera_to_world,
            ref_frame.world_to_camera,
            item["depth"],
            item["mask"],
            item["intrinsics"],
            src_frame.camera_to_world,
            src_frame.world_to_camera,
            method=align_method,
            stride=stride,
        )
        results[index] = result
        if not result.sufficient_overlap:
            print(f"[posed] WARNING {src_frame.name}: overlap too small ({result.inlier_count} inliers). Scale left at 1.")
            applied_scales[index] = 1.0
        elif align == "none":
            applied_scales[index] = 1.0
            print(
                f"[posed] {src_frame.name}: align=none, estimated scale {result.estimated_scale:.4f} not applied. "
                f"depth residual {result.residual_before_m} m"
            )
        else:
            applied_scales[index] = result.scale
            print(
                f"[posed] {src_frame.name}: scale {result.scale:.4f} "
                f"({result.inlier_count} inliers), "
                f"depth residual {result.residual_before_m:.4f} m -> {result.residual_after_m:.4f} m"
            )

    unaligned_points: List[np.ndarray] = []
    unaligned_colors: List[np.ndarray] = []
    aligned_tint_points: List[np.ndarray] = []
    aligned_tint_colors: List[np.ndarray] = []
    aligned_rgb_points: List[np.ndarray] = []
    aligned_rgb_colors: List[np.ndarray] = []

    for index, item in enumerate(inferred):
        frame = item["frame"]
        valid = _valid_pixels(item["points"], item["depth"], item["mask"])
        world_unscaled, _ = _cloud_from_view(
            item["points"], item["colors"], valid, frame.camera_to_world, 1.0
        )
        world_scaled, rgb_colors = _cloud_from_view(
            item["points"], item["colors"], valid, frame.camera_to_world, applied_scales[index]
        )
        if len(world_unscaled):
            unaligned_points.append(world_unscaled)
            unaligned_colors.append(_tint(len(world_unscaled), index))
        if len(world_scaled):
            aligned_tint_points.append(world_scaled)
            aligned_tint_colors.append(_tint(len(world_scaled), index))
            aligned_rgb_points.append(world_scaled)
            aligned_rgb_colors.append(rgb_colors)

    def _concat(chunks: List[np.ndarray], width: int, dtype) -> np.ndarray:
        if not chunks:
            return np.zeros((0, width), dtype=dtype)
        return np.concatenate(chunks, axis=0)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rgb_ply(
        out_dir / "unaligned_by_source.ply",
        _concat(unaligned_points, 3, np.float32),
        _concat(unaligned_colors, 3, np.uint8),
    )
    write_rgb_ply(
        out_dir / "aligned_by_source.ply",
        _concat(aligned_tint_points, 3, np.float32),
        _concat(aligned_tint_colors, 3, np.uint8),
    )
    rgb_pts = _concat(aligned_rgb_points, 3, np.float32)
    rgb_cols = _concat(aligned_rgb_colors, 3, np.uint8)
    if voxel_size > 0.0:
        rgb_pts, rgb_cols = voxel_downsample_points(rgb_pts, rgb_cols, voxel_size)
    write_rgb_ply(out_dir / "aligned_rgb.ply", rgb_pts, rgb_cols)

    ref_origin = ref_frame.origin
    view_records = []
    for index, item in enumerate(inferred):
        frame = item["frame"]
        result = results[index]
        if align == "none" and result is not None:
            # Scale was not applied, so the after-residual matches the unscaled cloud.
            result = ScaleResult(
                scale=1.0,
                estimated_scale=result.estimated_scale,
                inlier_count=result.inlier_count,
                residual_before_m=result.residual_before_m,
                residual_after_m=result.residual_before_m,
                naive_depth_ratio=result.naive_depth_ratio,
                sufficient_overlap=result.sufficient_overlap,
            )
        baseline = None if index == reference else float(np.linalg.norm(frame.origin - ref_origin))
        view_records.append(
            _view_record(
                frame,
                item["median_depth"],
                is_reference=(index == reference),
                applied_scale=applied_scales[index],
                result=result,
                baseline_to_reference_m=baseline,
            )
        )

    report = {
        "align": align,
        "frame_index": int(frame_index),
        "reference_index": int(reference),
        "reference_name": ref_frame.name,
        "world": "arkit",
        "views": view_records,
    }
    if len(inferred) == 2:
        origins = [item["frame"].origin for item in inferred]
        report["baseline_m"] = float(np.linalg.norm(origins[0] - origins[1]))

    scales_path = out_dir / "scales.json"
    with scales_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"[posed] Wrote {out_dir / 'unaligned_by_source.ply'}")
    print(f"[posed] Wrote {out_dir / 'aligned_by_source.ply'}")
    print(f"[posed] Wrote {out_dir / 'aligned_rgb.ply'} ({len(rgb_pts)} points)")
    print(f"[posed] Wrote {scales_path}")


if __name__ == "__main__":
    main()
