#!/usr/bin/env python3
"""
Multi-Panorama MoGe-3 Inference & Single-World Merging CLI.

Splits each equirectangular panorama into 12 MoGe perspective faces, runs MoGe-3
metric depth/point inference, computes each face's pose in a single MoGe world
coordinate frame from jobdata.json, and merges all panoramas into:
  - depth.npy (per-panorama or per-face)
  - pointcloud.ply (unified binary RGB point cloud in MoGe world axis)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import click
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from standalone_moge.custom_deps import utils3d_moge as utils3d
from standalone_moge.model import MoGeModel
from standalone_moge.utils.download_weights import MODELS, download_single, normalize_model_name
from standalone_moge.utils.moge_world_poses import (
    compute_face_moge_poses,
    compute_panorama_moge_pose,
    discover_panorama_inputs,
    extract_heading,
    extract_origin_xyz,
    get_depth_scale,
    load_depth_scales,
    transform_points_to_world,
    voxel_downsample_points,
    write_rgb_ply,
)
from standalone_moge.utils.panorama import (
    get_panorama_cameras,
    merge_panorama_depth,
    spherical_uv_to_directions,
    split_panorama_image,
)

FACES_PER_PANO = 12


def _infer_faces(
    model: MoGeModel,
    splitted_images: List[np.ndarray],
    splitted_intrinsics: List[np.ndarray],
    device: torch.device,
    batch_size: int,
    resolution_level: int,
    num_tokens: Optional[int],
    refine_steps: int,
    use_fp16: bool,
    face_owners: Optional[List[str]] = None,
):
    """Run MoGeModel.infer on perspective faces and return per-face outputs."""
    n_faces = len(splitted_images)
    n_batches = max(1, math.ceil(n_faces / batch_size)) if n_faces else 0
    print(
        f"[MoGe World][infer] Queued {n_faces} faces for normalization/forward "
        f"(batch_size={batch_size}, forwards={n_batches})"
    )

    depths, points_list, masks = [], [], []
    max_b = 0
    for batch_idx, i in enumerate(
        trange(0, n_faces, batch_size, desc="Inferring faces", leave=False)
    ):
        batch_slice = splitted_images[i : i + batch_size]
        image_tensor = (
            torch.tensor(np.stack(batch_slice) / 255.0, dtype=torch.float32, device=device)
            .permute(0, 3, 1, 2)
        )
        fov_x, _ = np.rad2deg(
            utils3d.np.intrinsics_to_fov(np.array(splitted_intrinsics[i : i + batch_size]))
        )
        fov_x_tensor = torch.tensor(fov_x, dtype=torch.float32, device=device)

        b = int(image_tensor.shape[0])
        max_b = max(max_b, b)
        end = i + b - 1
        owner_note = ""
        if face_owners is not None:
            owners = sorted(set(face_owners[i : i + b]))
            owner_note = f", panos={owners}"
        print(
            f"[MoGe World][infer] Forward {batch_idx + 1}/{n_batches}: "
            f"B={b} faces entering DINOv2 ImageNet normalize "
            f"(faces {i}-{end} of {n_faces}{owner_note})"
        )

        infer_kwargs = {
            "fov_x": fov_x_tensor,
            "resolution_level": resolution_level,
            "apply_mask": False,
            "refine_steps": refine_steps,
            "use_fp16": use_fp16,
        }
        if num_tokens is not None:
            infer_kwargs["num_tokens"] = num_tokens

        with torch.no_grad():
            output = model.infer(image_tensor, **infer_kwargs)

        depths.extend(list(output["depth"].cpu().numpy()))
        points_list.extend(list(output["points"].cpu().numpy()))
        masks.extend(list(output["mask"].cpu().numpy()))

    print(
        f"[MoGe World][infer] Done: {n_faces} faces in {n_batches} forwards, max B={max_b}"
    )
    return depths, points_list, masks


def _prepare_panorama(
    record: Dict[str, Any],
    image_path: Path,
    heading_source: str,
    resize_to: Optional[int],
    split_resolution: int,
    local_extrinsics: np.ndarray,
    local_intrinsics: List[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Load, resize, pose, and split one panorama into faces. Returns None if skipped."""
    job_id = record["jobId"]
    origin = extract_origin_xyz(record)
    heading = extract_heading(record, heading_source=heading_source)
    if origin is None or heading is None:
        print(f"[WARNING] Skipping {job_id}: missing origin or heading")
        return None

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[WARNING] Skipping unreadable image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image_rgb.shape[:2]
    height, width = orig_height, orig_width
    if resize_to is not None and (orig_height > resize_to or orig_width > resize_to):
        height = min(resize_to, int(resize_to * orig_height / orig_width))
        width = min(resize_to, int(resize_to * orig_width / orig_height))
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)

    T_pano_to_moge = compute_panorama_moge_pose(origin, heading)
    face_c2w_list, face_w2c_list = compute_face_moge_poses(T_pano_to_moge, local_extrinsics)
    splitted_images = split_panorama_image(
        image_rgb, local_extrinsics, local_intrinsics, split_resolution
    )

    return {
        "job_id": job_id,
        "heading": heading,
        "image_rgb": image_rgb,
        "height": height,
        "width": width,
        "T_pano_to_moge": T_pano_to_moge,
        "face_c2w_list": face_c2w_list,
        "face_w2c_list": face_w2c_list,
        "splitted_images": splitted_images,
    }


def _append_pose_summary(
    prepared: Dict[str, Any],
    *,
    depth_scale: float,
    depth_scale_metric: str,
    heading_source: str,
    poses_summary: List[Dict[str, Any]],
) -> None:
    poses_summary.append(
        {
            "jobId": prepared["job_id"],
            "headingDeg": float(prepared["heading"]),
            "headingSource": heading_source,
            "depthScale": float(depth_scale),
            "depthScaleMetric": depth_scale_metric,
            "T_pano_to_moge_world": prepared["T_pano_to_moge"].tolist(),
            "face_c2w_moge_world": [m.tolist() for m in prepared["face_c2w_list"]],
            "face_w2c_moge_world": [m.tolist() for m in prepared["face_w2c_list"]],
        }
    )


def _resize_pano_depth_mask(
    pano_depth,
    pano_mask,
    height: int,
    width: int,
):
    """Resize merged equirect depth/mask to (height, width); returns numpy arrays."""
    if isinstance(pano_depth, torch.Tensor):
        if pano_depth.shape[:2] != (height, width):
            pano_depth = F.interpolate(
                pano_depth.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            pano_mask = (
                F.interpolate(
                    pano_mask.float().unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="nearest",
                ).squeeze(0).squeeze(0)
                > 0.5
            )
        pano_depth_np = pano_depth.detach().cpu().numpy().astype(np.float32)
        pano_mask_np = pano_mask.detach().cpu().numpy().astype(bool)
    else:
        pano_depth_np = pano_depth.astype(np.float32)
        pano_mask_np = pano_mask.astype(bool)
        if pano_depth_np.shape[:2] != (height, width):
            pano_depth_np = cv2.resize(pano_depth_np, (width, height), interpolation=cv2.INTER_LINEAR)
            pano_mask_np = (
                cv2.resize(pano_mask_np.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            )
    return pano_depth_np, pano_mask_np


def _collect_scaled_face_world_points(
    prepared: Dict[str, Any],
    face_depths: List[np.ndarray],
    face_points: List[np.ndarray],
    face_masks: List[np.ndarray],
    *,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    save_face_depths_dir: Optional[Path] = None,
) -> tuple:
    """Return (world_pts_list, world_cols_list) for one pano's faces."""
    pts_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []
    for idx, (f_img, f_depth, f_pts, f_mask, f_c2w) in enumerate(
        zip(
            prepared["splitted_images"],
            face_depths,
            face_points,
            face_masks,
            prepared["face_c2w_list"],
        )
    ):
        scaled_depth = (f_depth * depth_scale).astype(np.float32)
        scaled_pts = (f_pts * depth_scale).astype(np.float32)
        if save_face_depths_dir is not None:
            np.save(str(save_face_depths_dir / f"face_{idx:02d}_depth.npy"), scaled_depth)
        valid = (
            np.asarray(f_mask, dtype=bool)
            & np.isfinite(scaled_depth)
            & (scaled_depth >= min_depth)
            & (scaled_depth <= max_depth)
        )
        if np.any(valid):
            pts_chunks.append(transform_points_to_world(scaled_pts[valid], f_c2w))
            col_chunks.append(f_img[valid])
    return pts_chunks, col_chunks


def _merge_prepared_pano(
    prepared: Dict[str, Any],
    face_depths: List[np.ndarray],
    face_points: List[np.ndarray],
    face_masks: List[np.ndarray],
    *,
    out_dir: Path,
    merge_mode: str,
    depth_scales: Optional[Dict[str, Any]],
    depth_scale_metric: str,
    heading_source: str,
    local_extrinsics: np.ndarray,
    local_intrinsics: List[np.ndarray],
    device: torch.device,
    min_depth: float,
    max_depth: float,
    all_world_pts: List[np.ndarray],
    all_world_cols: List[np.ndarray],
    poses_summary: List[Dict[str, Any]],
) -> None:
    """Apply per-pano merge_mode (faces | pano_then_world) and accumulate world points."""
    job_id = prepared["job_id"]
    image_rgb = prepared["image_rgb"]
    height = prepared["height"]
    width = prepared["width"]
    T_pano_to_moge = prepared["T_pano_to_moge"]

    pano_out_dir = out_dir / job_id
    pano_out_dir.mkdir(parents=True, exist_ok=True)
    depth_scale = get_depth_scale(depth_scales, job_id, metric=depth_scale_metric)

    if merge_mode == "faces":
        pts_chunks, col_chunks = _collect_scaled_face_world_points(
            prepared,
            face_depths,
            face_points,
            face_masks,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            save_face_depths_dir=pano_out_dir,
        )
        all_world_pts.extend(pts_chunks)
        all_world_cols.extend(col_chunks)
    elif merge_mode == "pano_then_world":
        face_distances = [np.linalg.norm(p, axis=-1).astype(np.float32) for p in face_points]
        merging_w, merging_h = min(1920, width), min(960, height)
        pano_depth, pano_mask = merge_panorama_depth(
            merging_w,
            merging_h,
            face_distances,
            face_masks,
            local_extrinsics,
            local_intrinsics,
            device=device,
            return_torch=(device.type == "cuda"),
        )
        pano_depth_np, pano_mask_np = _resize_pano_depth_mask(pano_depth, pano_mask, height, width)
        pano_depth_np = (pano_depth_np * depth_scale).astype(np.float32)
        np.save(str(pano_out_dir / "depth.npy"), pano_depth_np)

        uv = utils3d.np.uv_map(height, width)
        dirs = spherical_uv_to_directions(uv)
        pts_pano = pano_depth_np[:, :, None] * dirs
        valid = (
            pano_mask_np
            & np.isfinite(pano_depth_np)
            & (pano_depth_np >= min_depth)
            & (pano_depth_np <= max_depth)
        )
        if np.any(valid):
            all_world_pts.append(transform_points_to_world(pts_pano[valid], T_pano_to_moge))
            all_world_cols.append(image_rgb[valid])
    else:
        raise ValueError(f"Per-pano merge does not support merge_mode={merge_mode!r}")

    _append_pose_summary(
        prepared,
        depth_scale=depth_scale,
        depth_scale_metric=depth_scale_metric,
        heading_source=heading_source,
        poses_summary=poses_summary,
    )


def _merge_all_faces_world(
    prepared_panos: List[Dict[str, Any]],
    *,
    out_dir: Path,
    depth_scales: Optional[Dict[str, Any]],
    depth_scale_metric: str,
    heading_source: str,
    min_depth: float,
    max_depth: float,
    voxel_size: float,
    all_world_pts: List[np.ndarray],
    all_world_cols: List[np.ndarray],
    poses_summary: List[Dict[str, Any]],
) -> None:
    """Pose all N*12 faces into MoGe world and voxel-fuse overlaps into one cloud."""
    n_faces = sum(len(p["face_points"]) for p in prepared_panos)
    print(
        f"[MoGe World][merge] all_faces_world: fusing {n_faces} posed faces from "
        f"{len(prepared_panos)} panos in world space (voxel_size={voxel_size})"
    )
    for prepared in prepared_panos:
        job_id = prepared["job_id"]
        pano_out_dir = out_dir / job_id
        pano_out_dir.mkdir(parents=True, exist_ok=True)
        depth_scale = get_depth_scale(depth_scales, job_id, metric=depth_scale_metric)
        pts_chunks, col_chunks = _collect_scaled_face_world_points(
            prepared,
            prepared["face_depths"],
            prepared["face_points"],
            prepared["face_masks"],
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            save_face_depths_dir=pano_out_dir,
        )
        all_world_pts.extend(pts_chunks)
        all_world_cols.extend(col_chunks)
        _append_pose_summary(
            prepared,
            depth_scale=depth_scale,
            depth_scale_metric=depth_scale_metric,
            heading_source=heading_source,
            poses_summary=poses_summary,
        )
    if voxel_size <= 0.0:
        print(
            "[MoGe World][merge] WARNING: all_faces_world with voxel_size=0 skips overlap fusion; "
            "points are concatenated only. Use --voxel_size > 0 to average overlaps."
        )


def _merge_all_faces_sphere(
    prepared_panos: List[Dict[str, Any]],
    *,
    out_dir: Path,
    depth_scales: Optional[Dict[str, Any]],
    depth_scale_metric: str,
    heading_source: str,
    local_extrinsics: np.ndarray,
    local_intrinsics: List[np.ndarray],
    device: torch.device,
    min_depth: float,
    max_depth: float,
    all_world_pts: List[np.ndarray],
    all_world_cols: List[np.ndarray],
    poses_summary: List[Dict[str, Any]],
) -> None:
    """
    Force one equirect Poisson blend of all N*12 faces on a shared sphere.

    Geometrically assumes a shared panorama center (local icosahedron extrinsics
    repeated per pano). World unprojection uses the first panorama pose.
    """
    n_faces = sum(len(p["face_points"]) for p in prepared_panos)
    print(
        f"[MoGe World][merge] all_faces_sphere: Poisson-blending {n_faces} faces "
        f"from {len(prepared_panos)} panos onto one equirect sphere"
    )
    print(
        "[MoGe World][merge] WARNING: all_faces_sphere ignores per-pano translation; "
        "only valid when panos share roughly the same center. "
        f"World pose taken from first pano jobId={prepared_panos[0]['job_id']}."
    )

    face_distances: List[np.ndarray] = []
    face_masks: List[np.ndarray] = []
    face_extrinsics: List[np.ndarray] = []
    face_intrinsics: List[np.ndarray] = []

    for prepared in prepared_panos:
        job_id = prepared["job_id"]
        depth_scale = get_depth_scale(depth_scales, job_id, metric=depth_scale_metric)
        pano_out_dir = out_dir / job_id
        pano_out_dir.mkdir(parents=True, exist_ok=True)
        n = len(prepared["face_points"])
        for idx, (f_depth, f_pts, f_mask) in enumerate(
            zip(prepared["face_depths"], prepared["face_points"], prepared["face_masks"])
        ):
            scaled_depth = (f_depth * depth_scale).astype(np.float32)
            scaled_pts = (f_pts * depth_scale).astype(np.float32)
            np.save(str(pano_out_dir / f"face_{idx:02d}_depth.npy"), scaled_depth)
            face_distances.append(np.linalg.norm(scaled_pts, axis=-1).astype(np.float32))
            face_masks.append(np.asarray(f_mask, dtype=bool))
        # Repeat the same 12 local cameras for each pano (shared-sphere assumption).
        face_extrinsics.extend([local_extrinsics[i] for i in range(n)])
        face_intrinsics.extend(local_intrinsics[:n])
        _append_pose_summary(
            prepared,
            depth_scale=depth_scale,
            depth_scale_metric=depth_scale_metric,
            heading_source=heading_source,
            poses_summary=poses_summary,
        )

    anchor = prepared_panos[0]
    height, width = int(anchor["height"]), int(anchor["width"])
    merging_w, merging_h = min(1920, width), min(960, height)
    pano_depth, pano_mask = merge_panorama_depth(
        merging_w,
        merging_h,
        face_distances,
        face_masks,
        face_extrinsics,
        face_intrinsics,
        device=device,
        return_torch=(device.type == "cuda"),
    )
    pano_depth_np, pano_mask_np = _resize_pano_depth_mask(pano_depth, pano_mask, height, width)
    np.save(str(out_dir / "depth_all_faces_sphere.npy"), pano_depth_np)
    print(f"[MoGe World][merge] Wrote shared-sphere depth -> {out_dir / 'depth_all_faces_sphere.npy'}")

    uv = utils3d.np.uv_map(height, width)
    dirs = spherical_uv_to_directions(uv)
    pts_pano = pano_depth_np[:, :, None] * dirs
    valid = (
        pano_mask_np
        & np.isfinite(pano_depth_np)
        & (pano_depth_np >= min_depth)
        & (pano_depth_np <= max_depth)
    )
    if np.any(valid):
        all_world_pts.append(transform_points_to_world(pts_pano[valid], anchor["T_pano_to_moge"]))
        image_rgb = anchor["image_rgb"]
        if image_rgb.shape[:2] != (height, width):
            image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
        all_world_cols.append(image_rgb[valid])


@click.command(help="Multi-Panorama MoGe-3 Reconstruction in Single MoGe World Axis")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Directory of equirectangular panoramas.")
@click.option("--metadata", "-j", "metadata_path", type=click.Path(exists=True), default=None, help="Path to jobdata.json (defaults to jobdata.json inside --input).")
@click.option("--output", "-o", "output_path", type=click.Path(), default="./output_world", show_default=True, help="Output directory for depth.npy and pointcloud.ply.")
@click.option(
    "--merge_mode",
    type=click.Choice(["pano_then_world", "faces", "all_faces_world", "all_faces_sphere"]),
    default="pano_then_world",
    show_default=True,
    help=(
        "'pano_then_world': Poisson-merge 12 faces per pano then concat worlds; "
        "'faces': pose each face to world separately; "
        "'all_faces_world': pose all N*12 faces to world then voxel-fuse overlaps; "
        "'all_faces_sphere': Poisson-blend all N*12 faces on one shared equirect sphere (assumes shared center)."
    ),
)
@click.option("--infer_mode", type=click.Choice(["per_pano", "all_faces"]), default="per_pano", show_default=True, help="'per_pano': infer each pano's 12 faces separately; 'all_faces': queue all N*12 faces then infer (chunked by --batch_size).")
@click.option("--heading_source", type=click.Choice(["magneticHeading", "originHeading"]), default="magneticHeading", show_default=True, help="Heading field from jobdata.json (default matches handoff doc).")
@click.option("--model", "-m", "model_name", type=click.Choice(["vitl", "vitg", "moge-3-vitl", "moge-3-vitg"]), default="vitl", show_default=True)
@click.option("--pretrained", "pretrained_model_name_or_path", type=str, default=None)
@click.option("--checkpoint_dir", type=click.Path(), default=None)
@click.option("--device", "device_name", type=str, default="cuda", show_default=True)
@click.option("--fp16", "use_fp16", is_flag=True)
@click.option("--resolution_level", type=int, default=9, show_default=True)
@click.option("--num_tokens", type=int, default=None)
@click.option("--refine_steps", type=click.IntRange(min=0), default=3, show_default=True)
@click.option("--resize", "resize_to", type=int, default=None, help="Max dimension ceiling for input panoramas (default: None = keep original resolution).")
@click.option("--split_resolution", type=int, default=512, show_default=True)
@click.option("--batch_size", type=int, default=4, show_default=True)
@click.option("--min_depth", type=float, default=0.1, show_default=True, help="Minimum valid depth/distance in meters.")
@click.option("--max_depth", type=float, default=15.0, show_default=True, help="Maximum valid depth/distance in meters.")
@click.option("--voxel_size", type=float, default=0.02, show_default=True, help="Voxel size in meters for pointcloud.ply downsampling (0 = full dense).")
@click.option("--depth_scale_file", type=click.Path(exists=True), default=None, help="Optional JSON file with per-jobId depth scales (auto-uses moge_depth_scales.json in --input or alongside jobdata.json if present).")
@click.option("--depth_scale_metric", type=click.Choice(["median", "mean"]), default="median", show_default=True, help="Scale metric to use from --depth_scale_file.")
@click.option("--no_depth_scale", is_flag=True, default=False, help="Disable per-jobId depth scaling even if moge_depth_scales.json is present.")
@click.option("--no_flex", is_flag=True, default=False, help="Disable flex Gemma tuning and use direct inference (forces refine_steps=0 and num_tokens=None).")
def main(
    input_path: str,
    metadata_path: Optional[str],
    output_path: str,
    merge_mode: str,
    infer_mode: str,
    heading_source: str,
    model_name: str,
    pretrained_model_name_or_path: Optional[str],
    checkpoint_dir: Optional[str],
    device_name: str,
    use_fp16: bool,
    resolution_level: int,
    num_tokens: Optional[int],
    refine_steps: int,
    resize_to: Optional[int],
    split_resolution: int,
    batch_size: int,
    min_depth: float,
    max_depth: float,
    voxel_size: float,
    depth_scale_file: Optional[str],
    depth_scale_metric: str,
    no_depth_scale: bool,
    no_flex: bool,
) -> None:
    if no_flex:
        refine_steps = 0
        num_tokens = None
        print("[MoGe World] --no_flex: flex Gemma tuning disabled; using direct inference (refine_steps=0, num_tokens=None).")

    matched, meta_file = discover_panorama_inputs(input_path, metadata_path)
    if not matched:
        raise FileNotFoundError(f"No panoramas in {input_path} matched records in {meta_file}")

    depth_scales = None
    if no_depth_scale:
        print("[MoGe World] Depth scaling disabled (--no_depth_scale); using scale=1.0.")
    elif depth_scale_file is not None:
        depth_scales = load_depth_scales(depth_scale_file)
        print(f"[MoGe World] Using depth scales ({depth_scale_metric}) from: {depth_scale_file}")
    else:
        for candidate in (Path(input_path) / "moge_depth_scales.json", meta_file.parent / "moge_depth_scales.json"):
            if candidate.is_file():
                depth_scales = load_depth_scales(candidate)
                print(f"[MoGe World] Auto-detected depth scales ({depth_scale_metric}) from: {candidate}")
                break

    device = torch.device(device_name if torch.cuda.is_available() and "cuda" in device_name else "cpu")
    if use_fp16 and device.type != "cuda":
        use_fp16 = False

    model_key = normalize_model_name(model_name)
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

    print(
        f"[MoGe World] Loading {model_key.upper()} on {device} "
        f"(merge_mode={merge_mode}, infer_mode={infer_mode}, heading_source={heading_source})..."
    )
    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_extrinsics, local_intrinsics = get_panorama_cameras()

    all_world_pts: List[np.ndarray] = []
    all_world_cols: List[np.ndarray] = []
    poses_summary: List[Dict[str, Any]] = []

    print(
        f"[MoGe World][infer] Startup: infer_mode={infer_mode}, batch_size={batch_size}, "
        f"matched_panos={len(matched)}, expected_faces_if_all_ok={len(matched) * FACES_PER_PANO}"
    )

    prepared_panos: List[Dict[str, Any]] = []
    for record, image_path in tqdm(matched, desc="Preparing panoramas"):
        prepared = _prepare_panorama(
            record,
            image_path,
            heading_source=heading_source,
            resize_to=resize_to,
            split_resolution=split_resolution,
            local_extrinsics=local_extrinsics,
            local_intrinsics=local_intrinsics,
        )
        if prepared is not None:
            prepared_panos.append(prepared)

    if not prepared_panos:
        raise RuntimeError("No valid panoramas to process after prep.")

    n_panos = len(prepared_panos)
    n_faces_total = sum(len(p["splitted_images"]) for p in prepared_panos)
    print(
        f"[MoGe World][infer] Prepared {n_panos} panos -> {n_faces_total} faces "
        f"({FACES_PER_PANO} per pano)"
    )

    infer_kwargs = dict(
        model=model,
        device=device,
        batch_size=batch_size,
        resolution_level=resolution_level,
        num_tokens=num_tokens,
        refine_steps=refine_steps,
        use_fp16=use_fp16,
    )

    if infer_mode == "all_faces":
        all_images: List[np.ndarray] = []
        all_intrinsics: List[np.ndarray] = []
        face_owners: List[str] = []
        for prepared in prepared_panos:
            n = len(prepared["splitted_images"])
            all_images.extend(prepared["splitted_images"])
            all_intrinsics.extend(local_intrinsics[:n])
            face_owners.extend([prepared["job_id"]] * n)

        print(
            f"[MoGe World][infer] all_faces: single queue of {len(all_images)} faces "
            f"across {n_panos} panos"
        )
        all_depths, all_points, all_masks = _infer_faces(
            splitted_images=all_images,
            splitted_intrinsics=all_intrinsics,
            face_owners=face_owners,
            **infer_kwargs,
        )

        offset = 0
        for prepared in prepared_panos:
            n = len(prepared["splitted_images"])
            prepared["face_depths"] = all_depths[offset : offset + n]
            prepared["face_points"] = all_points[offset : offset + n]
            prepared["face_masks"] = all_masks[offset : offset + n]
            offset += n
    else:
        for prepared in tqdm(prepared_panos, desc="Inferring panoramas"):
            print(
                f"[MoGe World][infer] per_pano: queueing {len(prepared['splitted_images'])} faces "
                f"for jobId={prepared['job_id']}"
            )
            face_depths, face_points, face_masks = _infer_faces(
                splitted_images=prepared["splitted_images"],
                splitted_intrinsics=local_intrinsics,
                face_owners=[prepared["job_id"]] * len(prepared["splitted_images"]),
                **infer_kwargs,
            )
            prepared["face_depths"] = face_depths
            prepared["face_points"] = face_points
            prepared["face_masks"] = face_masks

    if merge_mode in ("pano_then_world", "faces"):
        for prepared in tqdm(prepared_panos, desc="Merging panoramas"):
            _merge_prepared_pano(
                prepared,
                prepared["face_depths"],
                prepared["face_points"],
                prepared["face_masks"],
                out_dir=out_dir,
                merge_mode=merge_mode,
                depth_scales=depth_scales,
                depth_scale_metric=depth_scale_metric,
                heading_source=heading_source,
                local_extrinsics=local_extrinsics,
                local_intrinsics=local_intrinsics,
                device=device,
                min_depth=min_depth,
                max_depth=max_depth,
                all_world_pts=all_world_pts,
                all_world_cols=all_world_cols,
                poses_summary=poses_summary,
            )
    elif merge_mode == "all_faces_world":
        _merge_all_faces_world(
            prepared_panos,
            out_dir=out_dir,
            depth_scales=depth_scales,
            depth_scale_metric=depth_scale_metric,
            heading_source=heading_source,
            min_depth=min_depth,
            max_depth=max_depth,
            voxel_size=voxel_size,
            all_world_pts=all_world_pts,
            all_world_cols=all_world_cols,
            poses_summary=poses_summary,
        )
    elif merge_mode == "all_faces_sphere":
        _merge_all_faces_sphere(
            prepared_panos,
            out_dir=out_dir,
            depth_scales=depth_scales,
            depth_scale_metric=depth_scale_metric,
            heading_source=heading_source,
            local_extrinsics=local_extrinsics,
            local_intrinsics=local_intrinsics,
            device=device,
            min_depth=min_depth,
            max_depth=max_depth,
            all_world_pts=all_world_pts,
            all_world_cols=all_world_cols,
            poses_summary=poses_summary,
        )
    else:
        raise ValueError(f"Unknown merge_mode={merge_mode!r}")

    if not all_world_pts:
        raise RuntimeError("No valid 3D points generated across any panorama.")

    merged_pts = np.concatenate(all_world_pts, axis=0)
    merged_cols = np.concatenate(all_world_cols, axis=0)
    if voxel_size > 0.0:
        merged_pts, merged_cols = voxel_downsample_points(merged_pts, merged_cols, voxel_size)

    ply_path = out_dir / "pointcloud.ply"
    write_rgb_ply(ply_path, merged_pts, merged_cols)

    with (out_dir / "poses_moge_world.json").open("w", encoding="utf-8") as handle:
        json.dump(poses_summary, handle, indent=2)

    print(f"[MoGe World] Wrote {len(merged_pts):,} points -> {ply_path}")


if __name__ == "__main__":
    main()
