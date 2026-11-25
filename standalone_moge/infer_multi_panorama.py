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
import os
import sys
from pathlib import Path
from typing import List, Optional

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
):
    """Run MoGeModel.infer on the 12 perspective faces and return per-face outputs."""
    depths, points_list, masks = [], [], []
    for i in trange(0, len(splitted_images), batch_size, desc="Inferring faces", leave=False):
        batch_slice = splitted_images[i : i + batch_size]
        image_tensor = (
            torch.tensor(np.stack(batch_slice) / 255.0, dtype=torch.float32, device=device)
            .permute(0, 3, 1, 2)
        )
        fov_x, _ = np.rad2deg(
            utils3d.np.intrinsics_to_fov(np.array(splitted_intrinsics[i : i + batch_size]))
        )
        fov_x_tensor = torch.tensor(fov_x, dtype=torch.float32, device=device)

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

    return depths, points_list, masks


@click.command(help="Multi-Panorama MoGe-3 Reconstruction in Single MoGe World Axis")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), required=True, help="Directory of equirectangular panoramas.")
@click.option("--metadata", "-j", "metadata_path", type=click.Path(exists=True), default=None, help="Path to jobdata.json (defaults to jobdata.json inside --input).")
@click.option("--output", "-o", "output_path", type=click.Path(), default="./output_world", show_default=True, help="Output directory for depth.npy and pointcloud.ply.")
@click.option("--merge_mode", type=click.Choice(["pano_then_world", "faces"]), default="pano_then_world", show_default=True, help="'pano_then_world': merge 12 faces per pano first via merge_panorama_depth then merge worlds; 'faces': merge all N*12 posed faces directly.")
@click.option("--heading_source", type=click.Choice(["magneticHeading", "originHeading"]), default="magneticHeading", show_default=True, help="Heading field from jobdata.json (default matches handoff doc).")
@click.option("--model", "-m", "model_name", type=click.Choice(["vitl", "vitg", "moge-3-vitl", "moge-3-vitg"]), default="vitl", show_default=True)
@click.option("--pretrained", "pretrained_model_name_or_path", type=str, default=None)
@click.option("--checkpoint_dir", type=click.Path(), default=None)
@click.option("--device", "device_name", type=str, default="cuda", show_default=True)
@click.option("--fp16", "use_fp16", is_flag=True)
@click.option("--resolution_level", type=int, default=9, show_default=True)
@click.option("--num_tokens", type=int, default=None)
@click.option("--refine_steps", type=click.IntRange(min=0), default=3, show_default=True)
@click.option("--split_resolution", type=int, default=512, show_default=True)
@click.option("--batch_size", type=int, default=4, show_default=True)
@click.option("--min_depth", type=float, default=0.1, show_default=True, help="Minimum valid depth/distance in meters.")
@click.option("--max_depth", type=float, default=15.0, show_default=True, help="Maximum valid depth/distance in meters.")
@click.option("--voxel_size", type=float, default=0.02, show_default=True, help="Voxel size in meters for pointcloud.ply downsampling (0 = full dense).")
@click.option("--depth_scale_file", type=click.Path(exists=True), default=None, help="Optional JSON file with per-jobId depth scales (auto-uses moge_depth_scales.json in --input or alongside jobdata.json if present).")
@click.option("--depth_scale_metric", type=click.Choice(["median", "mean"]), default="median", show_default=True, help="Scale metric to use from --depth_scale_file.")
@click.option("--no_depth_scale", is_flag=True, default=False, help="Disable per-jobId depth scaling even if moge_depth_scales.json is present.")
def main(
    input_path: str,
    metadata_path: Optional[str],
    output_path: str,
    merge_mode: str,
    heading_source: str,
    model_name: str,
    pretrained_model_name_or_path: Optional[str],
    checkpoint_dir: Optional[str],
    device_name: str,
    use_fp16: bool,
    resolution_level: int,
    num_tokens: Optional[int],
    refine_steps: int,
    split_resolution: int,
    batch_size: int,
    min_depth: float,
    max_depth: float,
    voxel_size: float,
    depth_scale_file: Optional[str],
    depth_scale_metric: str,
    no_depth_scale: bool,
) -> None:
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

    print(f"[MoGe World] Loading {model_key.upper()} on {device} (merge_mode={merge_mode}, heading_source={heading_source})...")
    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_extrinsics, local_intrinsics = get_panorama_cameras()

    all_world_pts: List[np.ndarray] = []
    all_world_cols: List[np.ndarray] = []
    poses_summary = []

    for record, image_path in tqdm(matched, desc="Panoramas"):
        job_id = record["jobId"]
        origin = extract_origin_xyz(record)
        heading = extract_heading(record, heading_source=heading_source)
        if origin is None or heading is None:
            print(f"[WARNING] Skipping {job_id}: missing origin or heading")
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARNING] Skipping unreadable image: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        T_pano_to_moge = compute_panorama_moge_pose(origin, heading)
        face_c2w_list, face_w2c_list = compute_face_moge_poses(T_pano_to_moge, local_extrinsics)

        splitted_images = split_panorama_image(image_rgb, local_extrinsics, local_intrinsics, split_resolution)
        face_depths, face_points, face_masks = _infer_faces(
            model=model,
            splitted_images=splitted_images,
            splitted_intrinsics=local_intrinsics,
            device=device,
            batch_size=batch_size,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            refine_steps=refine_steps,
            use_fp16=use_fp16,
        )

        pano_out_dir = out_dir / job_id
        pano_out_dir.mkdir(parents=True, exist_ok=True)
        depth_scale = get_depth_scale(depth_scales, job_id, metric=depth_scale_metric)

        if merge_mode == "faces":
            for idx, (f_img, f_depth, f_pts, f_mask, f_c2w) in enumerate(
                zip(splitted_images, face_depths, face_points, face_masks, face_c2w_list)
            ):
                scaled_depth = (f_depth * depth_scale).astype(np.float32)
                scaled_pts = (f_pts * depth_scale).astype(np.float32)
                np.save(str(pano_out_dir / f"face_{idx:02d}_depth.npy"), scaled_depth)
                valid = (
                    np.asarray(f_mask, dtype=bool)
                    & np.isfinite(scaled_depth)
                    & (scaled_depth >= min_depth)
                    & (scaled_depth <= max_depth)
                )
                if np.any(valid):
                    pts_world = transform_points_to_world(scaled_pts[valid], f_c2w)
                    all_world_pts.append(pts_world)
                    all_world_cols.append(f_img[valid])
        else:
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
                    pano_mask_np = cv2.resize(pano_mask_np.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0

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
                pts_world = transform_points_to_world(pts_pano[valid], T_pano_to_moge)
                all_world_pts.append(pts_world)
                all_world_cols.append(image_rgb[valid])

        poses_summary.append(
            {
                "jobId": job_id,
                "headingDeg": float(heading),
                "headingSource": heading_source,
                "depthScale": float(depth_scale),
                "depthScaleMetric": depth_scale_metric,
                "T_pano_to_moge_world": T_pano_to_moge.tolist(),
                "face_c2w_moge_world": [m.tolist() for m in face_c2w_list],
                "face_w2c_moge_world": [m.tolist() for m in face_w2c_list],
            }
        )

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
