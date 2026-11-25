"""
ARKit-to-MoGe World Coordinate Utilities & Multi-Panorama Point Cloud Merging.

Coordinate conventions:
- ARKit world: +Y up, XZ horizontal ground, heading 0° along -Z, clockwise from above, meters.
- MoGe panorama sphere / world: +Z up, -X front (heading 0°), +Y right, meters.
- Camera local frame (OpenCV RDF): +X right, +Y down, +Z forward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def get_scan_chain_index(record: dict) -> int:
    """Return scanChain.index as int for chronological sorting, or a large sentinel."""
    chain = record.get("scanChain")
    if isinstance(chain, dict) and "index" in chain:
        try:
            return int(chain["index"])
        except (ValueError, TypeError):
            pass
    return 999999


def extract_heading(record: dict, heading_source: str = "magneticHeading") -> Optional[float]:
    """
    Extract heading (degrees) from a jobdata record.

    Matches extract_heading() from moge_arkit_world_axis_handoff.md when
    heading_source='magneticHeading' (magneticHeading -> arKitData.originHeading ->
    headingAtPlace -> heading), or prioritizes arKitData.originHeading when
    heading_source='originHeading'.
    """
    arkit = record.get("arKitData") or {}
    if heading_source == "originHeading":
        order = [
            arkit.get("originHeading"),
            record.get("magneticHeading"),
            record.get("headingAtPlace"),
            arkit.get("headingAtPlace"),
            record.get("heading"),
            arkit.get("heading"),
        ]
    else:
        order = [
            record.get("magneticHeading"),
            arkit.get("originHeading"),
            record.get("headingAtPlace"),
            arkit.get("headingAtPlace"),
            record.get("heading"),
            arkit.get("heading"),
        ]

    for val in order:
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def extract_origin_xyz(record: dict) -> Optional[np.ndarray]:
    """Extract [x, y, z] translation in ARKit world meters from arKitData.origin."""
    arkit = record.get("arKitData") or {}
    origin = arkit.get("origin")
    if not origin:
        return None
    return np.array([origin["x"], origin["y"], origin["z"]], dtype=np.float64)


def load_jobdata_records(metadata_path: Union[str, Path]) -> List[dict]:
    """Load jobdata.json array and sort chronologically by scanChain.index."""
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of panorama records")
    return sorted(data, key=lambda r: (get_scan_chain_index(r), str(r.get("jobId", ""))))


def discover_panorama_inputs(
    input_dir: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[Tuple[dict, Path]], Path]:
    """
    Match panorama image files in input_dir with records in jobdata.json,
    preserving chronological scanChain.index order.
    """
    in_path = Path(input_dir)
    if metadata_path is not None:
        meta_file = Path(metadata_path)
    else:
        candidates = list(in_path.glob("jobdata.json")) or [
            p for p in sorted(in_path.glob("*.json")) if p.name != "moge_depth_scales.json"
        ]
        if not candidates:
            raise FileNotFoundError(f"No metadata JSON found in {in_path}")
        meta_file = candidates[0]

    records = load_jobdata_records(meta_file)
    image_files = [
        p for p in sorted(in_path.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    by_stem = {p.stem: p for p in image_files}

    matched: List[Tuple[dict, Path]] = []
    for record in records:
        job_id = record.get("jobId")
        if not job_id:
            continue
        if job_id in by_stem:
            matched.append((record, by_stem[job_id]))
            continue
        for p in image_files:
            if job_id in p.name or p.stem in job_id:
                matched.append((record, p))
                break

    return matched, meta_file


def arkit_to_moge_world_matrix() -> np.ndarray:
    """
    Proper 3D rotation (det = +1) mapping ARKit world (+Y up, -Z yaw0, +X yaw90)
    to MoGe world (+Z up, -X yaw0, +Y yaw90): [x_a, y_a, z_a] -> [z_a, x_a, y_a].
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return T


def compute_panorama_moge_pose(origin_xyz: np.ndarray, heading_deg: float) -> np.ndarray:
    """
    Compute 4x4 camera-to-world pose T_pano_to_moge_world for a panorama station.

    At heading 0°, panorama front (-X_pano) points along -X_moge_world and
    right (+Y_pano) points along +Y_moge_world. Increasing heading rotates
    clockwise from above around +Z_moge_world.
    """
    theta = np.deg2rad(float(heading_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    # R_z(-theta): clockwise rotation from above around +Z_moge
    R_yaw = np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_arkit_to_moge = arkit_to_moge_world_matrix()
    t_moge = T_arkit_to_moge[:3, :3] @ np.asarray(origin_xyz, dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_yaw
    T[:3, 3] = t_moge
    return T


def compute_face_moge_poses(
    T_pano_to_moge: np.ndarray,
    local_extrinsics: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Given T_pano_to_moge (4x4 c2w) and local world-to-camera extrinsics (4x4 w2c)
    from get_panorama_cameras(), compute per-face (c2w_list, w2c_list) in MoGe world.
    """
    c2w_list: List[np.ndarray] = []
    w2c_list: List[np.ndarray] = []
    for ext in local_extrinsics:
        T_face_to_pano = np.linalg.inv(np.asarray(ext, dtype=np.float64))
        T_face_to_moge = T_pano_to_moge @ T_face_to_pano
        T_moge_to_face = np.linalg.inv(T_face_to_moge)
        c2w_list.append(T_face_to_moge)
        w2c_list.append(T_moge_to_face)
    return c2w_list, w2c_list


def transform_points_to_world(points_cam: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    """Transform (..., 3) camera-space points into world space using 4x4 c2w pose."""
    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]
    pts = np.asarray(points_cam, dtype=np.float64)
    return (pts @ R.T + t).astype(np.float32)


def voxel_downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy grid voxel downsampling averaging positions and RGB colors per voxel."""
    if voxel_size <= 0.0 or len(points) == 0:
        return points.astype(np.float32), colors.astype(np.uint8)

    coords = np.floor(points.astype(np.float64) / float(voxel_size)).astype(np.int64)
    # Offset to positive integers for fast 1D lexsort/unique
    coords -= coords.min(axis=0, keepdims=True)
    _, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    num_voxels = len(counts)

    out_pts = np.zeros((num_voxels, 3), dtype=np.float64)
    out_cols = np.zeros((num_voxels, 3), dtype=np.float64)
    np.add.at(out_pts, inverse, points)
    np.add.at(out_cols, inverse, colors.astype(np.float64))

    out_pts /= counts[:, None]
    out_cols = np.clip(np.rint(out_cols / counts[:, None]), 0, 255).astype(np.uint8)
    return out_pts.astype(np.float32), out_cols


def write_rgb_ply(filepath: Union[str, Path], points: np.ndarray, colors: np.ndarray) -> None:
    """Write binary little-endian RGB point cloud PLY (float x,y,z + uchar r,g,b)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    structured = np.empty(n, dtype=dtype)
    if n > 0:
        pts_f32 = np.asarray(points, dtype=np.float32)
        cols_u8 = np.asarray(colors, dtype=np.uint8)
        structured["x"] = pts_f32[:, 0]
        structured["y"] = pts_f32[:, 1]
        structured["z"] = pts_f32[:, 2]
        structured["red"] = cols_u8[:, 0]
        structured["green"] = cols_u8[:, 1]
        structured["blue"] = cols_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        if n > 0:
            handle.write(structured.tobytes())


def load_depth_scales(scale_file_path: Union[str, Path]) -> Dict[str, dict]:
    """Load per-jobId depth scale dictionary (e.g. moge_depth_scales.json)."""
    path = Path(scale_file_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by jobId")
    return data


def get_depth_scale(
    scales: Optional[Dict[str, dict]],
    job_id: str,
    metric: str = "median",
) -> float:
    """Return depth scale multiplier for job_id using metric ('median' or 'mean'), defaulting to 1.0."""
    if not scales or job_id not in scales:
        return 1.0
    entry = scales[job_id]
    if isinstance(entry, (int, float)):
        return float(entry)
    val = entry.get(metric)
    if val is None:
        return 1.0
    return float(val)
