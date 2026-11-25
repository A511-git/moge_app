#!/usr/bin/env python3
"""Unit and geometry verification tests for ARKit-to-MoGe world pose & merging."""

import json
import tempfile
from pathlib import Path

import numpy as np

from standalone_moge.custom_deps import utils3d_moge as utils3d
from standalone_moge.utils.moge_world_poses import (
    arkit_to_moge_world_matrix,
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
    directions_to_spherical_uv,
    get_panorama_cameras,
    spherical_uv_to_directions,
)


def _assert_so3(R: np.ndarray, label: str) -> None:
    assert abs(float(np.linalg.det(R)) - 1.0) < 1e-6, f"{label}: det={np.linalg.det(R)}"
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6), f"{label}: not orthonormal"


def test_so3_rotations() -> None:
    T_a2m = arkit_to_moge_world_matrix()
    _assert_so3(T_a2m[:3, :3], "arkit_to_moge")

    local_ext, _ = get_panorama_cameras()
    assert len(local_ext) == 12

    for heading in [0.0, 45.0, 90.0, 180.0, 270.0, 359.5]:
        T_pano = compute_panorama_moge_pose(np.array([1.2, -0.5, 3.4]), heading)
        _assert_so3(T_pano[:3, :3], f"pano({heading})")
        c2w_list, w2c_list = compute_face_moge_poses(T_pano, local_ext)
        assert len(c2w_list) == 12 and len(w2c_list) == 12
        for idx, (c2w, w2c) in enumerate(zip(c2w_list, w2c_list)):
            _assert_so3(c2w[:3, :3], f"face_c2w({heading}, {idx})")
            _assert_so3(w2c[:3, :3], f"face_w2c({heading}, {idx})")
            assert np.allclose(c2w @ w2c, np.eye(4), atol=1e-6)


def test_jobdata_origin_pose_alignment() -> None:
    jobdata_path = Path(__file__).resolve().parent / "jobdata" / "jobdata.json"
    matched, _ = discover_panorama_inputs(jobdata_path.parent, jobdata_path)
    assert len(matched) == 8

    # Verify chronological scanChain.index ordering
    indices = [r["scanChain"]["index"] for r, _ in matched]
    assert indices == sorted(indices)

    T_a2m = arkit_to_moge_world_matrix()
    R_gl_to_cv = np.diag([1.0, -1.0, -1.0])
    # Front-face OpenCV RDF at heading=0 in ARKit world -> MoGe pano frame (-X front, +Y right, +Z up)
    R_front_cv_at_0 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    R_front_to_pano = np.linalg.inv(T_a2m[:3, :3] @ R_front_cv_at_0)

    for record, _ in matched:
        origin = extract_origin_xyz(record)
        heading_origin = extract_heading(record, heading_source="originHeading")
        assert origin is not None and heading_origin is not None

        T_pano = compute_panorama_moge_pose(origin, heading_origin)

        # Stored column-major OpenGL originPose -> OpenCV RDF -> MoGe world
        P_gl = np.array(record["arKitData"]["originPose"], dtype=np.float64).reshape(4, 4, order="F")
        R_arkit_cv = P_gl[:3, :3] @ R_gl_to_cv
        R_pano_from_pose = T_a2m[:3, :3] @ R_arkit_cv @ R_front_to_pano
        t_from_pose = T_a2m[:3, :3] @ P_gl[:3, 3]

        assert np.allclose(T_pano[:3, :3], R_pano_from_pose, atol=1e-5), record["jobId"]
        assert np.allclose(T_pano[:3, 3], t_from_pose, atol=1e-6), record["jobId"]


def test_face_to_world_ray_consistency() -> None:
    local_ext, local_intr = get_panorama_cameras()
    T_pano = compute_panorama_moge_pose(np.array([2.0, -1.0, 4.0]), 65.5)
    face_c2w, _ = compute_face_moge_poses(T_pano, local_ext)

    uv = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
    depth = np.ones((2,), dtype=np.float32)

    for idx in range(12):
        pts_cam = utils3d.np.unproject_cv(uv, depth, intrinsics=local_intr[idx])
        pts_pano = utils3d.np.unproject_cv(uv, depth, extrinsics=local_ext[idx], intrinsics=local_intr[idx])
        sph_uv = directions_to_spherical_uv(pts_pano)
        dirs_pano = spherical_uv_to_directions(sph_uv) * np.linalg.norm(pts_pano, axis=-1, keepdims=True)

        world_from_face = transform_points_to_world(pts_cam, face_c2w[idx])
        world_from_pano = transform_points_to_world(dirs_pano, T_pano)
        assert np.allclose(world_from_face, world_from_pano, atol=1e-4)


def test_synthetic_two_panorama_merge_and_ply() -> None:
    # Two panoramas at different ARKit positions & headings observing wall x_moge = 5.0
    T_p1 = compute_panorama_moge_pose(np.array([0.0, 0.0, 0.0]), 0.0)
    T_p2 = compute_panorama_moge_pose(np.array([1.5, -0.2, 2.0]), 40.0)
    local_ext, _ = get_panorama_cameras()
    f1_c2w, _ = compute_face_moge_poses(T_p1, local_ext)
    f2_c2w, _ = compute_face_moge_poses(T_p2, local_ext)

    # Create 100 world points on plane x_moge = 5.0, transform into face 0 camera coords, then back
    wall_world = np.column_stack(
        [
            np.full(100, 5.0, dtype=np.float64),
            np.linspace(-2.0, 2.0, 100),
            np.linspace(-1.0, 1.0, 100),
        ]
    )

    def world_to_cam(pts_w: np.ndarray, c2w: np.ndarray) -> np.ndarray:
        w2c = np.linalg.inv(c2w)
        return pts_w @ w2c[:3, :3].T + w2c[:3, 3]

    cam1 = world_to_cam(wall_world, f1_c2w[0])
    cam2 = world_to_cam(wall_world, f2_c2w[0])

    rec1 = transform_points_to_world(cam1, f1_c2w[0])
    rec2 = transform_points_to_world(cam2, f2_c2w[0])
    merged = np.concatenate([rec1, rec2], axis=0)
    cols = np.full((len(merged), 3), 200, dtype=np.uint8)

    assert np.allclose(merged[:, 0], 5.0, atol=1e-4)

    ds_pts, ds_cols = voxel_downsample_points(merged, cols, voxel_size=0.1)
    assert len(ds_pts) <= len(merged)
    assert np.allclose(ds_pts[:, 0], 5.0, atol=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = Path(tmpdir) / "pointcloud.ply"
        write_rgb_ply(ply_path, ds_pts, ds_cols)
        data = ply_path.read_bytes()
        assert data.startswith(b"ply\nformat binary_little_endian 1.0\n")
        assert f"element vertex {len(ds_pts)}".encode("ascii") in data


def test_depth_scales_loading() -> None:
    scales_path = Path(__file__).resolve().parent / "jobdata" / "moge_depth_scales.json"
    scales = load_depth_scales(scales_path)
    assert abs(get_depth_scale(scales, "9oPhlqgwrM", metric="median") - 1.8772) < 1e-6
    assert abs(get_depth_scale(scales, "9oPhlqgwrM", metric="mean") - 2.5576) < 1e-6
    assert get_depth_scale(scales, "nonexistent", metric="median") == 1.0
    assert get_depth_scale(None, "9oPhlqgwrM", metric="median") == 1.0


if __name__ == "__main__":
    test_so3_rotations()
    test_jobdata_origin_pose_alignment()
    test_face_to_world_ray_consistency()
    test_synthetic_two_panorama_merge_and_ply()
    test_depth_scales_loading()
    print("ALL 5 TESTS PASSED")
