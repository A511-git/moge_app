#!/usr/bin/env python3
"""Geometry tests for posed near/far scale alignment. No MoGe weights."""

import json
import tempfile
from pathlib import Path

import numpy as np

from standalone_moge.utils.moge_world_poses import transform_points_to_world
from standalone_moge.infer_posed_perspectives import fuse_cameras_auto
from standalone_moge.utils.posed_perspectives import (
    CameraIntrinsics,
    camera_points_to_world,
    choose_reference_index,
    discover_perspective_pairs,
    estimate_scale,
    load_cameras_auto,
    load_perspective_frame,
    matrix_from_column_major,
    naive_depth_ratio,
    per_point_scale,
    scale_world_about_camera,
    unproject_pixels,
)


SCALE_ERROR = 1.35
WIDTH = 160
HEIGHT = 120
FX = 200.0


def _intrinsics():
    return CameraIntrinsics(
        fx=FX,
        fy=FX,
        cx=(WIDTH - 1) / 2.0,
        cy=(HEIGHT - 1) / 2.0,
        width=WIDTH,
        height=HEIGHT,
    )


def _rigid_pose(origin, yaw_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )
    pose[:3, 3] = np.asarray(origin, dtype=np.float64)
    return pose


def _write_pose_json(path: Path, frame_index: int, camera_to_world: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> None:
    world_to_camera = np.linalg.inv(camera_to_world)
    document = {
        "convention": {"matrixOrder": "columnMajor", "cameraAxes": "opencvRDF"},
        "frames": [
            {
                "index": frame_index,
                "intrinsicsImage": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "cameraToWorld": camera_to_world.reshape(16, order="F").tolist(),
                "worldToCamera": world_to_camera.reshape(16, order="F").tolist(),
            }
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def render_plane_depth(camera_to_world, intrinsics, planes):
    """Z-buffer of planes. ``planes`` entries are (normal, rho, bounds_fn) with n·X = rho."""
    rotation = camera_to_world[:3, :3]
    origin = camera_to_world[:3, 3]
    us = np.arange(intrinsics.width, dtype=np.float64)
    vs = np.arange(intrinsics.height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    direction_cam = np.stack(
        [
            (uu - intrinsics.cx) / intrinsics.fx,
            (vv - intrinsics.cy) / intrinsics.fy,
            np.ones_like(uu),
        ],
        axis=-1,
    )
    direction_world = direction_cam @ rotation.T
    depth = np.full((intrinsics.height, intrinsics.width), np.inf, dtype=np.float64)
    for normal, rho, bounds in planes:
        denom = direction_world @ np.asarray(normal, dtype=np.float64)
        numer = float(rho) - float(np.dot(normal, origin))
        with np.errstate(divide="ignore", invalid="ignore"):
            hit_depth = numer / denom
        hit = np.isfinite(hit_depth) & (hit_depth > 0.05) & (np.abs(denom) > 1e-8)
        if bounds is not None:
            points = origin + hit_depth[..., None] * direction_world
            hit = hit & bounds(points)
        closer = hit & (hit_depth < depth)
        depth[closer] = hit_depth[closer]
    mask = np.isfinite(depth) & (depth < 1e8)
    return np.where(mask, depth, 0.0), mask


def _wall_plane():
    normal = np.array([0.0, 0.0, 1.0])

    def bounds(points: np.ndarray) -> np.ndarray:
        return (np.abs(points[..., 0]) <= 4.0) & (np.abs(points[..., 1]) <= 3.0)

    return normal, 5.0, bounds


def _median_plane_distance(points: np.ndarray, normal: np.ndarray, rho: float) -> float:
    return float(np.median(np.abs(points @ normal - rho)))


def test_column_major_parse_and_fov() -> None:
    matrix = np.array(
        [
            [0.2, 0.3, 0.4, 0.8183],
            [0.5, 0.1, -0.2, 0.2592],
            [0.1, 0.8, 0.3, 0.7011],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    flat = matrix.reshape(16, order="F")
    unpacked = matrix_from_column_major(flat.tolist())
    assert np.allclose(unpacked, matrix)
    assert not np.allclose(flat.reshape(4, 4), matrix)

    pose = _rigid_pose([0.8183, 0.2592, 0.7011], yaw_deg=14.79)
    fx = 1335.963
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "9oPhlqgwrM_poses_world_opencv_rdf.json"
        _write_pose_json(path, 11, pose, fx=fx, fy=fx, cx=714.871, cy=964.264)
        frame = load_perspective_frame(path, 11, image_width=1440, image_height=1920, name="far")

    assert frame.frame_index == 11
    assert np.allclose(frame.origin, [0.8183, 0.2592, 0.7011])
    assert np.allclose(frame.camera_to_world, pose)
    assert abs(frame.intrinsics.fov_x_deg - 56.644) < 0.01
    assert abs(frame.intrinsics.fov_y_deg - 71.401) < 0.05

    resized = frame.intrinsics.scaled_to(720, 960)
    assert abs(resized.fov_x_deg - frame.intrinsics.fov_x_deg) < 1e-9
    assert abs(resized.fov_y_deg - frame.intrinsics.fov_y_deg) < 1e-9
    assert abs(resized.fx - fx / 2.0) < 1e-6


def test_discover_pairs_by_capture_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "9oPhlqgwrM_11.jpg").write_bytes(b"")
        (folder / "sAHiABwkaK_11.jpg").write_bytes(b"")
        (folder / "9oPhlqgwrM_poses_world_opencv_rdf.json").write_text("{}", encoding="utf-8")
        (folder / "sAHiABwkaK_poses_world_opencv_rdf.json").write_text("{}", encoding="utf-8")
        pairs = discover_perspective_pairs(folder)

    assert [image.name for image, _ in pairs] == ["9oPhlqgwrM_11.jpg", "sAHiABwkaK_11.jpg"]
    assert pairs[0][1].name.startswith("9oPhlqgwrM_poses")
    assert pairs[1][1].name.startswith("sAHiABwkaK_poses")


def test_reference_is_the_near_view() -> None:
    assert choose_reference_index([8.85, 1.47]) == 1
    assert choose_reference_index([8.85, 1.47], override=0) == 0


def test_scale_about_camera_keeps_origin_and_matches_camera_scale() -> None:
    pose = _rigid_pose([0.82, 0.26, 0.70], yaw_deg=20.0)
    origin = pose[:3, 3]
    scaled_origin = scale_world_about_camera(origin.reshape(1, 3), origin, 1.35)
    assert np.allclose(scaled_origin, origin)

    zeros = np.zeros((1, 3), dtype=np.float64)
    assert np.allclose(camera_points_to_world(zeros, pose), origin)
    assert np.allclose(camera_points_to_world(zeros * 4.0, pose), origin)

    points_cam = np.array([[0.2, -0.4, 3.0], [1.0, 0.2, 8.0]], dtype=np.float64)
    scale = 1.0 / SCALE_ERROR
    about_camera = camera_points_to_world(points_cam * scale, pose)
    about_world = scale_world_about_camera(camera_points_to_world(points_cam, pose), origin, scale)
    assert np.allclose(about_camera, about_world)
    exported = transform_points_to_world(points_cam * scale, pose)
    assert np.allclose(about_camera, exported, atol=1e-5)


def test_wall_scale_recovers_and_plane_residual_collapses() -> None:
    intrinsics = _intrinsics()
    pose_ref = _rigid_pose([0.0, 0.0, 0.0])
    pose_src = _rigid_pose([0.0, 0.0, -7.0])
    wall = _wall_plane()
    depth_ref, mask_ref = render_plane_depth(pose_ref, intrinsics, [wall])
    depth_src, mask_src = render_plane_depth(pose_src, intrinsics, [wall])
    assert int(mask_ref.sum()) > 100
    assert int(mask_src.sum()) > 100
    depth_src = np.where(mask_src, depth_src * SCALE_ERROR, 0.0)

    world_to_ref = np.linalg.inv(pose_ref)
    world_to_src = np.linalg.inv(pose_src)
    result = estimate_scale(
        depth_ref,
        mask_ref,
        intrinsics,
        pose_ref,
        world_to_ref,
        depth_src,
        mask_src,
        intrinsics,
        pose_src,
        world_to_src,
        method="median_depth_ratio",
        stride=4,
    )
    assert result.sufficient_overlap
    assert abs(result.scale - (1.0 / SCALE_ERROR)) < 1e-3
    assert result.residual_after_m < result.residual_before_m
    assert result.residual_after_m < 1e-3
    assert abs(result.naive_depth_ratio - result.scale) > 0.05

    plane_result = estimate_scale(
        depth_ref,
        mask_ref,
        intrinsics,
        pose_ref,
        world_to_ref,
        depth_src,
        mask_src,
        intrinsics,
        pose_src,
        world_to_src,
        method="plane",
        stride=4,
    )
    assert abs(plane_result.scale - (1.0 / SCALE_ERROR)) < 1e-3

    normal, rho, _ = wall
    vs, us = np.mgrid[0:HEIGHT, 0:WIDTH]
    valid = mask_src
    observed = unproject_pixels(
        us[valid].astype(np.float64),
        vs[valid].astype(np.float64),
        depth_src[valid],
        intrinsics,
        pose_src,
    )
    corrected = scale_world_about_camera(observed, pose_src[:3, 3], result.scale)
    before = _median_plane_distance(observed, normal, rho)
    after = _median_plane_distance(corrected, normal, rho)
    assert before > 1.0
    assert after < 1e-3
    assert np.allclose(
        scale_world_about_camera(pose_src[:3, 3].reshape(1, 3), pose_src[:3, 3], result.scale),
        pose_src[:3, 3],
    )


def test_floor_and_wall_scale_is_constant_naive_ratio_is_not() -> None:
    """Corrected per-point scale is 1/1.35 on every surface. d_ref / Z_ref is not."""
    xs = np.linspace(-2.0, 2.0, 9)
    ys = np.linspace(-1.0, 1.0, 5)
    wall_x, wall_y = np.meshgrid(xs, ys)
    wall = np.column_stack(
        [wall_x.reshape(-1), wall_y.reshape(-1), np.full(wall_x.size, 5.0)]
    )
    floor_x = np.linspace(-2.0, 2.0, 5)
    floor_z = np.linspace(2.0, 8.0, 7)
    fx, fz = np.meshgrid(floor_x, floor_z)
    floor = np.column_stack([fx.reshape(-1), np.full(fx.size, 2.0), fz.reshape(-1)])
    points = np.vstack([wall, floor])

    origin_src = np.array([0.0, 0.0, -7.0])
    d_ref = points[:, 2]
    true_z_src = points[:, 2] - origin_src[2]
    z_rel = SCALE_ERROR * true_z_src
    z_ab = float(origin_src[2])
    scales = per_point_scale(d_ref, z_rel, z_ab)
    naive = naive_depth_ratio(d_ref, z_rel, z_ab)

    assert np.allclose(scales, 1.0 / SCALE_ERROR)
    assert float(np.std(scales)) < 1e-9
    assert float(np.std(naive)) > 1e-3
    assert abs(float(np.median(naive)) - (1.0 / SCALE_ERROR)) > 0.05

    observed = origin_src + SCALE_ERROR * (points - origin_src)
    corrected = scale_world_about_camera(observed, origin_src, float(np.median(scales)))
    wall_count = len(wall)
    normal = np.array([0.0, 0.0, 1.0])
    before = _median_plane_distance(observed[:wall_count], normal, 5.0)
    after = _median_plane_distance(corrected[:wall_count], normal, 5.0)
    assert before > 1.0
    assert after < 1e-6
    assert np.allclose(corrected, points)


def test_yawed_cameras_still_recover_scale() -> None:
    """A shared yaw makes camera +Z not world +Z, so a wrong matrix axis fails this."""
    intrinsics = _intrinsics()
    pose_ref = _rigid_pose([0.2, -0.1, 0.4], yaw_deg=20.0)
    forward = pose_ref[:3, :3] @ np.array([0.0, 0.0, 1.0])
    pose_src = pose_ref.copy()
    pose_src[:3, 3] = pose_ref[:3, 3] - 6.0 * forward
    wall = _wall_plane()
    depth_ref, mask_ref = render_plane_depth(pose_ref, intrinsics, [wall])
    depth_src, mask_src = render_plane_depth(pose_src, intrinsics, [wall])
    assert int(mask_ref.sum()) > 100
    assert int(mask_src.sum()) > 100
    depth_src = np.where(mask_src, depth_src * SCALE_ERROR, 0.0)

    result = estimate_scale(
        depth_ref,
        mask_ref,
        intrinsics,
        pose_ref,
        np.linalg.inv(pose_ref),
        depth_src,
        mask_src,
        intrinsics,
        pose_src,
        np.linalg.inv(pose_src),
        method="median_depth_ratio",
        stride=2,
    )
    assert result.sufficient_overlap
    assert abs(result.scale - (1.0 / SCALE_ERROR)) < 1e-3
    assert result.residual_after_m < 1e-2


def test_no_overlap_leaves_scale_at_one() -> None:
    intrinsics = _intrinsics()
    pose_ref = _rigid_pose([0.0, 0.0, 0.0])
    pose_src = _rigid_pose([0.0, 0.0, -7.0])
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    result = estimate_scale(
        depth,
        mask,
        intrinsics,
        pose_ref,
        np.linalg.inv(pose_ref),
        depth,
        mask,
        intrinsics,
        pose_src,
        np.linalg.inv(pose_src),
        stride=8,
    )
    assert result.scale == 1.0
    assert result.sufficient_overlap is False
    assert result.inlier_count == 0


def _k_block(fx, fy, cx, cy, width, height):
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
    }


def _auto_camera(name, scene, origin, metric_scale=5.0):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(origin, dtype=np.float64)
    flat = pose.reshape(16, order="F").tolist()
    inverse = np.linalg.inv(pose).reshape(16, order="F").tolist()
    return {
        "image": f"{name}.jpg",
        "scene": scene,
        "frameIndex": 11,
        "metricScale": metric_scale,
        "networkImage": {"width": 2, "height": 2},
        "originalImage": {"width": 4, "height": 4},
        "intrinsicsNetwork": _k_block(4.0, 4.0, 0.5, 0.5, 2, 2),
        "intrinsicsImage": _k_block(8.0, 8.0, 1.5, 1.5, 4, 4),
        "position": {"x": float(origin[0]), "y": float(origin[1]), "z": float(origin[2])},
        "cameraToWorld": flat,
        "worldToCamera": inverse,
        "depth": f"depths/{name}_depth.npy",
    }


def _write_cameras_auto(folder: Path, cameras) -> Path:
    document = {
        "comment": "ignored",
        "convention": {
            "frame": "view0World",
            "cameraAxes": "opencvRDF",
            "units": "meters",
            "matrixOrder": "columnMajor",
            "projection": "s [u, v, 1]^T = K [X, Y, Z]^T with s = Z",
        },
        "source": {
            "mode": "auto",
            "poseMap": "ignored",
            "input": "/ignored",
            "maxDepthMeters": 25.0,
        },
        "cameras": cameras,
    }
    path = folder / "cameras_auto.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_cameras_auto_loads_without_depth_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        cameras = [
            _auto_camera("near", "nearScene", [0.0, 0.0, 0.0], metric_scale=5.25),
            _auto_camera("far", "farScene", [0.0, 0.0, 3.0], metric_scale=5.25),
        ]
        for camera in cameras:
            camera.pop("depth")
            camera.pop("frameIndex")
            camera.pop("networkImage")
            camera.pop("intrinsicsNetwork")
            camera.pop("worldToCamera")
            camera.pop("position")
        path = _write_cameras_auto(folder, cameras)
        document = load_cameras_auto(path)

    assert document.world_frame == "view0World"
    assert document.metric_scale == 5.25
    assert document.max_depth_m == 25.0
    assert [view.scene for view in document.views] == ["nearScene", "farScene"]
    assert np.allclose(document.views[1].origin, [0.0, 0.0, 3.0])
    assert document.views[0].intrinsics_image.fx == 8.0
    assert abs(document.views[0].intrinsics_image.fov_x_deg - (2.0 * np.degrees(np.arctan(4.0 / 16.0)))) < 1e-9
    assert not hasattr(document.views[0], "depth_path")


def test_cameras_auto_rejects_bad_convention_and_split_scales() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        path = _write_cameras_auto(folder, [_auto_camera("only", "scene", [0.0, 0.0, 0.0])])
        document = json.loads(path.read_text(encoding="utf-8"))
        document["cameras"][0]["metricScale"] = 1.0
        document["cameras"].append(_auto_camera("other", "scene2", [1.0, 0.0, 0.0], metric_scale=2.0))
        path.write_text(json.dumps(document), encoding="utf-8")
        try:
            load_cameras_auto(path)
        except ValueError as exc:
            assert "metricScale" in str(exc)
        else:
            raise AssertionError("split metricScale should fail")

        document["cameras"] = document["cameras"][:1]
        document["cameras"][0]["metricScale"] = 5.0
        document["convention"]["cameraAxes"] = "arkit"
        path.write_text(json.dumps(document), encoding="utf-8")
        try:
            load_cameras_auto(path)
        except ValueError as exc:
            assert "opencvRDF" in str(exc)
        else:
            raise AssertionError("wrong camera axes should fail")


def test_fuse_cameras_auto_scales_moge_depth() -> None:
    import standalone_moge.infer_posed_perspectives as posed

    cv2 = __import__("cv2")

    def _fake_infer(model, rgb, fov_x_deg, device, resolution_level, num_tokens, refine_steps, use_fp16):
        height, width = rgb.shape[:2]
        depth = np.full((height, width), 2.0, dtype=np.float32)
        points = np.zeros((height, width, 3), dtype=np.float32)
        points[..., 2] = depth
        mask = np.ones((height, width), dtype=bool)
        return points, depth, mask

    original = posed._infer_maps
    posed._infer_maps = _fake_infer
    try:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = _write_cameras_auto(
                folder,
                [
                    _auto_camera("near", "nearScene", [0.0, 0.0, 0.0]),
                    _auto_camera("far", "farScene", [1.0, 0.0, 0.0]),
                ],
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            document["cameras"][1]["frameIndex"] = 0
            path.write_text(json.dumps(document), encoding="utf-8")
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            image[..., 0] = 255
            cv2.imwrite(str(folder / "near.jpg"), image)
            cv2.imwrite(str(folder / "far.jpg"), image)
            cv2.imwrite(str(folder / "stray.jpg"), image)
            out = folder / "out"
            report = fuse_cameras_auto(
                path,
                out,
                voxel_size=0.0,
                model=None,
                device=None,
                resolution_level=5,
                num_tokens=None,
                refine_steps=0,
                use_fp16=False,
                resize_to=None,
            )
            assert (out / "aligned_rgb.ply").is_file()
            assert (out / "unaligned_by_source.ply").is_file()
            assert (out / "ingest.json").is_file()
    finally:
        posed._infer_maps = original

    assert report["world"] == "view0World"
    assert report["depth"] == "moge"
    assert report["metric_scale"] == 5.0
    assert [view["name"] for view in report["views"]] == ["far", "near"]
    assert report["align"] == "metric_scale"
    assert report["views"][0]["scale"] == 5.0
    assert report["views"][0]["fx"] == 8.0
    assert abs(report["baseline_m"] - 1.0) < 1e-6
    assert abs(report["views"][0]["median_depth_m"] - 10.0) < 1e-6


def test_single_face_cameras_auto_ingest() -> None:
    path = Path(r"C:\Users\A-511\Downloads\single_face\cameras_auto.json")
    if not path.is_file():
        print("skip test_single_face_cameras_auto_ingest")
        return
    document = load_cameras_auto(path)
    assert document.world_frame == "view0World"
    assert document.camera_axes == "opencvRDF"
    assert document.units == "meters"
    assert document.max_depth_m == 25.0
    assert abs(document.metric_scale - 5.244701862335205) < 1e-9
    assert [view.image_name for view in document.views] == [
        "9oPhlqgwrM_11.jpg",
        "sAHiABwkaK_11.jpg",
    ]
    assert [view.scene for view in document.views] == ["9oPhlqgwrM", "sAHiABwkaK"]
    far, near = document.views
    assert np.allclose(
        far.origin,
        [-9.332343324786052e-05, -0.00086489028763026, -0.003577522002160549],
    )
    assert np.allclose(near.origin, [-0.25358846783638, -0.7677051424980164, 4.3514933586120605])
    assert far.intrinsics_image.width == 1440 and far.intrinsics_image.height == 1920
    assert abs(far.intrinsics_image.fx - 1400.5274168635976) < 1e-6
    assert abs(near.intrinsics_image.fx - 1396.816952833765) < 1e-6
    assert far.image_path.is_file()
    assert near.image_path.is_file()


if __name__ == "__main__":
    test_column_major_parse_and_fov()
    test_discover_pairs_by_capture_id()
    test_reference_is_the_near_view()
    test_scale_about_camera_keeps_origin_and_matches_camera_scale()
    test_wall_scale_recovers_and_plane_residual_collapses()
    test_floor_and_wall_scale_is_constant_naive_ratio_is_not()
    test_yawed_cameras_still_recover_scale()
    test_no_overlap_leaves_scale_at_one()
    test_cameras_auto_loads_without_depth_files()
    test_cameras_auto_rejects_bad_convention_and_split_scales()
    test_fuse_cameras_auto_scales_moge_depth()
    test_single_face_cameras_auto_ingest()
    print("ALL 12 TESTS PASSED")
