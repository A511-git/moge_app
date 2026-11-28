"""Posed perspective views: ARKit intrinsics, OpenCV RDF poses, overlap scale.

MoGe stays a single-image model. This module only parses per-frame
``cameraToWorld`` poses (column-major, OpenCV RDF camera to ARKit world) and
solves one relative scale so a second view lands on the reference surface.

Scale is applied about the source camera, which keeps the ARKit origin fixed::

    P_world = (s * P_cam) @ R.T + t

With a translated baseline, depth-from-the-reference of that scaled cloud is
affine, not a constant ratio::

    Z_ref(s * (P - t_src) + t_src) = s * z_rel + z_AB

``z_rel`` is the reference-camera z of ``(P_src - t_src)`` and ``z_AB`` is the
reference-camera z of ``(t_src - t_ref)``. The scale that matches a reference
depth ``d_ref`` is::

    s = median( (d_ref - z_AB) / z_rel )

Correspondences are made by projecting the reference cloud into the source
camera. An unknown source scale moves points along source rays, so those
pixels stay put. Projecting the already-scaled source cloud into the reference
would sample the wrong reference depth wherever depth varies across the image.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 30.0
SCALE_MIN = 0.05
SCALE_MAX = 20.0
Z_REL_EPS = 1e-3
MIN_INLIERS = 30


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, plus the image size they were measured on."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Scale K with the image. Horizontal and vertical FOV stay the same."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid target size {width}x{height}")
        sx = float(width) / float(self.width)
        sy = float(height) / float(self.height)
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=int(width),
            height=int(height),
        )

    @property
    def fov_x_deg(self) -> float:
        return fov_x_degrees(self.width, self.fx)

    @property
    def fov_y_deg(self) -> float:
        return fov_y_degrees(self.height, self.fy)


@dataclass
class PerspectiveFrame:
    """One ARKit frame: JSON pose plus the intrinsics of the RGB image."""

    name: str
    frame_index: int
    intrinsics: CameraIntrinsics
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    image_path: Optional[Path] = None
    poses_path: Optional[Path] = None

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.camera_to_world[:3, 3], dtype=np.float64).copy()


@dataclass
class ScaleResult:
    """Robust scale of a source cloud about its camera, matched to a reference."""

    scale: float
    estimated_scale: float
    inlier_count: int
    residual_before_m: Optional[float]
    residual_after_m: Optional[float]
    naive_depth_ratio: Optional[float]
    sufficient_overlap: bool


@dataclass
class CamerasAutoView:
    """One camera from ``cameras_auto.json``.

    MoGe predicts the depth. This record supplies the photo intrinsics, the
    OpenCV RDF extrinsics, and the shared ``metric_scale`` applied to those
    camera-space points. Any ``depth`` path in the JSON is ignored.
    """

    image_name: str
    scene: Optional[str]
    metric_scale: float
    intrinsics_image: CameraIntrinsics
    camera_to_world: np.ndarray
    image_path: Path
    source_path: Path

    @property
    def name(self) -> str:
        return Path(self.image_name).stem

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.camera_to_world[:3, 3], dtype=np.float64).copy()


@dataclass
class CamerasAuto:
    """Document-level convention plus every camera in ``cameras_auto.json``."""

    world_frame: str
    camera_axes: str
    units: str
    matrix_order: str
    max_depth_m: float
    metric_scale: float
    views: List[CamerasAutoView]
    path: Path


def fov_x_degrees(width: float, fx: float) -> float:
    """Horizontal FOV in degrees from image width and ``fx``."""
    if fx <= 0 or width <= 0:
        raise ValueError(f"fx and width must be positive, got fx={fx}, width={width}")
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * float(fx))))


def fov_y_degrees(height: float, fy: float) -> float:
    """Vertical FOV in degrees from image height and ``fy``."""
    if fy <= 0 or height <= 0:
        raise ValueError(f"fy and height must be positive, got fy={fy}, height={height}")
    return math.degrees(2.0 * math.atan(float(height) / (2.0 * float(fy))))


def matrix_from_column_major(values: Sequence[float]) -> np.ndarray:
    """Unpack a length-16 column-major 4x4 matrix (JSON ``cameraToWorld``)."""
    flat = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if flat.size != 16:
        raise ValueError(f"Expected 16 matrix values, got {flat.size}")
    return flat.reshape(4, 4, order="F")


def _capture_id(path: Path) -> str:
    """``9oPhlqgwrM_11.jpg`` and ``9oPhlqgwrM_poses_....json`` share this id."""
    return path.stem.split("_", 1)[0]


def discover_perspective_pairs(input_dir: Union[str, Path]) -> List[Tuple[Path, Path]]:
    """Pair each image in a folder with the pose JSON that shares its capture id."""
    folder = Path(input_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    images = [
        path for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    json_paths = [
        path for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() == ".json"
    ]
    pose_paths = [path for path in json_paths if "pose" in path.stem.lower()] or json_paths
    by_id: Dict[str, Path] = {}
    for pose_path in pose_paths:
        by_id[_capture_id(pose_path)] = pose_path

    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for image_path in images:
        pose_path = by_id.get(_capture_id(image_path))
        if pose_path is None:
            missing.append(image_path.name)
            continue
        pairs.append((image_path, pose_path))

    if missing:
        raise FileNotFoundError(
            f"No pose JSON for {missing} in {folder}. "
            "Expected a file such as '<captureId>_poses_world_opencv_rdf.json'."
        )
    if not pairs:
        raise FileNotFoundError(f"No perspective images found in {folder}")
    return pairs


def load_perspective_frame(
    poses_path: Union[str, Path],
    frame_index: int,
    image_width: int,
    image_height: int,
    image_path: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
) -> PerspectiveFrame:
    """Load one ``frames[index]`` entry and attach the RGB image size to K."""
    path = Path(poses_path)
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{path} has no 'frames' array")

    matches = [frame for frame in frames if int(frame.get("index", -1)) == int(frame_index)]
    if len(matches) != 1:
        raise ValueError(
            f"{path} has {len(matches)} frames with index {frame_index}, expected 1"
        )
    frame = matches[0]
    intrinsics_json = frame.get("intrinsicsImage") or {}
    try:
        fx = float(intrinsics_json["fx"])
        fy = float(intrinsics_json["fy"])
        cx = float(intrinsics_json["cx"])
        cy = float(intrinsics_json["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} frame {frame_index} is missing intrinsicsImage fx/fy/cx/cy") from exc

    camera_to_world = matrix_from_column_major(frame["cameraToWorld"])
    world_to_camera = matrix_from_column_major(frame["worldToCamera"])
    if camera_to_world.shape != (4, 4) or world_to_camera.shape != (4, 4):
        raise ValueError(f"{path} frame {frame_index} poses are not 4x4")
    inverse_error = np.linalg.norm(camera_to_world @ world_to_camera - np.eye(4))
    if inverse_error > 1e-3:
        raise ValueError(
            f"{path} frame {frame_index} cameraToWorld and worldToCamera disagree "
            f"(error {inverse_error:.3e})"
        )

    image = Path(image_path) if image_path is not None else None
    frame_name = name or (image.stem if image is not None else f"{path.stem}:{frame_index}")
    return PerspectiveFrame(
        name=frame_name,
        frame_index=int(frame_index),
        intrinsics=CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=int(image_width),
            height=int(image_height),
        ),
        camera_to_world=camera_to_world,
        world_to_camera=world_to_camera,
        image_path=image,
        poses_path=path,
    )


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _parse_intrinsics_block(block: object, label: str) -> CameraIntrinsics:
    """Read ``fx/fy/cx/cy`` and check the nested ``K`` repeats them."""
    if not isinstance(block, dict):
        raise ValueError(f"{label} must be an object")
    try:
        fx = _finite_float(block["fx"], f"{label}.fx")
        fy = _finite_float(block["fy"], f"{label}.fy")
        cx = _finite_float(block["cx"], f"{label}.cx")
        cy = _finite_float(block["cy"], f"{label}.cy")
        width = int(block["width"])
        height = int(block["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} is missing fx/fy/cx/cy/width/height") from exc
    if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"{label} focal and size must be positive")
    try:
        matrix = np.asarray(block["K"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}.K is missing") from exc
    expected = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.allclose(matrix, expected, rtol=0.0, atol=1e-3):
        raise ValueError(f"{label}.K does not match fx/fy/cx/cy")
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def _size_pair(block: object, label: str) -> Tuple[int, int]:
    if not isinstance(block, dict):
        raise ValueError(f"{label} must be an object")
    try:
        width = int(block["width"])
        height = int(block["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} is missing width/height") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} size must be positive")
    return width, height


def _resolve_sibling(root: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a path string")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path


def load_cameras_auto(path: Union[str, Path]) -> CamerasAuto:
    """Ingest ``cameras_auto.json``.

    Geometry used per camera: image name, ``intrinsicsImage``,
    ``cameraToWorld``, and ``metricScale``. ``metricScale`` must be shared.
    ``scene`` is recorded when present and may be null.
    ``frameIndex``, ``intrinsicsNetwork``, ``worldToCamera``, ``position``,
    and any ``depth`` path are ignored. MoGe predicts depth.
    """
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{json_path} is not a JSON object")

    convention = document.get("convention")
    if not isinstance(convention, dict):
        raise ValueError(f"{json_path} has no convention object")
    camera_axes = convention.get("cameraAxes")
    matrix_order = convention.get("matrixOrder")
    units = convention.get("units")
    if camera_axes != "opencvRDF":
        raise ValueError(f"{json_path} cameraAxes must be opencvRDF, got {camera_axes!r}")
    if matrix_order != "columnMajor":
        raise ValueError(f"{json_path} matrixOrder must be columnMajor, got {matrix_order!r}")
    if units != "meters":
        raise ValueError(f"{json_path} units must be meters, got {units!r}")
    world_frame = str(convention.get("frame") or "")

    source = document.get("source")
    if not isinstance(source, dict) or "maxDepthMeters" not in source:
        raise ValueError(f"{json_path} is missing source.maxDepthMeters")
    max_depth_m = _finite_float(source["maxDepthMeters"], f"{json_path} source.maxDepthMeters")
    if max_depth_m <= 0:
        raise ValueError(f"{json_path} source.maxDepthMeters must be positive")

    cameras = document.get("cameras")
    if not isinstance(cameras, list) or len(cameras) == 0:
        raise ValueError(f"{json_path} has no cameras")

    root = json_path.parent
    views: List[CamerasAutoView] = []
    for index, camera in enumerate(cameras):
        label = f"{json_path} cameras[{index}]"
        if not isinstance(camera, dict):
            raise ValueError(f"{label} is not an object")
        image_name = camera.get("image")
        scene = camera.get("scene")
        if not isinstance(image_name, str) or not image_name:
            raise ValueError(f"{label} is missing image")
        if scene is not None and not isinstance(scene, str):
            raise ValueError(f"{label} scene must be a string or null")

        metric_scale = _finite_float(camera.get("metricScale"), f"{label}.metricScale")
        if metric_scale <= 0:
            raise ValueError(f"{label}.metricScale must be positive")

        photo_size = _size_pair(camera.get("originalImage"), f"{label}.originalImage")
        intrinsics_image = _parse_intrinsics_block(camera.get("intrinsicsImage"), f"{label}.intrinsicsImage")
        if (intrinsics_image.width, intrinsics_image.height) != photo_size:
            raise ValueError(f"{label} intrinsicsImage size != originalImage")

        try:
            camera_to_world = matrix_from_column_major(camera["cameraToWorld"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} is missing a 4x4 cameraToWorld") from exc

        views.append(
            CamerasAutoView(
                image_name=image_name,
                scene=scene,
                metric_scale=metric_scale,
                intrinsics_image=intrinsics_image,
                camera_to_world=camera_to_world,
                image_path=_resolve_sibling(root, image_name, f"{label}.image"),
                source_path=json_path,
            )
        )

    scales = np.asarray([view.metric_scale for view in views], dtype=np.float64)
    if not np.allclose(scales, scales[0], rtol=1e-5, atol=1e-6):
        raise ValueError(f"{json_path} metricScale is not shared across cameras: {scales.tolist()}")

    return CamerasAuto(
        world_frame=world_frame,
        camera_axes=str(camera_axes),
        units=str(units),
        matrix_order=str(matrix_order),
        max_depth_m=max_depth_m,
        metric_scale=float(scales[0]),
        views=views,
        path=json_path,
    )


def match_cameras_to_images(
    document: CamerasAuto,
    image_dir: Union[str, Path],
) -> Tuple[List[CamerasAutoView], List[str]]:
    """Match image files in ``image_dir`` to camera entries by filename.

    An image with no camera entry is skipped. A camera whose image file is
    not in that folder is skipped too.
    """
    folder = Path(image_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    by_name: Dict[str, CamerasAutoView] = {}
    for view in document.views:
        key = Path(view.image_name).name
        if key in by_name:
            raise ValueError(f"{document.path} lists {key} more than once")
        by_name[key] = view

    images = [
        path for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    matched: List[CamerasAutoView] = []
    skipped: List[str] = []
    seen = set()
    for image in images:
        view = by_name.get(image.name)
        if view is None:
            skipped.append(image.name)
            continue
        view.image_path = image
        matched.append(view)
        seen.add(image.name)
    for name in by_name:
        if name not in seen:
            skipped.append(name)
    if not matched:
        raise FileNotFoundError(
            f"No image in {folder} has a cameras_auto entry. Skipped {skipped}."
        )
    return matched, skipped


def camera_points_to_world(points_cam: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """``P_world = P_cam @ R.T + t`` in float64. Same pose product as ``transform_points_to_world``."""
    rotation = np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
    origin = np.asarray(camera_to_world, dtype=np.float64)[:3, 3]
    points = np.asarray(points_cam, dtype=np.float64)
    return points @ rotation.T + origin


def scale_world_about_camera(
    points_world: np.ndarray,
    origin: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale a world cloud about a camera origin. The origin itself does not move."""
    points = np.asarray(points_world, dtype=np.float64)
    camera_origin = np.asarray(origin, dtype=np.float64)
    return float(scale) * (points - camera_origin) + camera_origin


def median_valid_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
) -> float:
    """Median finite depth on ``mask`` inside ``[depth_min, depth_max]``."""
    depth_map = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth_map)
    valid &= (depth_map >= depth_min) & (depth_map <= depth_max)
    if not np.any(valid):
        return float("inf")
    return float(np.median(depth_map[valid]))


def choose_reference_index(
    median_depths: Sequence[float],
    override: Optional[int] = None,
) -> int:
    """Pick the nearer view (smallest median depth), or ``override`` when set."""
    if len(median_depths) == 0:
        raise ValueError("No views to choose a reference from")
    if override is not None:
        index = int(override)
        if index < 0 or index >= len(median_depths):
            raise IndexError(
                f"reference index {index} is outside 0..{len(median_depths) - 1}"
            )
        return index
    depths = np.asarray(list(median_depths), dtype=np.float64)
    if not np.any(np.isfinite(depths)):
        return 0
    return int(np.nanargmin(depths))


def per_point_scale(d_ref: np.ndarray, z_rel: np.ndarray, z_ab: float) -> np.ndarray:
    """Scale about the source camera that places each sample on the reference depth."""
    return (np.asarray(d_ref, dtype=np.float64) - float(z_ab)) / np.asarray(z_rel, dtype=np.float64)


def naive_depth_ratio(d_ref: np.ndarray, z_rel: np.ndarray, z_ab: float) -> np.ndarray:
    """``d_ref / Z_ref(P_src)``. Not the scale to apply about the source camera."""
    z_ref = np.asarray(z_rel, dtype=np.float64) + float(z_ab)
    return np.asarray(d_ref, dtype=np.float64) / z_ref


def depth_disagreement(
    d_ref: np.ndarray,
    z_rel: np.ndarray,
    z_ab: float,
    scale: float,
) -> np.ndarray:
    """Absolute meters between reference depth and the scaled source point's Z."""
    predicted = float(scale) * np.asarray(z_rel, dtype=np.float64) + float(z_ab)
    return np.abs(predicted - np.asarray(d_ref, dtype=np.float64))


def _pixel_grid(height: int, width: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    vs = np.arange(0, height, stride, dtype=np.int64)
    us = np.arange(0, width, stride, dtype=np.int64)
    uu, vv = np.meshgrid(us, vs)
    return uu.reshape(-1).astype(np.float64), vv.reshape(-1).astype(np.float64)


def _check_depth_shape(depth: np.ndarray, mask: np.ndarray, intrinsics: CameraIntrinsics, label: str) -> None:
    expected = (int(intrinsics.height), int(intrinsics.width))
    if depth.shape != expected:
        raise ValueError(f"{label} depth shape {depth.shape} != intrinsics image {expected}")
    if mask.shape != expected:
        raise ValueError(f"{label} mask shape {mask.shape} != intrinsics image {expected}")


def unproject_pixels(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Back-project pixel depths through K and ``cameraToWorld`` to world points."""
    z = np.asarray(depth, dtype=np.float64)
    x = (np.asarray(u, dtype=np.float64) - intrinsics.cx) / intrinsics.fx * z
    y = (np.asarray(v, dtype=np.float64) - intrinsics.cy) / intrinsics.fy * z
    points_cam = np.stack([x, y, z], axis=-1)
    return camera_points_to_world(points_cam, camera_to_world)


def _sample_nearest(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor sample. ``inside`` is true when the rounded pixel is in range."""
    height, width = image.shape[:2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    inside &= np.isfinite(u) & np.isfinite(v)
    sampled = np.full(u.shape, np.nan, dtype=np.float64)
    if np.any(inside):
        sampled[inside] = np.asarray(image, dtype=np.float64)[vi[inside], ui[inside]]
    return sampled, inside


def collect_overlap_correspondences(
    depth_ref: np.ndarray,
    mask_ref: np.ndarray,
    intrinsics_ref: CameraIntrinsics,
    camera_to_world_ref: np.ndarray,
    world_to_camera_ref: np.ndarray,
    depth_src: np.ndarray,
    mask_src: np.ndarray,
    intrinsics_src: CameraIntrinsics,
    camera_to_world_src: np.ndarray,
    world_to_camera_src: np.ndarray,
    stride: int = 8,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Project the metric reference into the source image and gather scale terms.

    Returns ``d_ref``, ``z_rel``, ``z_AB``, and the observed source world points
    ``P_src`` (built from the sampled source depth, so they carry the source scale).
    """
    _check_depth_shape(depth_ref, mask_ref, intrinsics_ref, "reference")
    _check_depth_shape(depth_src, mask_src, intrinsics_src, "source")

    u_ref, v_ref = _pixel_grid(intrinsics_ref.height, intrinsics_ref.width, stride)
    ui = u_ref.astype(np.int64)
    vi = v_ref.astype(np.int64)
    d_ref = np.asarray(depth_ref, dtype=np.float64)[vi, ui]
    ref_ok = np.asarray(mask_ref, dtype=bool)[vi, ui]
    ref_ok &= np.isfinite(d_ref) & (d_ref >= depth_min) & (d_ref <= depth_max)
    if not np.any(ref_ok):
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            0.0,
            np.zeros((0, 3), dtype=np.float64),
        )

    u_ref = u_ref[ref_ok]
    v_ref = v_ref[ref_ok]
    d_ref = d_ref[ref_ok]
    points_ref = unproject_pixels(
        u_ref, v_ref, d_ref, intrinsics_ref, camera_to_world_ref
    )

    rotation_w2c = np.asarray(world_to_camera_src, dtype=np.float64)[:3, :3]
    translation_w2c = np.asarray(world_to_camera_src, dtype=np.float64)[:3, 3]
    points_src_cam = points_ref @ rotation_w2c.T + translation_w2c
    z_src = points_src_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u_src = intrinsics_src.fx * points_src_cam[:, 0] / z_src + intrinsics_src.cx
        v_src = intrinsics_src.fy * points_src_cam[:, 1] / z_src + intrinsics_src.cy

    in_front = z_src >= depth_min
    d_src, inside = _sample_nearest(depth_src, u_src, v_src)
    mask_sampled, mask_inside = _sample_nearest(mask_src.astype(np.float64), u_src, v_src)
    src_ok = in_front & inside & mask_inside & (mask_sampled >= 0.5)
    src_ok &= np.isfinite(d_src) & (d_src >= depth_min) & (d_src <= depth_max)

    u_src = u_src[src_ok]
    v_src = v_src[src_ok]
    d_src = d_src[src_ok]
    d_ref = d_ref[src_ok]
    points_src = unproject_pixels(
        np.rint(u_src),
        np.rint(v_src),
        d_src,
        intrinsics_src,
        camera_to_world_src,
    )

    origin_src = np.asarray(camera_to_world_src, dtype=np.float64)[:3, 3]
    origin_ref = np.asarray(camera_to_world_ref, dtype=np.float64)[:3, 3]
    z_axis_ref = np.asarray(world_to_camera_ref, dtype=np.float64)[2, :3]
    z_rel = (points_src - origin_src) @ z_axis_ref
    z_ab = float((origin_src - origin_ref) @ z_axis_ref)

    finite = np.isfinite(z_rel) & np.isfinite(d_ref)
    z_of_src_in_ref = z_rel + z_ab
    finite &= z_of_src_in_ref >= depth_min
    return d_ref[finite], z_rel[finite], z_ab, points_src[finite]


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.05,
    iterations: int = 256,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """RANSAC plane ``n · X = rho``, refit by SVD on the inliers.

    Returns the unit normal and ``rho``. The normal sign is arbitrary; scale
    ratios that use both ``n`` and ``rho`` are unchanged if both flip.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    if len(pts) < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    rng = np.random.default_rng(seed)
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    count = len(pts)
    for _ in range(int(iterations)):
        choice = rng.choice(count, size=3, replace=False)
        edge_a = pts[choice[1]] - pts[choice[0]]
        edge_b = pts[choice[2]] - pts[choice[0]]
        normal = np.cross(edge_a, edge_b)
        length = float(np.linalg.norm(normal))
        if length < 1e-8:
            continue
        normal = normal / length
        rho = float(np.dot(normal, pts[choice[0]]))
        inliers = np.abs(pts @ normal - rho) <= threshold
        inlier_count = int(inliers.sum())
        if inlier_count > best_count:
            best_count = inlier_count
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum()) < 3:
        raise ValueError("Plane RANSAC did not find 3 inliers")

    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    rho = float(np.dot(normal, centroid))
    return normal.astype(np.float64), rho


def plane_point_scales(
    points_world: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    rho: float,
    denom_eps: float = 1e-4,
) -> np.ndarray:
    """Per-point scale about ``origin`` that drops each point onto ``n · X = rho``."""
    camera_origin = np.asarray(origin, dtype=np.float64)
    relative = np.asarray(points_world, dtype=np.float64) - camera_origin
    denom = relative @ np.asarray(normal, dtype=np.float64)
    numer = float(rho) - float(np.dot(normal, camera_origin))
    scales = np.full(len(relative), np.nan, dtype=np.float64)
    ok = np.abs(denom) > denom_eps
    scales[ok] = numer / denom[ok]
    return scales


def _empty_scale_result() -> ScaleResult:
    return ScaleResult(
        scale=1.0,
        estimated_scale=1.0,
        inlier_count=0,
        residual_before_m=None,
        residual_after_m=None,
        naive_depth_ratio=None,
        sufficient_overlap=False,
    )


def _summarize_scales(
    candidate_scales: np.ndarray,
    d_ref: np.ndarray,
    z_rel: np.ndarray,
    z_ab: float,
    *,
    scale_min: float,
    scale_max: float,
    min_inliers: int,
) -> ScaleResult:
    candidates = np.asarray(candidate_scales, dtype=np.float64)
    keep = np.isfinite(candidates) & (candidates >= scale_min) & (candidates <= scale_max)
    inlier_count = int(keep.sum())
    if inlier_count == 0:
        return _empty_scale_result()

    estimated = float(np.median(candidates[keep]))
    d_in = d_ref[keep]
    z_in = z_rel[keep]
    residual_before = float(np.median(depth_disagreement(d_in, z_in, z_ab, 1.0)))
    naive = naive_depth_ratio(d_in, z_in, z_ab)
    naive = naive[np.isfinite(naive)]
    naive_median = float(np.median(naive)) if len(naive) else None
    sufficient = inlier_count >= int(min_inliers)
    applied = estimated if sufficient else 1.0
    residual_after = float(np.median(depth_disagreement(d_in, z_in, z_ab, applied)))
    return ScaleResult(
        scale=applied,
        estimated_scale=estimated,
        inlier_count=inlier_count,
        residual_before_m=residual_before,
        residual_after_m=residual_after,
        naive_depth_ratio=naive_median,
        sufficient_overlap=sufficient,
    )


def estimate_scale(
    depth_ref: np.ndarray,
    mask_ref: np.ndarray,
    intrinsics_ref: CameraIntrinsics,
    camera_to_world_ref: np.ndarray,
    world_to_camera_ref: np.ndarray,
    depth_src: np.ndarray,
    mask_src: np.ndarray,
    intrinsics_src: CameraIntrinsics,
    camera_to_world_src: np.ndarray,
    world_to_camera_src: np.ndarray,
    *,
    method: str = "median_depth_ratio",
    stride: int = 8,
    depth_min: float = DEPTH_MIN_M,
    depth_max: float = DEPTH_MAX_M,
    scale_min: float = SCALE_MIN,
    scale_max: float = SCALE_MAX,
    min_inliers: int = MIN_INLIERS,
    plane_threshold: float = 0.05,
) -> ScaleResult:
    """Estimate the source scale about its camera that matches the reference.

    ``median_depth_ratio`` uses overlap correspondences.
    ``plane`` fits one RANSAC plane on the reference cloud (the floor, on the
    warehouse sample) and solves the same about-camera scale onto that plane.
    Depth residuals are always measured on the overlap correspondences.
    """
    if method not in ("median_depth_ratio", "plane"):
        raise ValueError(f"Unknown align method '{method}'")

    d_ref, z_rel, z_ab, points_src = collect_overlap_correspondences(
        depth_ref,
        mask_ref,
        intrinsics_ref,
        camera_to_world_ref,
        world_to_camera_ref,
        depth_src,
        mask_src,
        intrinsics_src,
        camera_to_world_src,
        world_to_camera_src,
        stride=stride,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    usable = np.abs(z_rel) > Z_REL_EPS
    d_ref = d_ref[usable]
    z_rel = z_rel[usable]
    points_src = points_src[usable]
    if len(d_ref) == 0:
        return _empty_scale_result()

    if method == "median_depth_ratio":
        candidates = per_point_scale(d_ref, z_rel, z_ab)
    else:
        u_ref, v_ref = _pixel_grid(intrinsics_ref.height, intrinsics_ref.width, stride)
        ui = u_ref.astype(np.int64)
        vi = v_ref.astype(np.int64)
        d_plane = np.asarray(depth_ref, dtype=np.float64)[vi, ui]
        plane_ok = np.asarray(mask_ref, dtype=bool)[vi, ui]
        plane_ok &= np.isfinite(d_plane) & (d_plane >= depth_min) & (d_plane <= depth_max)
        if int(plane_ok.sum()) < 3:
            return _empty_scale_result()
        points_ref = unproject_pixels(
            u_ref[plane_ok],
            v_ref[plane_ok],
            d_plane[plane_ok],
            intrinsics_ref,
            camera_to_world_ref,
        )
        try:
            normal, rho = fit_plane_ransac(points_ref, threshold=plane_threshold)
        except ValueError:
            return _empty_scale_result()
        origin_src = np.asarray(camera_to_world_src, dtype=np.float64)[:3, 3]
        candidates = plane_point_scales(points_src, origin_src, normal, rho)

    return _summarize_scales(
        candidates,
        d_ref,
        z_rel,
        z_ab,
        scale_min=scale_min,
        scale_max=scale_max,
        min_inliers=min_inliers,
    )
