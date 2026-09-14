#!/usr/bin/env python3
import os
import sys
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# Ensure parent directory is in sys.path for direct script execution
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import json
import itertools
import time
from typing import Optional

import cv2
import click
import numpy as np
import torch
from tqdm import tqdm, trange

try:
    from standalone_moge.custom_deps import utils3d_moge as utils3d
except ImportError:
    try:
        from .custom_deps import utils3d_moge as utils3d
    except (ImportError, ValueError):
        try:
            import utils3d_moge as utils3d
        except ImportError:
            import utils3d

try:
    from standalone_moge.model import MoGeModel
    from standalone_moge.utils.vis import colorize_depth, colorize_normal
    from standalone_moge.utils.panorama import (
        spherical_uv_to_directions,
        get_panorama_cameras,
        split_panorama_image,
        merge_panorama_depth
    )
    from standalone_moge.utils.download_weights import download_single, normalize_model_name, MODELS
except ImportError:
    from .model import MoGeModel
    from .utils.vis import colorize_depth, colorize_normal
    from .utils.panorama import (
        spherical_uv_to_directions,
        get_panorama_cameras,
        split_panorama_image,
        merge_panorama_depth
    )
    from .utils.download_weights import download_single, normalize_model_name, MODELS



@click.command(help='Standalone MoGe-3 Panorama Inference CLI (Docker / Direct CLI)')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True, help='Input panorama image or folder path (JPG/PNG). [REQUIRED via CLI]')
@click.option('--output', '-o', 'output_path', type=click.Path(), envvar='MOGE_OUTPUT', default='./output', show_default=True, help='Output directory for generated artifacts. [env: MOGE_OUTPUT]')
@click.option('--model', '-m', 'model_name', type=click.Choice(['vitl', 'vitg', 'moge-3-vitl', 'moge-3-vitg']), envvar='MOGE_MODEL', default='vitl', show_default=True, help='MoGe-3 model variant: "vitl" (370M) or "vitg" (1.25B). [env: MOGE_MODEL]')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, envvar='MOGE_PRETRAINED', default=None, help='Custom model path or HuggingFace repo (e.g. /checkpoints/moge-3-vitl.pt or Ruicheng/moge-3-vitg). [env: MOGE_PRETRAINED]')
@click.option('--checkpoint_dir', 'checkpoint_dir', type=click.Path(), envvar='CHECKPOINT_DIR', default=None, help='Directory where model checkpoints are stored and auto-downloaded. [env: CHECKPOINT_DIR]')
@click.option('--device', 'device_name', type=str, envvar='MOGE_DEVICE', default='cuda', show_default=True, help='Device (e.g. "cuda", "cuda:0", "cpu"). [env: MOGE_DEVICE]')
@click.option('--fp16', 'use_fp16', is_flag=True, envvar='MOGE_FP16', help='Use FP16 precision for faster inference. [env: MOGE_FP16]')
@click.option('--resize', 'resize_to', type=int, envvar='MOGE_RESIZE', default=None, help='Max dimension ceiling (default: None = keep original resolution). [env: MOGE_RESIZE]')
@click.option('--resolution_level', type=int, envvar='MOGE_RESOLUTION_LEVEL', default=9, show_default=True, help='Inference resolution level [0-9]. [env: MOGE_RESOLUTION_LEVEL]')
@click.option('--num_tokens', type=int, envvar='MOGE_NUM_TOKENS', default=None, help='Number of tokens (overrides resolution_level). [env: MOGE_NUM_TOKENS]')
@click.option('--refine_steps', type=click.IntRange(min=0), envvar='MOGE_REFINE_STEPS', default=3, show_default=True, help='Sparse 3D refinement steps for v3. [env: MOGE_REFINE_STEPS]')
@click.option('--split_resolution', type=int, envvar='MOGE_SPLIT_RESOLUTION', default=512, show_default=True, help='Resolution for each splitted perspective view. [env: MOGE_SPLIT_RESOLUTION]')
@click.option('--batch_size', type=int, envvar='MOGE_BATCH_SIZE', default=4, show_default=True, help='Batch size for perspective view inference. [env: MOGE_BATCH_SIZE]')
@click.option('--debug', 'save_debug', is_flag=True, envvar='MOGE_DEBUG', help='Save debug artifacts (splitted perspective views, distance maps, and camera JSON metadata). [env: MOGE_DEBUG]')
@click.option('--maps', 'save_maps_', is_flag=True, envvar='MOGE_MAPS', help='Save visual maps and raw EXRs (depth.exr, points.exr, depth_vis.png, normal_vis.png, mask.png). [env: MOGE_MAPS]')
@click.option('--depth_npy/--no-depth_npy', 'save_depth_npy', envvar='MOGE_DEPTH_NPY', default=True, show_default=True, help='Save primary depth.npy float32 array. [env: MOGE_DEPTH_NPY]')
@click.option('--points_npy', 'save_points_npy', is_flag=True, envvar='MOGE_POINTS_NPY', help='Save 3D coordinates points.npy float32 array. [env: MOGE_POINTS_NPY]')
def main(
    input_path: str,
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
    split_resolution: int,
    batch_size: int,
    save_debug: bool,
    save_maps_: bool,
    save_depth_npy: bool,
    save_points_npy: bool
):
    """
    Executes standalone MoGe-3 panorama inference CLI on single images or entire folders.
    Supports switching between ViT-L (370M) and ViT-G (1.25B) backbones.
    """
    # Strict resolution level check
    if not (0 <= resolution_level <= 9):
        raise click.BadParameter(f"resolution_level must be between 0 and 9, got {resolution_level}")

    device = torch.device(device_name if torch.cuda.is_available() and "cuda" in device_name else "cpu")

    # Discover input images
    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'webp', 'WEBP']
    input_p = Path(input_path)
    if input_p.is_dir():
        image_paths = sorted(itertools.chain(*(input_p.rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [input_p]

    if not image_paths:
        raise FileNotFoundError(f"No valid panorama image files found at: {input_path}")

    # Determine model variant and weights
    model_key = normalize_model_name(model_name)
    model_params = MODELS.get(model_key, (None, None, "Unknown"))[2]

    # Resolve checkpoints directory
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
    elif os.environ.get("CHECKPOINT_DIR"):
        ckpt_dir = Path(os.environ["CHECKPOINT_DIR"])
    elif Path("/checkpoints").is_dir():
        ckpt_dir = Path("/checkpoints")
    else:
        ckpt_dir = Path("checkpoints")

    if pretrained_model_name_or_path is None:
        # Check if local checkpoint exists in ckpt_dir or download ONLY this configured model
        pretrained_model_name_or_path = download_single(model_key, ckpt_dir)

    if use_fp16 and (device == "cpu" or not torch.cuda.is_available()):
        if device != "cpu":
            device = "cpu"
        use_fp16 = False
        print("[MoGe CLI] Note: FP16 is only supported on CUDA GPU devices. Disabling fp16 for CPU inference.")

    print(f"[MoGe CLI] Loading MoGe-3 ({model_key.upper()} - {model_params}) from '{pretrained_model_name_or_path}' on {device} (fp16={use_fp16})...")
    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    print(f"[MoGe CLI] Processing {len(image_paths)} image(s) for 5 runs...")
    for run_idx in range(5):
        print(f"\n[MoGe CLI] --- Starting Run {run_idx + 1}/5 ---")

        for image_path in tqdm(image_paths, desc=f"Total Panoramas (Run {run_idx+1})", disable=len(image_paths) <= 1):
            run_start_time = time.time()
            image_bgr = cv2.imread(str(image_path))

            if image_bgr is None:

                print(f"[WARNING] Skipping unreadable image: {image_path}")

                continue


            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            orig_height, orig_width = image_rgb.shape[:2]

            image = image_rgb.copy()


            # Handle optional resize

            target_height, target_width = orig_height, orig_width

            if resize_to is not None and (orig_height > resize_to or orig_width > resize_to):

                target_height = min(resize_to, int(resize_to * orig_height / orig_width))

                target_width = min(resize_to, int(resize_to * orig_width / orig_height))

                image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


            # Output folder per image

            if input_p.is_dir():

                rel_parent = image_path.relative_to(input_p).parent

                save_path = Path(output_path, rel_parent, image_path.stem)

            else:

                save_path = Path(output_path, image_path.stem)

            save_path.mkdir(exist_ok=True, parents=True)


            t0 = time.time()
            # 1. Split equirectangular panorama into perspective views

            splitted_extrinsics, splitted_intrinsics = get_panorama_cameras()

            splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intrinsics, split_resolution)


            t1 = time.time()
            print(f"[Timing] Split panorama: {t1 - t0:.3f}s")
            # 2. Infer views

            splitted_distance_maps, splitted_masks = [], []

            if save_debug:

                splitted_depth_maps, splitted_points_maps, splitted_normal_maps = [], [], []


            for i in trange(0, len(splitted_images), batch_size, desc="Inferring views", leave=False, disable=len(splitted_images) <= batch_size):

                batch_slice = splitted_images[i:i + batch_size]

                image_tensor = torch.tensor(

                    np.stack(batch_slice) / 255.0,

                    dtype=torch.float32,

                    device=device

                ).permute(0, 3, 1, 2)


                fov_x, _ = np.rad2deg(utils3d.np.intrinsics_to_fov(np.array(splitted_intrinsics[i:i + batch_size])))

                fov_x_tensor = torch.tensor(fov_x, dtype=torch.float32, device=device)


                infer_kwargs = {

                    'fov_x': fov_x_tensor,

                    'resolution_level': resolution_level,

                    'apply_mask': False,

                    'refine_steps': refine_steps,

                    'use_fp16': use_fp16,

                }

                if num_tokens is not None:

                    infer_kwargs['num_tokens'] = num_tokens


                with torch.no_grad():

                    output = model.infer(image_tensor, **infer_kwargs)


                distance_map = output['points'].norm(dim=-1).cpu().numpy()

                mask = output['mask'].cpu().numpy()

                splitted_distance_maps.extend(list(distance_map))

                splitted_masks.extend(list(mask))


                if save_debug:

                    splitted_depth_maps.extend(list(output['depth'].cpu().numpy()))

                    splitted_points_maps.extend(list(output['points'].cpu().numpy()))

                    if 'normal' in output and output['normal'] is not None:

                        splitted_normal_maps.extend(list(output['normal'].cpu().numpy()))


            # Save splitted views if requested in debug mode

            if save_debug:

                splitted_dir = save_path / 'splitted'

                splitted_dir.mkdir(exist_ok=True, parents=True)

                cameras_meta = []

                for i in range(len(splitted_images)):

                    cv2.imwrite(str(splitted_dir / f'{i:02d}.jpg'), cv2.cvtColor(splitted_images[i], cv2.COLOR_RGB2BGR))

                    cv2.imwrite(str(splitted_dir / f'{i:02d}_mask.png'), (splitted_masks[i] * 255).astype(np.uint8))

                    cv2.imwrite(str(splitted_dir / f'{i:02d}_depth.exr'), splitted_depth_maps[i], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    cv2.imwrite(str(splitted_dir / f'{i:02d}_depth_vis.png'), cv2.cvtColor(colorize_depth(splitted_depth_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))

                    cv2.imwrite(str(splitted_dir / f'{i:02d}_distance.exr'), splitted_distance_maps[i], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    cv2.imwrite(str(splitted_dir / f'{i:02d}_distance_vis.png'), cv2.cvtColor(colorize_depth(splitted_distance_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))

                    cv2.imwrite(str(splitted_dir / f'{i:02d}_points.exr'), splitted_points_maps[i], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                    if len(splitted_normal_maps) > i:

                        cv2.imwrite(str(splitted_dir / f'{i:02d}_normal.png'), cv2.cvtColor(colorize_normal(splitted_normal_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))


                    fov_xi, fov_yi = np.rad2deg(utils3d.np.intrinsics_to_fov(splitted_intrinsics[i]))

                    cam_info = {

                        'index': i,

                        'image': f'{i:02d}.jpg',

                        'fov_x': round(float(fov_xi), 2),

                        'fov_y': round(float(fov_yi), 2),

                        'intrinsics': splitted_intrinsics[i].tolist(),

                        'extrinsics': splitted_extrinsics[i].tolist(),

                    }

                    cameras_meta.append(cam_info)

                    with open(splitted_dir / f'{i:02d}_camera.json', 'w') as f:

                        json.dump(cam_info, f, indent=2)


                with open(splitted_dir / 'cameras.json', 'w') as f:

                    json.dump({'views': cameras_meta}, f, indent=2)


            t2 = time.time()
            print(f"[Timing] Infer views: {t2 - t1:.3f}s")
            # 3. Merge panoramic depth using sparse linear solver

            merging_width, merging_height = min(1920, target_width), min(960, target_height)

            panorama_depth, panorama_mask = merge_panorama_depth(

                merging_width,

                merging_height,

                splitted_distance_maps,

                splitted_masks,

                splitted_extrinsics,

                splitted_intrinsics

            )

            panorama_depth = panorama_depth.astype(np.float32)


            t3 = time.time()
            print(f"[Timing] Merge panorama: {t3 - t2:.3f}s")
            # 4. Upscale back to EXACT original input image dimensions

            if panorama_depth.shape[:2] != (target_height, target_width):

                panorama_depth = cv2.resize(panorama_depth, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

                panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST) > 0


            # Compute 3D coordinate points only if requested for points.npy or visual maps

            points = None

            if save_points_npy or save_maps_:

                uv = utils3d.np.uv_map(target_height, target_width)

                directions = spherical_uv_to_directions(uv)

                points = panorama_depth[:, :, None] * directions


            # Track generated file paths

            npy_files = {}

            map_files = {}

            debug_files = {}


            # Write primary depth.npy

            if save_depth_npy:

                depth_npy_p = save_path / 'depth.npy'

                np.save(str(depth_npy_p), panorama_depth)

                npy_files['depth.npy'] = str(depth_npy_p.resolve())


            if save_points_npy:

                points_npy_p = save_path / 'points.npy'

                np.save(str(points_npy_p), points)

                npy_files['points.npy'] = str(points_npy_p.resolve())


            # Write optional visual maps

            if save_maps_:

                normals, normals_mask = utils3d.np.point_map_to_normal_map(points, panorama_mask)


                img_p = save_path / 'image.jpg'

                cv2.imwrite(str(img_p), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                map_files['image.jpg'] = str(img_p.resolve())


                dvis_p = save_path / 'depth_vis.png'

                cv2.imwrite(str(dvis_p), cv2.cvtColor(colorize_depth(panorama_depth, mask=panorama_mask), cv2.COLOR_RGB2BGR))

                map_files['depth_vis.png'] = str(dvis_p.resolve())


                nvis_p = save_path / 'normal_vis.png'

                cv2.imwrite(str(nvis_p), cv2.cvtColor(colorize_normal(normals, mask=normals_mask), cv2.COLOR_RGB2BGR))

                map_files['normal_vis.png'] = str(nvis_p.resolve())


                dexr_p = save_path / 'depth.exr'

                cv2.imwrite(str(dexr_p), panorama_depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                map_files['depth.exr'] = str(dexr_p.resolve())


                pexr_p = save_path / 'points.exr'

                cv2.imwrite(str(pexr_p), points, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

                map_files['points.exr'] = str(pexr_p.resolve())


                mask_p = save_path / 'mask.png'

                cv2.imwrite(str(mask_p), (panorama_mask * 255).astype(np.uint8))

                map_files['mask.png'] = str(mask_p.resolve())


            if save_debug:

                debug_files['debug_folder'] = str((save_path / 'splitted').resolve())


            t4 = time.time()
            print(f"[Timing] Upscale & Postprocess: {t4 - t3:.3f}s")
            print(f"[Timing] TOTAL Image Processing: {t4 - run_start_time:.3f}s")
            # Print detailed generated file paths in CLI

            print("\n" + "=" * 65)

            print(f"📦 Artifacts Generated for: {image_path.name}")

            print("=" * 65)

            if npy_files:

                print("NPY Files:")

                for k, v in npy_files.items():

                    print(f"  • {k:<18} : {v}")

            if map_files:

                print("\nMaps Files:")

                for k, v in map_files.items():

                    print(f"  • {k:<18} : {v}")

            if debug_files:

                print("\nDebug Files (Splitted Views):")

                for k, v in debug_files.items():

                    print(f"  • {k:<18} : {v}")

            print("=" * 65)


    print(f"\n[MoGe CLI] Inference finished successfully! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
