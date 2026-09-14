from typing import *
from numbers import Number
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

try:
    from ..custom_deps import utils3d_moge as utils3d
except (ImportError, ValueError):
    try:
        from standalone_moge.custom_deps import utils3d_moge as utils3d
    except ImportError:
        try:
            from .. import utils3d_moge as utils3d
        except (ImportError, ValueError):
            try:
                from standalone_moge import utils3d_moge as utils3d
            except ImportError:
                import utils3d_moge as utils3d

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift
from .utils import (
    wrap_module_with_gradient_checkpointing,
    unwrap_module_with_gradient_checkpointing,
    wrap_module_with_autocast
)
from .modules.dinov2_encoder import DINOv2Encoder
from .modules.mlp import MLP
from .modules.conv_stack import ConvStack
from .modules.sparse_unet import Sparse3DUNet


class MoGeModel(nn.Module):
    """
    Self-contained MoGe-3 model architecture supporting both ViT-L (370M) and ViT-G (1.25B)
    with self-guided sparse volumetric refinement.
    """
    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    mask_head: ConvStack
    normal_head: ConvStack
    scale_head: MLP
    refiner: Sparse3DUNet
    onnx_compatible_mode: bool

    def __init__(
        self,
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: Literal['linear', 'sinh', 'exp', 'sinh_exp'] = 'exp',
        num_tokens_range: List[int] = [1200, 3600],
        refiner: Optional[Dict[str, Any]] = None,
        refiner_depth_resolution: float = 256,
        **deprecated_kwargs,
    ):
        super(MoGeModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range

        self.encoder = DINOv2Encoder(**encoder)
        self.neck = ConvStack(**neck)
        if points_head is not None:
            self.points_head = ConvStack(**points_head)
            self.points_head.fp32_output_projection = True
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head)
            self.normal_head.fp32_output_projection = True
        if scale_head is not None:
            self.scale_head = MLP(**scale_head)

        if refiner is not None:
            refiner_cfg = dict(refiner)
            self.refiner_depth_resolution = refiner_depth_resolution
            self.refiner = Sparse3DUNet(**refiner_cfg)
        else:
            warnings.warn("Warning: refiner is not enabled.")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def onnx_compatible_mode(self) -> bool:
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, IO[bytes]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        **hf_kwargs
    ) -> 'MoGeModel':
        """
        Load a model from a local checkpoint file or Hugging Face repository.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to checkpoint file or HF repo ID (e.g. Ruicheng/moge-3-vitl, Ruicheng/moge-3-vitg).
        - `model_kwargs`: additional keyword arguments to override parameters in checkpoint.
        - `hf_kwargs`: additional keyword arguments passed to `hf_hub_download`. Ignored for local paths.

        ### Returns:
        - A new instance of `MoGeModel` with parameters loaded from checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        if missing_keys:
            warnings.warn(
                f"{len(missing_keys)} parameter(s) are absent from the checkpoint and keep their random "
                f"initialization: {missing_keys}"
            )
        if unexpected_keys:
            warnings.warn(
                f"{len(unexpected_keys)} parameter(s) in the checkpoint have no counterpart in the model "
                f"and were ignored: {unexpected_keys}"
            )

        return model

    def init_weights(self):
        self.encoder.init_weights()
        if hasattr(self, 'refiner'):
            self.refiner.init_weights()

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        for head in ['points_head', 'normal_head', 'mask_head']:
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()
        if hasattr(self, 'refiner'):
            self.refiner.enable_gradient_checkpointing()

    def enable_mixed_precision(self, dtype: torch.dtype = torch.bfloat16):
        """Enable fine-grained mixed precision: run encoder in `dtype`, keep neck and heads in fp32."""
        for handle in getattr(self, '_autocast_handles', []):
            handle.remove()

        module_dtype_map = [
            (self.encoder, dtype),
            (self.neck, torch.float32),
            *((getattr(self, head, None), torch.float32) for head in ['points_head', 'normal_head', 'mask_head', 'scale_head']),
        ]
        self._autocast_handles = [
            wrap_module_with_autocast(module, device_type='cuda', dtype=module_dtype)
            for module, module_dtype in module_dtype_map if module is not None
        ]

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == 'linear':
            pass
        elif self.remap_output == 'sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def _voxelize(
        self,
        point_coord: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.Tensor]:
        """
        Convert dense point coordinates to a sparse representation.

        - point_coord: [B, H, W, 3] at (x/z, y/z, logz).

        Returns (feats, coords, shape, logz):
        - feats:  (M, 3) fp32 input features [uv, logz].
        - coords: (M, 4) int32, columns (batch, i, j, z_bin).
        - shape:  Size([B, H, W, z_extent, in_channels]).
        - logz:   [B, H, W] dense fp32 log-depth (for residual update).
        """
        if point_coord.ndim != 4 or point_coord.shape[-1] != 3:
            raise ValueError(f"point_coord must be [B, H, W, 3], got {point_coord.shape}")

        point_coord = point_coord.float()
        bsz, height, width, _ = point_coord.shape
        device = point_coord.device

        logz = point_coord[..., 2]
        zq = torch.round(logz * self.refiner_depth_resolution).long()
        z_offset = zq.amin(dim=(1, 2), keepdim=True)
        z_idx = zq - z_offset
        z_extent = z_idx.amax().item() + 1

        i = torch.arange(height, device=device, dtype=torch.long).view(1, height, 1).expand(bsz, height, width)
        j = torch.arange(width, device=device, dtype=torch.long).view(1, 1, width).expand(bsz, height, width)
        batch = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1, 1).expand(bsz, height, width)
        coords = torch.stack([batch, i, j, z_idx], dim=-1).reshape(-1, 4).to(torch.int32)

        feats = point_coord.reshape(-1, 3)
        shape = torch.Size([bsz, height, width, z_extent, feats.shape[-1]])
        return feats, coords, shape, logz

    def _refine_logz(
        self,
        point_coord: torch.Tensor,
        encoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        bsz, height, width, _ = point_coord.shape
        feats, coords, shape, logz = self._voxelize(point_coord)
        out: torch.Tensor = self.refiner(feats, coords, shape, encoder_feature)
        out_logz = out.float().squeeze(-1).reshape(bsz, height, width)
        refined_logz = logz + out_logz
        return refined_logz

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor],
        refine_steps: int = 3,
        refiner_detach_backbone: bool = True,
        return_per_step: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if refine_steps > 0 and not hasattr(self, 'refiner'):
            raise ValueError("Refiner is not enabled but refine_steps > 0.")

        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # Backbone encoding
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level,
                height=base_h * 2 ** level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        neck_features = self.neck(features)

        # Heads decoding
        raw_coord = self.points_head(neck_features)[-1] if hasattr(self, 'points_head') else None
        normal, mask = (
            getattr(self, head)(neck_features)[-1] if hasattr(self, head) else None
            for head in ['normal_head', 'mask_head']
        )
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None

        # Refine point map in factorized coordinate space
        coord_per_step: List[torch.Tensor] = []
        coords: Optional[torch.Tensor] = None
        points: Optional[torch.Tensor] = None
        points_per_step: Optional[List[torch.Tensor]] = None
        if raw_coord is not None:  # raw_coord is B3HW at (x/z, y/z, logz)
            current_coord = raw_coord.permute(0, 2, 3, 1).float()  # BHW3 at (x/z, y/z, logz)
            if return_per_step:
                coord_per_step.append(current_coord)

            if refine_steps > 0:
                refiner_feature: torch.Tensor = features[0]

                for _ in range(refine_steps):
                    feature_for_refiner = refiner_feature.detach() if refiner_detach_backbone else refiner_feature
                    refined_logz = self._refine_logz(current_coord.detach(), feature_for_refiner)
                    current_coord = torch.cat([current_coord[..., :2], refined_logz.unsqueeze(-1)], dim=-1)
                    if return_per_step:
                        coord_per_step.append(current_coord)

            coords = torch.stack(coord_per_step, dim=1) if return_per_step else current_coord.unsqueeze(1)

        # Resize and remap outputs
        resize = lambda x, channel_last=False: F.interpolate(
            x.movedim(-1, -3) if channel_last else x,
            (img_h, img_w),
            mode='bilinear',
            align_corners=False,
            antialias=False,
        ).movedim(-3, -1 if channel_last else -3)

        if coords is not None:
            num_point_steps = coords.shape[1]
            coords = resize(coords.flatten(0, 1), channel_last=True)
            coords = coords.unflatten(0, (batch_size, num_point_steps))
            points_all = self._remap_points(coords)
            points = points_all[:, -1]
            if return_per_step:
                points_per_step = list(points_all.unbind(dim=1))

        if normal is not None:
            normal = resize(normal)
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = resize(mask)
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            'points': points,
            'points_per_step': points_per_step,
            'normal': normal,
            'mask': mask,
            'metric_scale': metric_scale,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        refine_steps: int = 3,
        return_per_step: bool = False,
        use_fp16: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function for MoGe-3.

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W).
        - `num_tokens`: number of base ViT tokens to use for inference. If None, computed from `resolution_level`.
        - `resolution_level`: inference resolution level from 0 to 9. Higher values use more tokens. Default: 9.
        - `force_projection`: if True, recompute each point map from depth map and intrinsics. Default: True.
        - `apply_mask`: if True, mask invalid points and depths using the predicted mask. Default: True.
        - `fov_x`: horizontal camera field of view in degrees. If None, inferred automatically. Default: None.
        - `refine_steps`: number of sparse 3D refinement updates (0-5). Default: 3.
        - `return_per_step`: if True, return predictions for initial estimate and all refinement steps. Default: False.
        - `use_fp16`: if True, use mixed precision to accelerate inference. Default: False.

        ### Returns
        Dictionary containing predictions:
        - `points`: camera-space point map of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: depth map of shape (B, H, W) or (H, W).
        - `intrinsics`: camera intrinsics matrix of shape (B, 3, 3) or (3, 3).
        - `mask`: predicted validity mask of shape (B, H, W) or (H, W).
        - `normal`: surface normal map of shape (B, H, W, 3) or (H, W, 3).
        - `points_per_step`, `depth_per_step`, `intrinsics_per_step` (when `return_per_step=True`).
        """
        if refine_steps > 0 and not hasattr(self, 'refiner'):
            raise ValueError("Refiner is not enabled but refine_steps > 0.")

        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        is_cuda = (self.device.type == 'cuda')
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=bool(use_fp16 and is_cuda and self.dtype != torch.float16)):
            output = self.forward(image, num_tokens=num_tokens, refine_steps=refine_steps, return_per_step=return_per_step)
        affine_points, normal, mask, metric_scale = (output.get(k, None) for k in ['points', 'normal', 'mask', 'metric_scale'])
        affine_points_per_step = output.get('points_per_step', None)

        # Always process output in fp32 precision
        if affine_points_per_step is None:
            affine_points_per_step = [affine_points] if affine_points is not None else None
        affine_points_per_step = [p.float() for p in affine_points_per_step] if affine_points_per_step is not None else None
        normal, mask, metric_scale, fov_x = map(
            lambda x: x.float() if isinstance(x, torch.Tensor) else x,
            [normal, mask, metric_scale, fov_x]
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None

            if affine_points_per_step is not None:
                if fov_x is not None:
                    focal_fixed = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=self.device, dtype=torch.float32) / 2))
                    if focal_fixed.ndim == 0:
                        focal_fixed = focal_fixed[None].expand(affine_points_per_step[-1].shape[0])
                else:
                    focal_fixed = None

                points_per_step, depth_per_step, intrinsics_per_step = [], [], []
                for affine_points in affine_points_per_step:
                    if focal_fixed is None:
                        focal_i, shift_i = recover_focal_shift(affine_points, mask_binary)
                    else:
                        focal_i = focal_fixed
                        _, shift_i = recover_focal_shift(affine_points, mask_binary, focal=focal_i)
                    fx_i = focal_i / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
                    fy_i = focal_i / 2 * (1 + aspect_ratio ** 2) ** 0.5
                    intrinsics_i = utils3d.pt.intrinsics_from_focal_center(fx_i, fy_i, 0.5, 0.5)

                    points = affine_points.clone()
                    points[..., 2] += shift_i[..., None, None]
                    depth = points[..., 2].clone()

                    if force_projection:
                        points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics_i)

                    if metric_scale is not None:
                        points *= metric_scale[:, None, None, None]
                        depth *= metric_scale[:, None, None]

                    points_per_step.append(points)
                    depth_per_step.append(depth)
                    intrinsics_per_step.append(intrinsics_i)

                intrinsics = intrinsics_per_step[-1]

                if mask_binary is not None:
                    mask_per_step = [mask_binary & (d > 0) for d in depth_per_step]
                    mask_binary = mask_per_step[-1]
                else:
                    mask_per_step = None

                points = points_per_step[-1]
                depth = depth_per_step[-1]
            else:
                points_per_step = None
                depth_per_step = None
                intrinsics_per_step = None
                mask_per_step = None
                points, depth, intrinsics = None, None, None

            if apply_mask:
                if mask_per_step is not None:
                    points_per_step = [torch.where(m[..., None], p, torch.inf) for p, m in zip(points_per_step, mask_per_step)]
                    depth_per_step = [torch.where(m, d, torch.inf) for d, m in zip(depth_per_step, mask_per_step)]
                    points, depth = points_per_step[-1], depth_per_step[-1]
                if mask_binary is not None and normal is not None:
                    normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

            if not return_per_step:
                points_per_step = None
                depth_per_step = None
                intrinsics_per_step = None

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'normal': normal,
            'points_per_step': points_per_step,
            'intrinsics_per_step': intrinsics_per_step,
            'depth_per_step': depth_per_step,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {
                k: [item.squeeze(0) for item in v] if isinstance(v, list) else v.squeeze(0)
                for k, v in return_dict.items()
            }

        return return_dict
