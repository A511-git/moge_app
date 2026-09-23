import torch

_CMAP_LUT_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}


def get_cmap_lut_torch(cmap_name: str, device: torch.device) -> torch.Tensor:
    key = (cmap_name, str(device))
    if key not in _CMAP_LUT_CACHE:
        lut_np = matplotlib.colormaps[cmap_name](np.linspace(0, 1, 256))[..., :3]
        lut_uint8 = (lut_np * 255.0).clip(0, 255).astype(np.uint8)
        _CMAP_LUT_CACHE[key] = torch.tensor(lut_uint8, dtype=torch.uint8, device=device)
    return _CMAP_LUT_CACHE[key]


def colorize_depth_torch(depth: torch.Tensor, mask: Optional[torch.Tensor] = None, normalize: bool = True, cmap: str = 'Spectral') -> torch.Tensor:
    if mask is None:
        valid = (depth > 0) & torch.isfinite(depth)
    else:
        valid = (depth > 0) & mask & torch.isfinite(depth)

    if not valid.any():
        return torch.zeros((*depth.shape, 3), dtype=torch.uint8, device=depth.device)

    disp = 1.0 / torch.clamp(depth, min=1e-4)

    if normalize:
        valid_disp = disp[valid]
        num_elements = valid_disp.numel()
        if num_elements > 100_000:
            step = max(1, num_elements // 50_000)
            sample_disp = valid_disp[::step]
        else:
            sample_disp = valid_disp
        q = torch.quantile(sample_disp.float(), torch.tensor([0.001, 0.99], device=depth.device))
        min_disp, max_disp = q[0], q[1]
        disp_norm = (disp - min_disp) / torch.clamp(max_disp - min_disp, min=1e-6)
    else:
        disp_norm = disp

    val = torch.clamp(1.0 - disp_norm, 0.0, 1.0)
    indices = torch.clamp((val * 255.0).round().long(), 0, 255)

    lut = get_cmap_lut_torch(cmap, depth.device)
    colored = lut[indices]
    colored = torch.where(valid.unsqueeze(-1), colored, torch.zeros_like(colored))
    return colored


def colorize_normal_torch(normal: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    scale = torch.tensor([0.5, -0.5, -0.5], device=normal.device, dtype=normal.dtype)
    n_colored = normal * scale + 0.5
    n_uint8 = torch.clamp(n_colored * 255.0, 0.0, 255.0).to(torch.uint8)
    if mask is not None:
        n_uint8 = torch.where(mask.unsqueeze(-1), n_uint8, torch.zeros_like(n_uint8))
    return n_uint8


def colorize_depth(depth: Union[np.ndarray, torch.Tensor], mask: Optional[Union[np.ndarray, torch.Tensor]] = None, normalize: bool = True, cmap: str = 'Spectral') -> Union[np.ndarray, torch.Tensor]:
    if isinstance(depth, torch.Tensor):
        return colorize_depth_torch(depth, mask=mask, normalize=normalize, cmap=cmap)
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], nan=0.0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_depth_affine(depth: np.ndarray, mask: np.ndarray = None, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        depth = np.where(mask, depth, np.nan)

    min_depth, max_depth = np.nanquantile(depth, 0.001), np.nanquantile(depth, 0.999)
    depth = (depth - min_depth) / (max_depth - min_depth)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](depth)[..., :3], nan=0.0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_disparity(disparity: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        disparity = np.where(mask, disparity, np.nan)
    
    if normalize:
        min_disp, max_disp = np.nanquantile(disparity, 0.001), np.nanquantile(disparity, 0.999)
        disparity = (disparity - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disparity)[..., :3], nan=0.0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_segmentation(segmentation: np.ndarray, cmap: str = 'Set1') -> np.ndarray:
    colored = matplotlib.colormaps[cmap]((segmentation % 20) / 20)[..., :3]
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_normal(normal: Union[np.ndarray, torch.Tensor], mask: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(normal, torch.Tensor):
        return colorize_normal_torch(normal, mask=mask)
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal


def colorize_error_map(error_map: np.ndarray, mask: np.ndarray = None, cmap: str = 'plasma', value_range: Tuple[float, float] = None):
    vmin, vmax = value_range if value_range is not None else (np.nanmin(error_map), np.nanmax(error_map))
    cmap = matplotlib.colormaps[cmap]
    colorized_error_map = cmap(((error_map - vmin) / (vmax - vmin)).clip(0, 1))[..., :3]
    if mask is not None:
        colorized_error_map = np.where(mask[..., None], colorized_error_map, 0)
    colorized_error_map = np.ascontiguousarray((colorized_error_map.clip(0, 1) * 255).astype(np.uint8))
    return colorized_error_map
