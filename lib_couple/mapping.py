from base64 import b64decode as decode
from io import BytesIO as bIO
from typing import Iterable, List

import numpy as np
import torch
from modules.prompt_parser import SdConditioning
from PIL import Image

from .blend_compositor import composite_region_masks
from .region_shapes import RegionShape
from .shape_renderer import ShapeRenderer


def empty_tensor(h: int, w: int):
    return torch.zeros((h, w)).unsqueeze(0)


def basic_mapping(
    sd_model,
    couples: list,
    width: int,
    height: int,
    line_count: int,
    is_horizontal: bool,
    background: str,
    tile_size: int,
    tile_weight: float,
    bg_weight: float,
) -> dict:
    fc_args: dict = {}

    for tile in range(line_count):
        # ===== Cond =====
        texts = SdConditioning([couples[tile]], False, width, height, None)
        cond = sd_model.get_learned_conditioning(texts)
        pos_cond = [[cond["crossattn"]]] if sd_model.is_sdxl else [[cond]]
        fc_args[f"cond_{tile + 1}"] = pos_cond
        # ===== Cond =====

        # ===== Mask =====
        mask = torch.zeros((height, width))

        if background == "First Line":
            if tile == 0:
                mask = torch.ones((height, width)) * bg_weight
            else:
                if is_horizontal:
                    mask[:, (tile - 1) * tile_size : tile * tile_size] = tile_weight
                else:
                    mask[(tile - 1) * tile_size : tile * tile_size, :] = tile_weight
        else:
            if is_horizontal:
                mask[:, tile * tile_size : (tile + 1) * tile_size] = tile_weight
            else:
                mask[tile * tile_size : (tile + 1) * tile_size, :] = tile_weight

        fc_args[f"mask_{tile + 1}"] = mask.unsqueeze(0)
        # ===== Mask =====

    if background == "Last Line":
        fc_args[f"mask_{line_count}"] = (
            torch.ones((height, width)) * bg_weight
        ).unsqueeze(0)

    return fc_args


def advanced_mapping(
    sd_model, couples: list, width: int, height: int, mapping: list
) -> dict:
    fc_args: dict = {}
    assert len(couples) == len(mapping)

    for tile_index, (x1, x2, y1, y2, w) in enumerate(mapping):
        # ===== Cond =====
        texts = SdConditioning([couples[tile_index]], False, width, height, None)
        cond = sd_model.get_learned_conditioning(texts)
        pos_cond = [[cond["crossattn"]]] if sd_model.is_sdxl else [[cond]]
        fc_args[f"cond_{tile_index + 1}"] = pos_cond
        # ===== Cond =====

        # ===== Mask =====
        x_from = int(width * x1)
        x_to = int(width * x2)
        y_from = int(height * y1)
        y_to = int(height * y2)

        mask = torch.zeros((height, width))
        mask[y_from:y_to, x_from:x_to] = w
        fc_args[f"mask_{tile_index + 1}"] = mask.unsqueeze(0)
        # ===== Mask =====

    return fc_args


@torch.no_grad()
def b64image2tensor(img: str | Image.Image, width: int, height: int) -> torch.Tensor:
    if isinstance(img, str):
        image_bytes = decode(img)
        image = Image.open(bIO(image_bytes)).convert("L")
    else:
        image = img.convert("L")

    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)

    return image


def _is_shape_entry(entry: dict) -> bool:
    return isinstance(entry, dict) and (
        "shape_type" in entry or "shapes" in entry
    )


def _ensure_shape_list(entry: dict) -> List[RegionShape]:
    if "mask" in entry:
        raise ValueError("Cannot mix raster masks and shape definitions in one group")

    if "shapes" in entry:
        shapes_source = entry["shapes"]
        if not isinstance(shapes_source, Iterable) or isinstance(
            shapes_source, (str, bytes)
        ):
            raise ValueError("shapes must be an iterable of shape definitions")
        shape_dicts = list(shapes_source)
        if not shape_dicts:
            return []
        return [RegionShape.from_dict(shape_dict) for shape_dict in shape_dicts]

    return [RegionShape.from_dict(entry)]


def mask_mapping(
    sd_model,
    couples: list,
    width: int,
    height: int,
    line_count: int,
    mapping: list[dict],
    background: str,
    bg_weight: float,
    *,
    use_shapes: bool = False,
) -> dict:
    """
    Prepare mask tensors for the Couple attention pipeline.

    Parameters
    ----------
    mapping:
        Either legacy raster masks (dicts with ``mask`` & ``weight``) or
        shape descriptions (dicts containing ``shape_type``).
    use_shapes:
        Forces interpretation of the mapping entries as shape definitions.
        When False, the format is auto-detected.
    """
    fc_args: dict = {}

    renderer = ShapeRenderer()
    tensors: list[torch.Tensor] = []
    for index, entry in enumerate(mapping):
        if use_shapes or _is_shape_entry(entry):
            try:
                regions = _ensure_shape_list(entry)
            except Exception as exc:
                entry_type = type(entry).__name__
                raise ValueError(
                    f"Failed to parse shape mapping entry at index {index} "
                    f"({entry_type}): {exc}"
                ) from exc

            mask_array = composite_region_masks(regions, width, height, renderer)
            tensors.append(torch.from_numpy(mask_array.copy()).unsqueeze(0))
            continue

        if isinstance(entry, dict) and "mask" in entry:
            weight = float(entry.get("weight", 1.0))
            tensors.append(b64image2tensor(entry["mask"], width, height) * weight)
            continue

        entry_type = type(entry).__name__
        raise ValueError(
            f"Unsupported mapping entry at index {index}: expected a dict with "
            f"'mask' data or shape definition, got {entry_type}"
        )

    for layer in range(line_count):
        # ===== Cond =====
        texts = SdConditioning([couples[layer]], False, width, height, None)
        cond = sd_model.get_learned_conditioning(texts)
        pos_cond = [[cond["crossattn"]]] if sd_model.is_sdxl else [[cond]]
        fc_args[f"cond_{layer + 1}"] = pos_cond
        # ===== Cond =====

        # ===== Mask =====
        mask = torch.zeros((height, width))

        if background == "First Line":
            mask = (
                tensors[layer - 1]
                if layer > 0
                else torch.ones((height, width)) * bg_weight
            )
        elif background == "Last Line":
            mask = (
                tensors[layer]
                if layer < line_count - 1
                else torch.ones((height, width)) * bg_weight
            )
        else:
            mask = tensors[layer]

        fc_args[f"mask_{layer + 1}"] = mask.unsqueeze(0) if mask.dim() == 2 else mask
        # ===== Mask =====

    return fc_args
