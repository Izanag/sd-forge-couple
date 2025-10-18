"""
Blend mode based compositor for shape-derived masks.

Masks are rendered as float arrays in the [0, 1] range and combined
according to the W3C compositing rules (normal, multiply, overlay).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .region_shapes import BlendMode, RegionShape
from .shape_renderer import ShapeRenderer


def blend_normal(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Standard source-over behaviour."""
    return np.ones_like(bottom)


def blend_multiply(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Darken by multiplying the colour channels."""
    return top * bottom


def blend_overlay(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """
    Overlay blend mode.

    Equivalent to multiply for the darker range and screen for the brighter.
    """
    result = np.empty_like(top)
    mask = bottom <= 0.5
    result[mask] = 2.0 * top[mask] * bottom[mask]
    inv_top = 1.0 - top[~mask]
    inv_bottom = 1.0 - bottom[~mask]
    result[~mask] = 1.0 - 2.0 * inv_top * inv_bottom
    return result


def _apply_blend_mode(
    top: np.ndarray, bottom: np.ndarray, mode: BlendMode
) -> np.ndarray:
    if mode is BlendMode.NORMAL:
        return blend_normal(top, bottom)
    if mode is BlendMode.MULTIPLY:
        return blend_multiply(top, bottom)
    if mode is BlendMode.OVERLAY:
        return blend_overlay(top, bottom)
    raise ValueError(f"Unsupported blend mode: {mode}")


def composite_layers(
    layers: Sequence[Tuple[np.ndarray, BlendMode]], base: np.ndarray | None = None
) -> np.ndarray:
    """
    Composite a stack of layers onto the base image.

    Parameters
    ----------
    layers:
        Sequence of (mask_array, blend_mode) tuples.  Each mask_array must
        already be normalised to the [0, 1] range.
    base:
        Optional starting canvas.  Defaults to zeros if omitted.
    """
    if not layers:
        raise ValueError("At least one layer is required for compositing")

    if base is None:
        base = np.zeros_like(layers[0][0], dtype=np.float32)
    else:
        base = base.astype(np.float32, copy=True)

    for mask_array, mode in layers:
        top = np.clip(mask_array.astype(np.float32), 0.0, 1.0)
        bottom = np.clip(base, 0.0, 1.0)

        blended = _apply_blend_mode(top, bottom, mode)

        alpha = top  # treat luminance as coverage
        base = bottom + alpha * (blended - bottom)

    return np.clip(base, 0.0, 1.0)


def composite_region_masks(
    regions: Sequence[RegionShape],
    width: int,
    height: int,
    renderer: ShapeRenderer,
) -> np.ndarray:
    """
    Render and composite a list of RegionShape objects.

    Returns
    -------
    np.ndarray
        Single channel mask in float32 within [0, 1].
    """
    if not regions:
        return np.zeros((height, width), dtype=np.float32)

    layers: List[Tuple[np.ndarray, BlendMode]] = []
    for region in sorted(regions, key=lambda r: r.z_order):
        mask_array = renderer.render_shape_array(region, width, height)
        layers.append((mask_array, region.blend_mode))

    return composite_layers(layers)
