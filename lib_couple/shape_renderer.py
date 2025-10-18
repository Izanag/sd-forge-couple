"""
Vector shape rasteriser for Forge Couple.

The renderer consumes :class:`RegionShape` definitions and produces
anti-aliased grayscale masks using a simple supersampling strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .region_shapes import RegionShape, ShapeType


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _to_pixel(value: float, size: int) -> int:
    """Convert a normalised coordinate into pixel space."""
    return int(round(_clamp(value, 0.0, 1.0) * (size - 1)))


@dataclass(slots=True)
class ShapeRenderer:
    """Rasterise :class:`RegionShape` objects into grayscale PIL images."""

    supersample_factor: int = 4

    def render_shape_array(
        self, shape: RegionShape, width: int, height: int
    ) -> np.ndarray:
        """
        Rasterise a vector region into a float mask in [0, 1].

        Parameters
        ----------
        shape:
            Region definition to render.
        width, height:
            Target resolution in pixels.
        """

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        shape.validate()

        ss = max(1, int(self.supersample_factor))
        ss_width = width * ss
        ss_height = height * ss

        canvas = Image.new("L", (ss_width, ss_height), 0)
        draw = ImageDraw.Draw(canvas)

        match shape.shape_type:
            case ShapeType.RECTANGLE:
                rect = self._denormalise_rectangle(shape.parameters, ss_width, ss_height)
                draw.rectangle(rect, fill=255)
            case ShapeType.ELLIPSE:
                bbox = self._denormalise_ellipse(shape.parameters, ss_width, ss_height)
                draw.ellipse(bbox, fill=255)
            case ShapeType.POLYGON:
                points = self._denormalise_polygon(shape.parameters, ss_width, ss_height)
                draw.polygon(points, fill=255)
            case ShapeType.BEZIER:
                path = self._denormalise_bezier(shape.parameters, ss_width, ss_height)
                if len(path) < 3:
                    raise ValueError("Bézier sampling produced an invalid path")
                draw.polygon(path, fill=255)
            case _:
                raise ValueError(f"Unsupported shape type: {shape.shape_type}")

        canvas = self._apply_feather(canvas, shape, width, height)
        canvas = canvas.resize((width, height), resample=Image.Resampling.LANCZOS)

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        if shape.weight != 1.0:
            array *= float(shape.weight)
        np.clip(array, 0.0, 1.0, out=array)

        return array

    def render_shape(self, shape: RegionShape, width: int, height: int) -> Image.Image:
        """
        Rasterise a vector region into a grayscale image suitable for previews.
        """

        array = self.render_shape_array(shape, width, height)
        return Image.fromarray((array * 255.0).astype(np.uint8), mode="L")

    def _apply_feather(
        self, mask_image: Image.Image, shape: RegionShape, width: int, height: int
    ) -> Image.Image:
        """Apply feathering / hardness falloff using a Gaussian blur."""
        edge_feather = 0.0
        if shape.feather_edges:
            try:
                edge_feather = max(float(value) for value in shape.feather_edges.values())
            except (TypeError, ValueError):
                edge_feather = 0.0

        uniform_feather = max(float(shape.feather), edge_feather)
        if uniform_feather <= 0.0 or shape.hardness >= 1.0:
            return mask_image

        radius_norm = uniform_feather * min(width, height)
        hardness = _clamp(shape.hardness, 0.0, 1.0)
        effective = radius_norm * (1.0 - hardness) * self.supersample_factor
        if effective <= 0.0:
            return mask_image

        return mask_image.filter(ImageFilter.GaussianBlur(radius=effective))

    def _denormalise_rectangle(
        self, params: dict, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        x1 = _to_pixel(params["x1"], width)
        y1 = _to_pixel(params["y1"], height)
        x2 = _to_pixel(params["x2"], width)
        y2 = _to_pixel(params["y2"], height)

        # Ensure a valid drawable area
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)

        return x1, y1, x2, y2

    def _denormalise_ellipse(
        self, params: dict, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        cx = _to_pixel(params["cx"], width)
        cy = _to_pixel(params["cy"], height)
        rx = max(1, int(round(_clamp(params["rx"], 0.0, 1.0) * width)))
        ry = max(1, int(round(_clamp(params["ry"], 0.0, 1.0) * height)))

        x1 = _clamp(cx - rx, 0, width - 1)
        y1 = _clamp(cy - ry, 0, height - 1)
        x2 = _clamp(cx + rx, 0, width - 1)
        y2 = _clamp(cy + ry, 0, height - 1)

        return int(x1), int(y1), int(x2), int(y2)

    def _denormalise_polygon(
        self, params: dict, width: int, height: int
    ) -> List[Tuple[int, int]]:
        points = params.get("points", [])
        return [
            (int(_to_pixel(x, width)), int(_to_pixel(y, height))) for x, y in points
        ]

    def _denormalise_bezier(
        self, params: dict, width: int, height: int, samples_per_segment: int = 80
    ) -> List[Tuple[int, int]]:
        control_points = params.get("control_points", [])
        if len(control_points) < 4:
            raise ValueError("Bézier curves require at least four control points")

        coords = [
            (float(_to_pixel(x, width)), float(_to_pixel(y, height)))
            for x, y in control_points
        ]

        points: List[Tuple[int, int]] = []
        segment_count = (len(coords) - 1) // 3

        for idx in range(segment_count):
            p0, p1, p2, p3 = coords[idx * 3 : idx * 3 + 4]
            for j, t in enumerate(np.linspace(0.0, 1.0, samples_per_segment, endpoint=True)):
                if idx > 0 and j == 0:
                    # avoid duplicate vertices at segment boundaries
                    continue
                x = (
                    (1 - t) ** 3 * p0[0]
                    + 3 * (1 - t) ** 2 * t * p1[0]
                    + 3 * (1 - t) * t**2 * p2[0]
                    + t**3 * p3[0]
                )
                y = (
                    (1 - t) ** 3 * p0[1]
                    + 3 * (1 - t) ** 2 * t * p1[1]
                    + 3 * (1 - t) * t**2 * p2[1]
                    + t**3 * p3[1]
                )
                points.append((int(round(x)), int(round(y))))

        if points[0] != points[-1]:
            points.append(points[0])

        return points
