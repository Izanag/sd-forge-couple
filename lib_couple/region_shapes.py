"""
Data structures describing vector-based mask regions for Forge Couple.

The legacy mask workflow stores binary raster images only.  This module
introduces a shape-centric representation that keeps the analytic
definition of a region (rectangle, ellipse, polygon, Bézier) together
with rendering metadata that is required for on-demand rasterisation.

All coordinates are expressed in the unit interval [0, 1] and are later
scaled to the target canvas size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Sequence


class ShapeType(str, Enum):
    """Supported analytic region primitives."""

    RECTANGLE = "RECTANGLE"
    ELLIPSE = "ELLIPSE"
    POLYGON = "POLYGON"
    BEZIER = "BEZIER"


class BlendMode(str, Enum):
    """Blend modes recognised by the layered compositor."""

    NORMAL = "NORMAL"
    MULTIPLY = "MULTIPLY"
    OVERLAY = "OVERLAY"


def _ensure_normalised(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]; got {value!r}")
    return float(value)


def _ensure_sequence(
    values: Sequence[Sequence[float]], minimum: int, name: str
) -> List[List[float]]:
    if len(values) < minimum:
        raise ValueError(f"{name} requires at least {minimum} entries")
    normalised = []
    for idx, pair in enumerate(values):
        if len(pair) != 2:
            raise ValueError(f"{name}[{idx}] must contain two values (x, y)")
        x, y = pair
        normalised.append(
            [_ensure_normalised(float(x), f"{name}[{idx}].x"), _ensure_normalised(float(y), f"{name}[{idx}].y")]
        )
    return normalised


@dataclass(slots=True)
class RegionShape:
    """
    Serializable, validated description of a single vector region.

    Attributes
    ----------
    shape_type:
        Primitive type used to define the region.  Supported values are
        `RECTANGLE`, `ELLIPSE`, `POLYGON`, and `BEZIER`.
    parameters:
        Shape specific payload.  Values depend on `shape_type`:
            - RECTANGLE: {"x1": float, "y1": float, "x2": float, "y2": float}
              All coordinates are normalised corner positions.
            - ELLIPSE: {"cx": float, "cy": float, "rx": float, "ry": float}
              Centre point with normalised radii (fractions of width/height).
            - POLYGON: {"points": [(x, y), ...]}
              List of at least three normalised vertices.
            - BEZIER: {"control_points": [(x, y), ...]}
              Cubic Bézier control points.  Multiple concatenated segments
              may be provided; the list length must be 4 + 3 * n.
    z_order:
        Render order relative to other regions; lower values are rendered
        first.
    blend_mode:
        How the rasterised region should be composited with the existing
        mask stack.  Supported values mirror `BlendMode`.
    feather:
        Soft edge percentage expressed in the unit interval [0, 1].  The
        renderer converts this to a blur radius relative to the canvas size.
    hardness:
        Controls the transition falloff.  A value of 1.0 keeps the edge
        sharp, whereas 0.0 applies the full feather radius.
    weight:
        Final scalar multiplier applied after rasterisation.  Values should
        stay within [0, ∞); the compositor will clamp results to [0, 1].
    """

    shape_type: ShapeType
    parameters: Dict[str, Any] = field(default_factory=dict)
    z_order: int = 0
    blend_mode: BlendMode = BlendMode.NORMAL
    feather: float = 0.0
    hardness: float = 1.0
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the region into JSON friendly primitives."""
        return {
            "shape_type": self.shape_type.value,
            "parameters": self.parameters,
            "z_order": self.z_order,
            "blend_mode": self.blend_mode.value,
            "feather": self.feather,
            "hardness": self.hardness,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionShape":
        """Create a validated RegionShape from an arbitrary dictionary."""
        if "shape_type" not in data:
            raise ValueError("shape_type is required for RegionShape deserialisation")

        shape_type = ShapeType(data["shape_type"])
        parameters = dict(data.get("parameters", {}))
        blend_mode = BlendMode(data.get("blend_mode", BlendMode.NORMAL.value))

        instance = cls(
            shape_type=shape_type,
            parameters=parameters,
            z_order=int(data.get("z_order", 0)),
            blend_mode=blend_mode,
            feather=float(data.get("feather", 0.0)),
            hardness=float(data.get("hardness", 1.0)),
            weight=float(data.get("weight", 1.0)),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """Ensure that parameters and metadata fall within valid bounds."""
        self.feather = float(self.feather)
        if not 0.0 <= self.feather <= 1.0:
            raise ValueError("feather must be within [0, 1]")

        self.hardness = float(self.hardness)
        if not 0.0 <= self.hardness <= 1.0:
            raise ValueError("hardness must be within [0, 1]")

        if self.weight < 0.0:
            raise ValueError("weight must be non-negative")

        if self.shape_type is ShapeType.RECTANGLE:
            self._validate_rectangle()
        elif self.shape_type is ShapeType.ELLIPSE:
            self._validate_ellipse()
        elif self.shape_type is ShapeType.POLYGON:
            self._validate_polygon()
        elif self.shape_type is ShapeType.BEZIER:
            self._validate_bezier()
        else:
            raise ValueError(f"Unsupported shape type: {self.shape_type}")

    def _validate_rectangle(self) -> None:
        required = ("x1", "y1", "x2", "y2")
        self._assert_params(required)
        x1 = _ensure_normalised(float(self.parameters["x1"]), "x1")
        y1 = _ensure_normalised(float(self.parameters["y1"]), "y1")
        x2 = _ensure_normalised(float(self.parameters["x2"]), "x2")
        y2 = _ensure_normalised(float(self.parameters["y2"]), "y2")

        if not (x1 < x2 and y1 < y2):
            raise ValueError("Rectangle must have positive area (x1<x2 and y1<y2)")

    def _validate_ellipse(self) -> None:
        required = ("cx", "cy", "rx", "ry")
        self._assert_params(required)
        _ensure_normalised(float(self.parameters["cx"]), "cx")
        _ensure_normalised(float(self.parameters["cy"]), "cy")
        rx = float(self.parameters["rx"])
        ry = float(self.parameters["ry"])
        if rx <= 0.0 or ry <= 0.0:
            raise ValueError("Ellipse radii must be positive")
        _ensure_normalised(rx, "rx")
        _ensure_normalised(ry, "ry")

    def _validate_polygon(self) -> None:
        self._assert_params(("points",))
        points = _ensure_sequence(self.parameters["points"], 3, "points")
        self.parameters["points"] = points

    def _validate_bezier(self) -> None:
        self._assert_params(("control_points",))
        control_points = _ensure_sequence(
            self.parameters["control_points"], 4, "control_points"
        )
        if (len(control_points) - 1) % 3 != 0:
            raise ValueError(
                "control_points length must be 4 + 3n for concatenated cubic segments"
            )
        self.parameters["control_points"] = control_points

    def _assert_params(self, expected: Iterable[str]) -> None:
        missing = [key for key in expected if key not in self.parameters]
        if missing:
            raise ValueError(
                f"Missing required parameters for {self.shape_type.value}: {missing}"
            )
