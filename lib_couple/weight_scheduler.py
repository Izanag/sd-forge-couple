"""Dynamic weight scheduling utilities for the Forge Couple extension.

This module keeps weight parsing and interpolation logic separate from the core
region mapping features so it can be reused by UIs or future extensions.

Supported syntax formats:
    - Static float: ``"1.5"`` keeps legacy behaviour and returns ``1.5``.
    - Full schedule: ``"[0.5:1.5:0.0:1.0:linear]"`` (percentage) or
      ``"[0.5:1.5:0:20:ease-in]"`` (step based).
    - Shorthand schedule: ``"[0.5:1.5:cosine]"`` which implies ``0-1`` percent.

Curve types:
    linear, ease-in, ease-out, ease-in-out, cosine, sigmoid, exponential,
    bounce.

Example:
    >>> schedule = parse_weight("[0.5:1.5:0.0:1.0:cosine]")
    >>> calculate_weight(schedule.start_weight, schedule.end_weight, 0.25, schedule.curve_type)
    0.75
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Union

from lib_couple.logging import logger

SUPPORTED_CURVES: tuple[str, ...] = (
    "linear",
    "ease-in",
    "ease-out",
    "ease-in-out",
    "cosine",
    "sigmoid",
    "exponential",
    "bounce",
)

_NUMBER = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
_FULL_PATTERN = re.compile(
    rf"""
    \[
        \s*{_NUMBER}\s*:\s*
        {_NUMBER}\s*:\s*
        {_NUMBER}\s*:\s*
        {_NUMBER}\s*:\s*
        ([\w\-]+)
    \s*\]
    """,
    re.VERBOSE,
)
_SHORTHAND_PATTERN = re.compile(
    rf"""
    \[
        \s*{_NUMBER}\s*:\s*
        {_NUMBER}\s*:\s*
        ([\w\-]+)
    \s*\]
    """,
    re.VERBOSE,
)
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_SIGMOID_K = 12.0
_SIGMOID_S0 = 1 / (1 + math.exp(0.5 * _SIGMOID_K))
_SIGMOID_S1 = 1 / (1 + math.exp(-0.5 * _SIGMOID_K))
_SIGMOID_RANGE = _SIGMOID_S1 - _SIGMOID_S0


@dataclass
class WeightSchedule:
    """Container representing a dynamic weight schedule."""

    start_weight: float
    end_weight: float
    start_range: float
    end_range: float
    curve_type: str
    is_percentage: bool

    def __post_init__(self) -> None:
        self.curve_type = self.curve_type.lower()
        if not math.isfinite(self.start_weight) or not math.isfinite(self.end_weight):
            raise ValueError("Weights must be finite numbers.")
        is_valid, error = validate_weight_schedule(self)
        if not is_valid:
            logger.error("Invalid weight schedule: %s (%s)", self, error)
            raise ValueError(error)

        # Warn when weights exceed typical SD ranges but still allow them.
        for label, value in (("start", self.start_weight), ("end", self.end_weight)):
            if not 0.0 <= value <= 10.0:
                logger.warning(
                    "WeightSchedule %s weight %.3f is outside recommended range (0-10)",
                    label,
                    value,
                )

    def __str__(self) -> str:  # pragma: no cover - trivial
        range_label = "%" if self.is_percentage else "steps"
        return (
            f"WeightSchedule({self.start_weight:.3f}->{self.end_weight:.3f} "
            f"over {self.start_range:.3f}-{self.end_range:.3f} {range_label}, "
            f"{self.curve_type})"
        )


def parse_weight(weight_str: str) -> Union[float, WeightSchedule]:
    """Parse weight expression into either a float or a WeightSchedule.

    Args:
        weight_str: Raw user-provided string.

    Returns:
        A float for legacy static weights or a WeightSchedule for dynamic ones.

    Raises:
        ValueError: If the syntax does not match any supported format.

    Examples:
        >>> parse_weight("1.25")
        1.25
        >>> parse_weight("[0.5:1.5:0.0:1.0:linear]")
        WeightSchedule(...)
        >>> parse_weight("[0.5:1.5:cosine]")
        WeightSchedule(...)
    """

    if weight_str is None:
        raise ValueError("Weight must be a non-empty string.")

    text = weight_str.strip()
    if not text:
        raise ValueError("Weight must be a non-empty string.")

    try:
        static_value = float(text)
    except ValueError:
        pass
    else:
        if not math.isfinite(static_value):
            raise ValueError("Static weight must be a finite number.")
        return static_value

    match = _FULL_PATTERN.fullmatch(text)
    if match:
        start_weight, end_weight, start_range, end_range = map(
            float, match.group(1, 2, 3, 4)
        )
        raw_start_range, raw_end_range = match.group(3), match.group(4)
        curve = match.group(5).lower()
        is_percentage = _infer_percentage(
            raw_start_range, raw_end_range, start_range, end_range
        )
        return WeightSchedule(
            start_weight=start_weight,
            end_weight=end_weight,
            start_range=start_range,
            end_range=end_range,
            curve_type=curve,
            is_percentage=is_percentage,
        )

    match = _SHORTHAND_PATTERN.fullmatch(text)
    if match:
        start_weight, end_weight = map(float, match.group(1, 2))
        curve = match.group(3).lower()
        return WeightSchedule(
            start_weight=start_weight,
            end_weight=end_weight,
            start_range=0.0,
            end_range=1.0,
            curve_type=curve,
            is_percentage=True,
        )

    message = (
        "Invalid weight format: '{value}'. Expected one of:\n"
        "  - Static float, e.g. '1.0'\n"
        "  - Full schedule, e.g. '[0.5:1.5:0.0:1.0:linear]'\n"
        "  - Shorthand schedule, e.g. '[0.5:1.5:cosine]'"
    ).format(value=weight_str)
    logger.error(message)
    raise ValueError(message)


def calculate_weight(
    start_weight: float, end_weight: float, progress: float, curve_type: str
) -> float:
    """Interpolate a weight value using the requested easing curve.

    Args:
        start_weight: Value returned when ``progress`` equals ``0.0``.
        end_weight: Value returned when ``progress`` equals ``1.0``.
        progress: Normalised position between ``0.0`` and ``1.0`` (clamped).
        curve_type: Name of the easing function (see ``SUPPORTED_CURVES``).

    Returns:
        The interpolated weight after applying the easing curve.

    Notes:
        Progress values outside the expected range are clamped to protect
        against sampler rounding errors.

    Example:
        >>> calculate_weight(0.0, 1.0, 0.5, "ease-in")
        0.25
    """

    curve = (curve_type or "linear").lower()
    # Clamp to avoid surprises from floating-point accumulation in samplers.
    progress = max(0.0, min(1.0, progress))

    try:
        if curve == "linear":
            t = progress
        elif curve == "ease-in":
            t = progress**2
        elif curve == "ease-out":
            t = 1 - (1 - progress) ** 2
        elif curve == "ease-in-out":
            t = progress**2 * (3 - 2 * progress)
        elif curve == "cosine":
            t = (1 - math.cos(progress * math.pi)) / 2
        elif curve == "sigmoid":
            x = (progress - 0.5) * _SIGMOID_K
            s = 1 / (1 + math.exp(-x))
            t = (s - _SIGMOID_S0) / _SIGMOID_RANGE
        elif curve == "exponential":
            numerator = math.exp(progress * 2) - 1
            denominator = math.exp(2) - 1
            t = numerator / denominator
        elif curve == "bounce":
            if progress < 0.5:
                t = 2 * progress
            else:
                oscillation = math.sin((progress - 0.5) * math.pi * 4)
                t = 1 - abs(oscillation) * 0.2
        else:
            logger.warning("Unknown curve type '%s', using linear fallback.", curve)
            t = progress
    except OverflowError as error:
        logger.error("Curve calculation overflowed (%s). Falling back to linear.", error)
        t = progress

    return start_weight + (end_weight - start_weight) * t


def validate_weight_schedule(schedule: WeightSchedule) -> tuple[bool, str]:
    """Return (is_valid, message) indicating whether a schedule is usable."""

    if schedule.start_range >= schedule.end_range:
        return False, "start_range must be smaller than end_range"

    if schedule.curve_type not in SUPPORTED_CURVES:
        return False, f"Unsupported curve '{schedule.curve_type}'"

    if schedule.is_percentage:
        if not (0.0 <= schedule.start_range < schedule.end_range <= 1.0):
            return False, "Percentage ranges must stay within 0.0-1.0"
    else:
        if schedule.start_range < 0 or schedule.end_range < 0:
            return False, "Step ranges must be non-negative"

    return True, ""


def get_curve_samples(curve_type: str, num_samples: int = 100) -> list[float]:
    """Return representative samples for plotting a curve."""

    if num_samples <= 1:
        raise ValueError("num_samples must be greater than 1")

    step = 1 / (num_samples - 1)
    return [
        calculate_weight(0.0, 1.0, idx * step, curve_type)
        for idx in range(num_samples)
    ]


def _infer_percentage(
    raw_start: str, raw_end: str, start: float, end: float
) -> bool:
    """Infer whether a schedule uses percentages or sampler steps."""

    if "." in raw_start or "." in raw_end:
        return True

    start_is_int = bool(_INTEGER_PATTERN.fullmatch(raw_start))
    end_is_int = bool(_INTEGER_PATTERN.fullmatch(raw_end))

    if (not start_is_int or not end_is_int) and max(start, end) <= 1.0:
        return True

    return False


__all__ = [
    "SUPPORTED_CURVES",
    "WeightSchedule",
    "calculate_weight",
    "get_curve_samples",
    "parse_weight",
    "validate_weight_schedule",
]
