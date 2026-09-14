"""Spatial odor concentration field sampled at antenna sites."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def odor_concentration(
    xyz: NDArray[np.float64],
    sources: list[tuple[float, float, float, float]],
    wind: NDArray[np.float64] | None = None,
    *,
    length_m: float = 0.008,
) -> float:
    """Distance-decayed concentration with optional wind advection of the peak.

    C = sum strength * exp(-||x - (source + wind_shift)||^2 / (2 L^2))
    This is a computationally cheap physically motivated plume, not a CFD solve.
    """
    p = np.asarray(xyz, dtype=np.float64).reshape(3)
    w = np.zeros(3) if wind is None else np.asarray(wind, dtype=np.float64).reshape(3)
    total = 0.0
    two_l2 = 2.0 * length_m * length_m
    for sx, sy, sz, strength in sources:
        center = np.array([sx, sy, sz], dtype=np.float64) + w
        d2 = float(np.sum((p - center) ** 2))
        total += float(strength) * np.exp(-d2 / two_l2)
    return float(max(total, 0.0))


def sample_antennas(model, data, sources, wind=None) -> tuple[float, float]:
    left = data.site("antenna_L").xpos
    right = data.site("antenna_R").xpos
    return (
        odor_concentration(np.asarray(left), sources, wind),
        odor_concentration(np.asarray(right), sources, wind),
    )
