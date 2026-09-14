"""Whole-CNS level of detail: soma points in µm, not a per-frame synapse dump."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from malecns.units import MALE_CNS_UNITS_TO_UM

_SUPERCLASS_RGB = {
    "ol_sensory": (0.95, 0.85, 0.2),
    "cb_sensory": (0.9, 0.5, 0.1),
    "vnc_sensory": (0.85, 0.4, 0.2),
    "descending_neuron": (0.95, 0.15, 0.15),
    "ascending_neuron": (0.95, 0.45, 0.15),
    "vnc_motor": (0.2, 0.4, 0.95),
    "cb_motor": (0.35, 0.6, 1.0),
    "ol_intrinsic": (0.45, 0.55, 0.7),
    "cb_intrinsic": (0.5, 0.5, 0.55),
    "vnc_intrinsic": (0.4, 0.5, 0.65),
    "visual_projection": (0.7, 0.85, 0.2),
}


def soma_cloud_um(
    soma_xyz_native: NDArray[np.float64],
    superclass: Sequence[str | None],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    native = np.asarray(soma_xyz_native, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != 3:
        raise ValueError("soma_xyz_native must be (N, 3)")
    xyz = (native * MALE_CNS_UNITS_TO_UM).astype(np.float32)
    rgb = np.full((native.shape[0], 3), 0.55, dtype=np.float32)
    for i, sc in enumerate(superclass):
        if sc in _SUPERCLASS_RGB:
            rgb[i] = _SUPERCLASS_RGB[sc]
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], rgb[valid]
