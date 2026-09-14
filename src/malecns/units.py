"""Coordinate and unit conversion at the MaleCNS ingestion boundary.

Native MaleCNS SWC and syn-points XYZ are voxel coordinates in 8 nm units.
Arbor morphology geometry is micrometers.

    1 MaleCNS coordinate unit = 8 nm = 0.008 µm

Apply this factor to morphology x/y/z/radius and synapse x/y/z before any
spatial mapping. Raw files stay unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

MALE_CNS_VOXEL_NM = 8.0
NM_PER_UM = 1000.0
MALE_CNS_UNITS_TO_UM = MALE_CNS_VOXEL_NM / NM_PER_UM  # 0.008
UM_PER_M = 1e6

_ATOL_UM = 1e-9


def native_to_um(values: ArrayLike) -> NDArray[np.float64] | tuple[float, ...]:
    """Scale native 8 nm units to micrometers.

    Sequences of Python numbers round-trip as tuples so tests can compare
    exactly; NumPy arrays stay arrays.
    """
    if isinstance(values, (tuple, list)):
        return tuple(float(v) * MALE_CNS_UNITS_TO_UM for v in values)
    array = np.asarray(values, dtype=np.float64)
    return array * MALE_CNS_UNITS_TO_UM


def um_to_m(values: ArrayLike) -> NDArray[np.float64]:
    """Micrometers to SI meters (MuJoCo)."""
    return np.asarray(values, dtype=np.float64) / UM_PER_M


def assert_same_physical_space(
    morph_xyz_um: Sequence[float] | NDArray[np.float64],
    synapse_xyz_um: Sequence[float] | NDArray[np.float64],
    *,
    atol: float = _ATOL_UM,
) -> None:
    """Require morphology and synapse coordinates to share µm space."""
    morph = np.asarray(morph_xyz_um, dtype=np.float64)
    syn = np.asarray(synapse_xyz_um, dtype=np.float64)
    if morph.shape[-1] < 3 or syn.shape[-1] < 3:
        raise ValueError("expected xyz (optionally radius) coordinates")
    if morph.shape != syn.shape:
        raise ValueError(
            f"morphology and synapse coordinate shapes differ: {morph.shape} vs {syn.shape}"
        )
    if not np.allclose(morph[..., :3], syn[..., :3], atol=atol, rtol=0.0):
        raise AssertionError(
            "morphology and synapse coordinates are not in the same physical space"
        )
