"""Contact and proprioceptive encoding from MuJoCo state."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def contact_forces(model, data) -> NDArray[np.float64]:
    """Net contact force magnitude per geom named as a tarsus/body, else all contacts."""
    n = max(int(data.ncon), 0)
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    forces = np.zeros(n, dtype=np.float64)
    import mujoco

    cbuf = np.zeros(6, dtype=np.float64)
    for i in range(n):
        mujoco.mj_contactForce(model, data, i, cbuf)
        forces[i] = float(np.linalg.norm(cbuf[:3]))
    return forces


def proprioception(model, data) -> dict[str, NDArray[np.float64]]:
    return {
        "qpos": np.array(data.qpos, dtype=np.float64, copy=True),
        "qvel": np.array(data.qvel, dtype=np.float64, copy=True),
        "act": np.array(data.act, dtype=np.float64, copy=True) if model.na else np.zeros(0),
    }
