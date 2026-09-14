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


def named_contact_forces(model, data) -> dict[str, float]:
    """Contact force magnitude keyed by geom name (tarsus, food, walls, ...)."""
    import mujoco

    n = max(int(data.ncon), 0)
    out: dict[str, float] = {}
    if n == 0:
        return out
    cbuf = np.zeros(6, dtype=np.float64)
    id2name = {}
    for i in range(model.ngeom):
        try:
            id2name[i] = str(model.geom(i).name)
        except Exception:
            id2name[i] = str(i)
    for i in range(n):
        mujoco.mj_contactForce(model, data, i, cbuf)
        mag = float(np.linalg.norm(cbuf[:3]))
        con = data.contact[i]
        for gid in (int(con.geom1), int(con.geom2)):
            name = id2name.get(gid, str(gid))
            out[name] = out.get(name, 0.0) + mag
    return out


def proprioception(model, data) -> dict[str, NDArray[np.float64]]:
    return {
        "qpos": np.array(data.qpos, dtype=np.float64, copy=True),
        "qvel": np.array(data.qvel, dtype=np.float64, copy=True),
        "act": np.array(data.act, dtype=np.float64, copy=True) if model.na else np.zeros(0),
    }
