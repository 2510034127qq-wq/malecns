"""Map CNS spike activity onto MuJoCo actuators. No environment-flag shortcuts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray


def motor_from_spikes(
    spike_rate: Mapping[int, float],
    *,
    n_actuator: int,
    motor_gids: Mapping[int, int],
    wall_near: bool | None = None,
) -> NDArray[np.float64]:
    """Translate per-gid spike counts/rates into actuator controls in [-1, 1].

    `wall_near` is accepted only to prove it is ignored. Behavior must not branch
    on high-level environment flags.
    """
    del wall_near
    ctrl = np.zeros(n_actuator, dtype=np.float64)
    for gid, actuator_index in motor_gids.items():
        if 0 <= int(actuator_index) < n_actuator:
            ctrl[int(actuator_index)] = float(np.tanh(0.15 * float(spike_rate.get(int(gid), 0.0))))
    return np.clip(ctrl, -1.0, 1.0)


def classify_motor_role(superclass: str | None, cell_type: str | None) -> str | None:
    sc = (superclass or "").lower()
    ty = (cell_type or "").lower()
    if "vnc_motor" in sc or "cb_motor" in sc:
        if "wing" in ty:
            return "wing"
        if "ti" in ty and "flex" in ty:
            return "tibia_flexor"
        if "tr" in ty and "flex" in ty:
            return "trochanter_flexor"
        return "leg"
    if "descending" in sc:
        return "descending"
    return None
