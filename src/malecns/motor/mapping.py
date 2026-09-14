"""Map CNS spike activity onto MuJoCo actuators. No environment-flag shortcuts.

Neural → actuator assignment is ASSUMED from MaleCNS somaSide, somaNeuromere,
and motor type names. It is not a measured muscle insertion table.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

MotorSpec = int | tuple[int, float]


def motor_from_spikes(
    spike_rate: Mapping[int, float],
    *,
    n_actuator: int,
    motor_gids: Mapping[int, MotorSpec],
    wall_near: bool | None = None,
) -> NDArray[np.float64]:
    """Sum per-gid rates onto actuators, then tanh. Flexor/extensor use opposite signs.

    `wall_near` is accepted only to prove it is ignored. Behavior must not branch
    on high-level environment flags.
    """
    del wall_near
    ctrl = np.zeros(n_actuator, dtype=np.float64)
    for gid, spec in motor_gids.items():
        if isinstance(spec, tuple):
            actuator_index, sign = int(spec[0]), float(spec[1])
        else:
            actuator_index, sign = int(spec), 1.0
        if 0 <= actuator_index < n_actuator:
            ctrl[actuator_index] += sign * float(spike_rate.get(int(gid), 0.0))
    ctrl = np.tanh(0.15 * ctrl)
    return np.clip(ctrl, -1.0, 1.0)


def classify_motor_role(superclass: str | None, cell_type: str | None) -> str | None:
    sc = (superclass or "").lower()
    ty = (cell_type or "").lower()
    if "vnc_motor" in sc or "cb_motor" in sc:
        if "wing" in ty or ty.startswith("dlm") or ty.startswith("dvm"):
            return "wing"
        if "ti" in ty and "flex" in ty:
            return "tibia_flexor"
        if "tr" in ty and "flex" in ty:
            return "trochanter_flexor"
        return "leg"
    if "descending" in sc:
        return "descending"
    return None


def _actuator_id(model, name: str) -> int | None:
    try:
        return int(model.actuator(name).id)
    except Exception:
        return None


def _side(side: str | None, instance: str | None, root_side: str | None = None) -> str | None:
    inst = instance or ""
    if inst.endswith("_L") or "_L_" in inst:
        return "L"
    if inst.endswith("_R") or "_R_" in inst:
        return "R"
    s = (side or "").upper()
    if s in {"L", "R"}:
        return s
    r = (root_side or "").upper()
    if r in {"L", "R"}:
        return r
    return None


def _leg_prefix(side: str | None, neuromere: str | None) -> str | None:
    if side not in {"L", "R"}:
        return None
    n = (neuromere or "").upper()
    if n == "T1":
        return f"{side}1"
    if n == "T2":
        return f"{side}2"
    if n == "T3":
        return f"{side}3"
    return None


def _mn_target(ty: str, side: str | None, neuromere: str | None) -> tuple[str, float] | None:
    """Return (actuator_name, sign). Sign +1 flex/depress, -1 extend/levate. ASSUMED."""
    t = ty.lower()
    if "dlm" in t or "dvm" in t or "wing" in t:
        name = "wing_R_stroke" if side == "R" else "wing_L_stroke"
        return name, 1.0
    if "cem" in t or "neck" in t or "cvn" in t:
        return ("head_yaw" if "yaw" in t or "cem" in t else "head_pitch"), 1.0
    if (neuromere or "").upper().startswith("A") and "ti" not in t and "tr" not in t:
        sign = -1.0 if "extens" in t else 1.0
        return "abdomen_pitch", sign
    leg = _leg_prefix(side, neuromere)
    if leg is None:
        if "mn" in t and side in {"L", "R"}:
            return (f"{side}1_pitch", 1.0)
        return None
    if "ti" in t or "tibia" in t or "ltm1" in t:
        sign = -1.0 if "extens" in t else 1.0
        return f"{leg}_knee", sign
    if "tr" in t or "trochanter" in t or "femur" in t or "fe reductor" in t or "ltm2" in t:
        sign = -1.0 if "extens" in t else 1.0
        return f"{leg}_pitch", sign
    if "rotat" in t or "promotor" in t or "remotor" in t or "abduct" in t or "adduct" in t:
        sign = -1.0 if "posterior" in t or "remotor" in t else 1.0
        return f"{leg}_yaw", sign
    if "tarsus" in t or t.startswith("ta "):
        return f"{leg}_knee", 1.0 if "depress" in t else -1.0
    return f"{leg}_pitch", 1.0


def build_motor_table(catalog, model) -> dict[int, tuple[int, float]]:
    """Map every vnc_motor / cb_motor / descending neuron onto a signed actuator."""
    table: dict[int, tuple[int, float]] = {}
    n = len(catalog)
    for gid in range(n):
        sc = (catalog.superclass[gid] or "").lower()
        ty = catalog.cell_type[gid] or ""
        side = _side(catalog.soma_side[gid], catalog.instance[gid], catalog.root_side[gid])
        neu = catalog.soma_neuromere[gid]
        target: tuple[str, float] | None = None
        if "vnc_motor" in sc or "cb_motor" in sc or "vnc_efferent" in sc or "cb_efferent" in sc:
            target = _mn_target(ty, side, neu)
            if target is None:
                name = "head_yaw" if "cb" in sc else "L1_pitch"
                if side == "R" and name.startswith("L"):
                    name = "R1_pitch"
                target = (name, 1.0)
        elif "descending" in sc:
            t = ty.lower()
            if "wing" in t or "dlm" in t or "p01" in t or "giant" in t:
                target = ("wing_L_stroke" if side != "R" else "wing_R_stroke", 1.0)
            else:
                target = ("head_yaw" if (gid % 2 == 0) else "head_pitch", 1.0)
        if target is None:
            continue
        aid = _actuator_id(model, target[0])
        if aid is None:
            continue
        table[gid] = (aid, target[1])
    return table
