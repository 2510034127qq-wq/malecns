"""Quasi-steady wing aerodynamics applied as MuJoCo body wrenches.

Literature: Dickinson, Lehmann & Sane 1999 Science; Sane 2003 J Exp Biol.
Coefficients are literature-derived / assumed, not MaleCNS measurements.
"""

from __future__ import annotations

import numpy as np

# Air density at 25 C, 1 atm.
RHO_AIR = 1.184  # kg/m^3  LITERATURE_DERIVED
WING_AREA = 2.0e-6  # m^2  LITERATURE_DERIVED (~Drosophila wing area)
CL = 1.5  # ASSUMED mid-stroke lift coefficient in the Dickinson range
CD = 1.0  # ASSUMED drag coefficient in the same quasi-steady family


def wing_wrench(model, data, body_name: str) -> np.ndarray:
    """World-frame force/torque from 0.5 rho C A |v| v using the wing linear velocity."""
    bid = int(model.body(body_name).id)
    vel = np.asarray(data.cvel[bid], dtype=np.float64)  # [ω, v] in 6D
    lin = vel[3:6]
    speed = float(np.linalg.norm(lin))
    if speed < 1e-6:
        return np.zeros(6, dtype=np.float64)
    direction = lin / speed
    lift_dir = np.array([0.0, 0.0, 1.0])
    qdyn = 0.5 * RHO_AIR * speed * speed * WING_AREA
    force = -CD * qdyn * direction + CL * qdyn * lift_dir
    wrench = np.zeros(6, dtype=np.float64)
    wrench[:3] = force
    return wrench


def apply_aerodynamics(model, data) -> None:
    for name in ("wing_L", "wing_R"):
        try:
            bid = int(model.body(name).id)
        except KeyError:
            continue
        data.xfrc_applied[bid] = wing_wrench(model, data, name)
