"""Select annotated sensory and motor cells for I/O. The rest of the network stays in the recipe."""

from __future__ import annotations

from malecns.data.catalog import BodyCatalog
from malecns.motor.mapping import classify_motor_role


def select_channel_gids(catalog: BodyCatalog) -> tuple[dict[str, int], dict[int, str]]:
    vision_l = vision_r = odor_l = odor_r = None
    motor_roles: dict[int, str] = {}
    for gid in range(len(catalog)):
        sc = catalog.superclass[gid]
        ty = catalog.cell_type[gid]
        side = catalog.soma_side[gid]
        cls = catalog.cell_class[gid]
        if vision_l is None and sc == "ol_sensory" and (ty or "").startswith("R") and side == "L":
            vision_l = gid
        if vision_r is None and sc == "ol_sensory" and (ty or "").startswith("R") and side == "R":
            vision_r = gid
        if odor_l is None and cls == "olfactory" and side == "L":
            odor_l = gid
        if odor_r is None and cls == "olfactory" and side == "R":
            odor_r = gid
        role = classify_motor_role(sc, ty)
        if role:
            motor_roles[gid] = role
        if vision_l is not None and vision_r is not None and odor_l is not None and odor_r is not None:
            if len(motor_roles) > 32:
                break
    sensory = {}
    if vision_l is not None:
        sensory["vision_l"] = vision_l
    if vision_r is not None:
        sensory["vision_r"] = vision_r
    if odor_l is not None:
        sensory["odor_l"] = odor_l
    if odor_r is not None:
        sensory["odor_r"] = odor_r
    return sensory, motor_roles
