"""Select annotated sensory and motor cells for I/O. The rest of the network stays in the recipe."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from malecns.data.catalog import BodyCatalog
from malecns.motor.mapping import build_motor_table


@dataclass
class IOPlan:
    """Every annotated photoreceptor, ORN, mechanosensory, proprio, MN, and DN gid."""

    photo_l_gids: NDArray[np.int32]
    photo_l_u: NDArray[np.float32]
    photo_l_v: NDArray[np.float32]
    photo_l_kind: NDArray[np.int8]  # 0=R1-6 lum, 1=R7 blue, 2=R8 green
    photo_r_gids: NDArray[np.int32]
    photo_r_u: NDArray[np.float32]
    photo_r_v: NDArray[np.float32]
    photo_r_kind: NDArray[np.int8]
    orn_l_gids: NDArray[np.int32]
    orn_r_gids: NDArray[np.int32]
    orn_mid_gids: NDArray[np.int32]
    mechano_gids: NDArray[np.int32]
    mechano_leg: NDArray[np.int8]  # -1 none, else index into L1,L2,L3,R1,R2,R3
    proprio_gids: NDArray[np.int32]
    proprio_joint: NDArray[np.int32]  # actuator/joint index, -1 = mean
    gustatory_gids: NDArray[np.int32]
    motor_gids: dict[int, tuple[int, float]]
    dn_gids: NDArray[np.int32]
    notes: list[str] = field(default_factory=list)

    def sensory_gid_set(self) -> set[int]:
        parts = [
            self.photo_l_gids,
            self.photo_r_gids,
            self.orn_l_gids,
            self.orn_r_gids,
            self.orn_mid_gids,
            self.mechano_gids,
            self.proprio_gids,
            self.gustatory_gids,
        ]
        out: set[int] = set()
        for arr in parts:
            out.update(int(x) for x in arr.tolist())
        return out


_LEG_ORDER = ("L1", "L2", "L3", "R1", "R2", "R3")


def _laterality(side: str | None, instance: str | None, root_side: str | None) -> str | None:
    inst = instance or ""
    if inst.endswith("_L") or inst.endswith("-L"):
        return "L"
    if inst.endswith("_R") or inst.endswith("-R"):
        return "R"
    s = (side or "").upper()
    if s in {"L", "R"}:
        return s
    r = (root_side or "").upper()
    if r in {"L", "R"}:
        return r
    return None


def _photo_kind(cell_type: str | None) -> int:
    t = (cell_type or "").upper()
    if t.startswith("R7"):
        return 1
    if t.startswith("R8"):
        return 2
    return 0


def _hex_uv(h1: float, h2: float) -> tuple[float, float]:
    if not np.isfinite(h1) or not np.isfinite(h2):
        return -1.0, -1.0
    u = float(np.clip((h1 - 1.0) / 35.0, 0.0, 1.0))
    v = float(np.clip((h2 - 1.0) / 38.0, 0.0, 1.0))
    return u, v


def _leg_index(side: str | None, neuromere: str | None) -> int:
    s = (side or "").upper()
    n = (neuromere or "").upper()
    if s not in {"L", "R"}:
        return -1
    seg = {"T1": "1", "T2": "2", "T3": "3"}.get(n)
    if seg is None:
        return -1
    name = f"{s}{seg}"
    return _LEG_ORDER.index(name)


def build_io_plan(catalog: BodyCatalog, model) -> IOPlan:
    photo_l: list[tuple[int, float, float, int]] = []
    photo_r: list[tuple[int, float, float, int]] = []
    orn_l: list[int] = []
    orn_r: list[int] = []
    orn_mid: list[int] = []
    mechano: list[tuple[int, int]] = []
    proprio: list[int] = []
    gustatory: list[int] = []
    dns: list[int] = []
    n = len(catalog)
    for gid in range(n):
        sc = catalog.superclass[gid] or ""
        cls = catalog.cell_class[gid] or ""
        sub = catalog.subclass[gid] or ""
        ty = catalog.cell_type[gid]
        side = _laterality(catalog.soma_side[gid], catalog.instance[gid], catalog.root_side[gid])
        neu = catalog.soma_neuromere[gid]
        if sc == "ol_sensory" and ty and ty.upper().startswith("R"):
            u, v = _hex_uv(float(catalog.assigned_ol_hex1[gid]), float(catalog.assigned_ol_hex2[gid]))
            kind = _photo_kind(ty)
            row = (gid, u, v, kind)
            if side == "R":
                photo_r.append(row)
            else:
                photo_l.append(row)
        if cls == "olfactory" or (ty or "").startswith("ORN_"):
            if side == "R":
                orn_r.append(gid)
            elif side == "L":
                orn_l.append(gid)
            else:
                orn_mid.append(gid)
        if cls in {"mechanosensory", "mechanosensory_tactile", "mechanosensory_tbc"} or "bristle" in sub.lower() or "campaniform" in sub.lower():
            if cls != "mechanosensory_proprioceptive":
                mechano.append((gid, _leg_index(side, neu)))
        if cls == "mechanosensory_proprioceptive" or "chordotonal" in sub.lower() or "hair plate" in sub.lower():
            proprio.append(gid)
        if cls in {"gustatory", "chemosensory"}:
            gustatory.append(gid)
        if sc in {"descending_neuron", "descending_neuron_tbc"}:
            dns.append(gid)

    motor = build_motor_table(catalog, model)
    n_act = int(model.nu)
    proprio_joint = np.full(len(proprio), -1, dtype=np.int32)
    if proprio and n_act:
        for i, gid in enumerate(proprio):
            proprio_joint[i] = i % n_act

    notes = [
        f"photoreceptors L={len(photo_l)} R={len(photo_r)} (all annotated ol_sensory R*)",
        f"ORNs L={len(orn_l)} R={len(orn_r)} mid={len(orn_mid)}",
        f"mechanosensory={len(mechano)} proprio={len(proprio)} gustatory={len(gustatory)}",
        f"motor map entries={len(motor)} descending={len(dns)}",
        "Hex1/Hex2 map photoreceptors onto the rendered eye image; missing hex uses eye mean.",
        "Motor mapping is ASSUMED from somaSide+somaNeuromere+type names; see motor.mapping.",
    ]

    def _split_photo(rows: list[tuple[int, float, float, int]]):
        if not rows:
            z = np.zeros(0, dtype=np.int32)
            zf = np.zeros(0, dtype=np.float32)
            zk = np.zeros(0, dtype=np.int8)
            return z, zf, zf.copy(), zk
        gids = np.array([r[0] for r in rows], dtype=np.int32)
        u = np.array([r[1] for r in rows], dtype=np.float32)
        v = np.array([r[2] for r in rows], dtype=np.float32)
        k = np.array([r[3] for r in rows], dtype=np.int8)
        return gids, u, v, k

    pl, plu, plv, plk = _split_photo(photo_l)
    pr, pru, prv, prk = _split_photo(photo_r)
    mg, ml = (np.array([], dtype=np.int32), np.array([], dtype=np.int8))
    if mechano:
        mg = np.array([a for a, _b in mechano], dtype=np.int32)
        ml = np.array([b for _a, b in mechano], dtype=np.int8)
    return IOPlan(
        photo_l_gids=pl,
        photo_l_u=plu,
        photo_l_v=plv,
        photo_l_kind=plk,
        photo_r_gids=pr,
        photo_r_u=pru,
        photo_r_v=prv,
        photo_r_kind=prk,
        orn_l_gids=np.asarray(orn_l, dtype=np.int32),
        orn_r_gids=np.asarray(orn_r, dtype=np.int32),
        orn_mid_gids=np.asarray(orn_mid, dtype=np.int32),
        mechano_gids=mg,
        mechano_leg=ml,
        proprio_gids=np.asarray(proprio, dtype=np.int32),
        proprio_joint=proprio_joint,
        gustatory_gids=np.asarray(gustatory, dtype=np.int32),
        motor_gids=motor,
        dn_gids=np.asarray(dns, dtype=np.int32),
        notes=notes,
    )


def select_channel_gids(catalog: BodyCatalog) -> tuple[dict[str, int], dict[int, str]]:
    """Backward-compatible summary (one gid per channel) plus all motor roles."""
    from malecns.motor.mapping import classify_motor_role

    vision_l = vision_r = odor_l = odor_r = None
    motor_roles: dict[int, str] = {}
    for gid in range(len(catalog)):
        sc = catalog.superclass[gid]
        ty = catalog.cell_type[gid]
        side = catalog.soma_side[gid]
        cls = catalog.cell_class[gid]
        if vision_l is None and sc == "ol_sensory" and (ty or "").startswith("R") and _laterality(side, catalog.instance[gid], catalog.root_side[gid]) == "L":
            vision_l = gid
        if vision_r is None and sc == "ol_sensory" and (ty or "").startswith("R") and _laterality(side, catalog.instance[gid], catalog.root_side[gid]) == "R":
            vision_r = gid
        if odor_l is None and cls == "olfactory" and _laterality(side, catalog.instance[gid], catalog.root_side[gid]) == "L":
            odor_l = gid
        if odor_r is None and cls == "olfactory" and _laterality(side, catalog.instance[gid], catalog.root_side[gid]) == "R":
            odor_r = gid
        role = classify_motor_role(sc, ty)
        if role:
            motor_roles[gid] = role
    sensory: dict[str, int] = {}
    if vision_l is not None:
        sensory["vision_l"] = vision_l
    if vision_r is not None:
        sensory["vision_r"] = vision_r
    if odor_l is not None:
        sensory["odor_l"] = odor_l
    if odor_r is not None:
        sensory["odor_r"] = odor_r
    return sensory, motor_roles
