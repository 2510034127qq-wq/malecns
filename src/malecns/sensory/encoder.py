"""Map body-rooted sensory samples to Arbor event generators.

Visual/olfactory/contact/proprio signals never include object XY coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malecns.environment.scenes import odor_sources_for
from malecns.sensory.channels import IOPlan
from malecns.sensory.mechanosensation import contact_forces, named_contact_forces, proprioception
from malecns.sensory.olfaction import sample_antennas
from malecns.sensory.vision import (
    compound_eye_image,
    compound_eye_sample,
    render_eyes,
    sample_image_uv,
)

_KIND_CHANNEL = {0: None, 1: 2, 2: 1}  # R1-6 lum, R7 blue, R8 green (ASSUMED spectral)
_LEG_GEOMS = ("L1_tarsus", "L2_tarsus", "L3_tarsus", "R1_tarsus", "R2_tarsus", "R3_tarsus")


@dataclass
class SensoryPacket:
    left_image: np.ndarray
    right_image: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    left_compound: np.ndarray
    right_compound: np.ndarray
    odor_l: float
    odor_r: float
    contacts: np.ndarray
    proprio: dict[str, np.ndarray]
    events: dict[int, list[tuple[float, float]]]
    vision_mean: float
    mechano_mean: float
    proprio_mean: float
    dn_drive: float


def _emit(events: dict[int, list[tuple[float, float]]], gid: int, t_ms: float, weight: float) -> None:
    if weight <= 1e-8:
        return
    events[int(gid)] = [(float(t_ms), float(weight))]


def encode_sensors(
    model,
    data,
    *,
    experiment: str,
    t_ms: float,
    sensory_gids: dict[str, int] | None = None,
    io_plan: IOPlan | None = None,
    gain: float = 0.05,
) -> SensoryPacket:
    left_img, right_img = render_eyes(model, data)
    left = compound_eye_sample(left_img)
    right = compound_eye_sample(right_img)
    odor_l, odor_r = sample_antennas(model, data, odor_sources_for(experiment))
    contacts = contact_forces(model, data)
    named = named_contact_forces(model, data)
    proprio = proprioception(model, data)
    events: dict[int, list[tuple[float, float]]] = {}
    vision_mean = float(0.5 * (left.mean() + right.mean()))
    mechano_mean = float(contacts.sum()) if contacts.size else 0.0
    q = proprio["qpos"][7:] if proprio["qpos"].size > 7 else proprio["qpos"]
    proprio_mean = float(np.mean(np.abs(q))) if q.size else 0.0
    food_contact = 0.0
    for name, force in named.items():
        if "food" in name or "odor" in name:
            food_contact += force

    if io_plan is not None:
        lum_l = float(left.mean())
        lum_r = float(right.mean())
        for gid, u, v, kind in zip(
            io_plan.photo_l_gids, io_plan.photo_l_u, io_plan.photo_l_v, io_plan.photo_l_kind, strict=True
        ):
            ch = _KIND_CHANNEL.get(int(kind), None)
            val = sample_image_uv(left_img, float(u), float(v), channel=ch) if u >= 0 else lum_l
            _emit(events, int(gid), t_ms, gain * val)
        for gid, u, v, kind in zip(
            io_plan.photo_r_gids, io_plan.photo_r_u, io_plan.photo_r_v, io_plan.photo_r_kind, strict=True
        ):
            ch = _KIND_CHANNEL.get(int(kind), None)
            val = sample_image_uv(right_img, float(u), float(v), channel=ch) if u >= 0 else lum_r
            _emit(events, int(gid), t_ms, gain * val)
        for gid in io_plan.orn_l_gids.tolist():
            _emit(events, int(gid), t_ms, gain * odor_l)
        for gid in io_plan.orn_r_gids.tolist():
            _emit(events, int(gid), t_ms, gain * odor_r)
        for gid in io_plan.orn_mid_gids.tolist():
            _emit(events, int(gid), t_ms, gain * 0.5 * (odor_l + odor_r))
        leg_forces = [float(named.get(g, 0.0)) for g in _LEG_GEOMS]
        body_force = mechano_mean
        for gid, leg in zip(io_plan.mechano_gids.tolist(), io_plan.mechano_leg.tolist(), strict=True):
            val = leg_forces[int(leg)] if 0 <= int(leg) < 6 else body_force
            _emit(events, int(gid), t_ms, gain * val)
        qpos = np.asarray(data.qpos, dtype=np.float64)
        for gid, j in zip(io_plan.proprio_gids.tolist(), io_plan.proprio_joint.tolist(), strict=True):
            if 0 <= int(j) < qpos.size:
                val = abs(float(qpos[int(j)]))
            else:
                val = proprio_mean
            _emit(events, int(gid), t_ms, gain * val)
        for gid in io_plan.gustatory_gids.tolist():
            _emit(events, int(gid), t_ms, gain * food_contact)
    else:
        sensory_gids = sensory_gids or {}
        if "vision_l" in sensory_gids:
            _emit(events, sensory_gids["vision_l"], t_ms, gain * float(left.mean()))
        if "vision_r" in sensory_gids:
            _emit(events, sensory_gids["vision_r"], t_ms, gain * float(right.mean()))
        if "odor_l" in sensory_gids:
            _emit(events, sensory_gids["odor_l"], t_ms, gain * odor_l)
        if "odor_r" in sensory_gids:
            _emit(events, sensory_gids["odor_r"], t_ms, gain * odor_r)
        if "contact" in sensory_gids:
            _emit(events, sensory_gids["contact"], t_ms, gain * mechano_mean)
        if "proprio" in sensory_gids:
            _emit(events, sensory_gids["proprio"], t_ms, gain * proprio_mean)

    return SensoryPacket(
        left_image=left_img,
        right_image=right_img,
        left_eye=left,
        right_eye=right,
        left_compound=compound_eye_image(left),
        right_compound=compound_eye_image(right),
        odor_l=odor_l,
        odor_r=odor_r,
        contacts=contacts,
        proprio=proprio,
        events=events,
        vision_mean=vision_mean,
        mechano_mean=mechano_mean,
        proprio_mean=proprio_mean,
        dn_drive=0.0,
    )
