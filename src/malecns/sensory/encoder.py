"""Map body-rooted sensory samples to Arbor event generators.

Visual/olfactory/contact/proprio signals never include object XY coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malecns.environment.scenes import odor_sources_for
from malecns.sensory.mechanosensation import contact_forces, proprioception
from malecns.sensory.olfaction import sample_antennas
from malecns.sensory.vision import compound_eye_sample, render_eyes


@dataclass
class SensoryPacket:
    left_image: np.ndarray
    right_image: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    odor_l: float
    odor_r: float
    contacts: np.ndarray
    proprio: dict[str, np.ndarray]
    events: dict[int, list[tuple[float, float]]]


def encode_sensors(
    model,
    data,
    *,
    experiment: str,
    t_ms: float,
    sensory_gids: dict[str, int],
    gain: float = 0.05,
) -> SensoryPacket:
    left_img, right_img = render_eyes(model, data)
    left = compound_eye_sample(left_img)
    right = compound_eye_sample(right_img)
    odor_l, odor_r = sample_antennas(model, data, odor_sources_for(experiment))
    contacts = contact_forces(model, data)
    proprio = proprioception(model, data)
    events: dict[int, list[tuple[float, float]]] = {}
    if "vision_l" in sensory_gids:
        events[sensory_gids["vision_l"]] = [(t_ms, float(gain * left.mean()))]
    if "vision_r" in sensory_gids:
        events[sensory_gids["vision_r"]] = [(t_ms, float(gain * right.mean()))]
    if "odor_l" in sensory_gids:
        events[sensory_gids["odor_l"]] = [(t_ms, float(gain * odor_l))]
    if "odor_r" in sensory_gids:
        events[sensory_gids["odor_r"]] = [(t_ms, float(gain * odor_r))]
    if "contact" in sensory_gids:
        events[sensory_gids["contact"]] = [(t_ms, float(gain * contacts.sum()))]
    if "proprio" in sensory_gids:
        q = proprio["qpos"][7:] if proprio["qpos"].size > 7 else proprio["qpos"]
        events[sensory_gids["proprio"]] = [(t_ms, float(gain * np.mean(np.abs(q))))]
    return SensoryPacket(
        left_image=left_img,
        right_image=right_img,
        left_eye=left,
        right_eye=right,
        odor_l=odor_l,
        odor_r=odor_r,
        contacts=contacts,
        proprio=proprio,
        events=events,
    )
