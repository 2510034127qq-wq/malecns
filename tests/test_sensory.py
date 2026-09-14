"""Sensory encoding uses body-relative samples, not environment object coordinates."""

import numpy as np

from malecns.environment.scenes import load_scene
from malecns.sensory.olfaction import odor_concentration
from malecns.sensory.vision import compound_eye_sample, render_eyes


def test_odor_is_a_spatial_field_sampled_at_a_point() -> None:
    sources = [(0.01, 0.0, 0.001, 1.0)]
    near = odor_concentration(np.array([0.01, 0.0, 0.001]), sources, wind=np.zeros(3))
    far = odor_concentration(np.array([0.05, 0.0, 0.001]), sources, wind=np.zeros(3))
    assert near > far
    assert far >= 0.0


def test_eye_images_change_when_head_orientation_changes() -> None:
    import mujoco

    model, data = load_scene("phototaxis")
    left0, right0 = render_eyes(model, data)
    assert left0.shape[2] == 3
    adr = int(np.asarray(model.joint("head_yaw").qposadr).reshape(-1)[0])
    data.qpos[adr] = 0.6
    mujoco.mj_forward(model, data)
    left1, right1 = render_eyes(model, data)
    assert compound_eye_sample(left0).shape[0] > 1
    assert not np.allclose(left0, left1) or not np.allclose(right0, right1)
