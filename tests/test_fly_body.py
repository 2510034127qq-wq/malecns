"""The MuJoCo fly has a head, thorax, abdomen, six legs, and two wings."""

import mujoco

from malecns.body.fly import FLY_BODY_NAMES, load_fly_model


def test_fly_model_has_legs_wings_and_head() -> None:
    model, data = load_fly_model()
    names = {model.body(i).name for i in range(model.nbody)}
    for required in FLY_BODY_NAMES:
        assert required in names, required
    assert model.body("head").id != model.body("thorax").id
    assert model.nu >= 20  # hinge actuators for legs, head, wings
    mujoco.mj_forward(model, data)
    assert data.qpos.size > 7  # freejoint plus articulated joints
