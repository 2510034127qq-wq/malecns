import mujoco
import pytest

from malecns.environment.scenes import SCENE_NAMES, load_scene


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_experiment_scenes_compile(name: str) -> None:
    model, data = load_scene(name)
    mujoco.mj_forward(model, data)
    assert model.nbody > 10
    assert "thorax" in {model.body(i).name for i in range(model.nbody)}
