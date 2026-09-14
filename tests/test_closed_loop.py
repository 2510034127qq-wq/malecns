"""Behavior must come from CNS motor output through the body, not scripted reflexes."""

from pathlib import Path

import numpy as np

from malecns.loop import run_closed_loop
from malecns.motor.mapping import motor_from_spikes


def test_motor_mapping_ignores_wall_flags() -> None:
    spikes = {1: 12.0}
    ctrl_a = motor_from_spikes(spikes, n_actuator=24, motor_gids={1: 0})
    ctrl_b = motor_from_spikes(spikes, n_actuator=24, motor_gids={1: 0}, wall_near=True)
    np.testing.assert_array_equal(ctrl_a, ctrl_b)
    assert ctrl_a[0] != 0.0


def test_motor_mapping_sums_onto_shared_actuator() -> None:
    one = motor_from_spikes({1: 10.0}, n_actuator=4, motor_gids={1: 0})
    two = motor_from_spikes({1: 10.0, 2: 10.0}, n_actuator=4, motor_gids={1: 0, 2: (0, 1.0)})
    assert two[0] > one[0]


def test_closed_loop_updates_senses_after_body_step(tmp_path: Path) -> None:
    report = run_closed_loop(
        experiment="sandbox",
        t_final_ms=2.0,
        spawn_viewer=False,
        full=False,
        workdir=tmp_path,
    )
    assert report.n_steps >= 1
    assert report.sensory_changed is True
    assert report.used_scripted_behavior is False
    assert report.body_moved is True
