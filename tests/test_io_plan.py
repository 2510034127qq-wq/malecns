"""Full I/O plan uses annotated photoreceptors/ORNs/MNs, not a handful of debug gids."""

from __future__ import annotations

import pytest

from malecns.environment.scenes import load_scene
from malecns.sensory.channels import build_io_plan


@pytest.mark.data
def test_io_plan_wires_annotated_photoreceptors_and_motors() -> None:
    from malecns.data.catalog import load_body_catalog

    catalog = load_body_catalog()
    model, _data = load_scene("sandbox")
    plan = build_io_plan(catalog, model)
    assert plan.photo_l_gids.size >= 2000
    assert plan.photo_r_gids.size >= 2000
    assert plan.orn_l_gids.size >= 500
    assert plan.orn_r_gids.size >= 500
    assert plan.mechano_gids.size >= 1000
    assert plan.proprio_gids.size >= 1000
    assert len(plan.motor_gids) >= 700
    assert plan.dn_gids.size >= 1000
    actuators = {spec[0] for spec in plan.motor_gids.values()}
    assert len(actuators) >= 8
