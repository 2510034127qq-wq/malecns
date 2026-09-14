"""A morphologically detailed recipe must instantiate and advance simulation time."""

from __future__ import annotations

from pathlib import Path

import pytest

from malecns.parameters.physiology import load_physiology
from malecns.simulation.recipe import MaleCNSRecipe, debug_two_cell_bundle
from malecns.simulation.runner import run_network


def test_two_cell_recipe_preserves_explicit_synapse(tmp_path: Path) -> None:
    phys = load_physiology()
    bundle = debug_two_cell_bundle(tmp_path, phys)
    recipe = MaleCNSRecipe(bundle, phys)
    assert recipe.num_cells() == 2
    conns = recipe.connections_on(1)
    assert len(conns) == 1
    kinds = {recipe.cell_kind(g) for g in range(2)}
    assert len(kinds) == 1


@pytest.mark.gpu
def test_cable_network_advances_simulation_time(tmp_path: Path) -> None:
    report = run_network(t_final_ms=1.0, full=False, workdir=tmp_path)
    assert report.n_cells == 2
    assert report.t_final_ms >= 1.0
    assert report.advanced is True
    assert "max-extent" in report.cv_policy_name
    assert "10" in report.cv_policy_name
    assert report.execution_mode in {"gpu", "multicore"}
    assert report.wall_s >= 0.0
