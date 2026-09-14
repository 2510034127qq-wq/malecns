"""Full local MaleCNS files must convert and become Arbor cable cells."""

from pathlib import Path

import numpy as np
import pytest

from malecns.morphology.convert import load_scaled_morphology
from malecns.parameters.physiology import load_physiology
from malecns.paths import SKELETONS_DIR, table_path
from malecns.synapses.mapping import map_points_to_cable
from malecns.units import MALE_CNS_UNITS_TO_UM


@pytest.mark.data
def test_native_swc_becomes_scaled_cable_cell() -> None:
    swc = SKELETONS_DIR / "10001.swc"
    if not swc.is_file():
        pytest.skip("native SWC 10001.swc not present")
    morph = load_scaled_morphology(swc)
    assert morph.num_branches >= 1
    assert morph.segment_prox.shape[0] >= 1
    # First sample of 10001 should map with ~0 residual after conversion.
    native = np.array([[37124.0, 22258.0, 36274.0]])  # DNp01 somaLocation from annotations
    mapped = map_points_to_cable(morph, native * MALE_CNS_UNITS_TO_UM)
    assert mapped.residual_um[0] < 50.0  # soma may sit off the coarse skeleton; still same space
    phys = load_physiology()
    from malecns.simulation.recipe import MaleCNSRecipe, NetworkBundle

    bundle = NetworkBundle(
        n_cells=1,
        body_ids=np.array([10001], dtype=np.int64),
        gid_of={10001: 0},
        swc_for_gid=lambda gid: swc,
        consensus_nt=["acetylcholine"],
        incidence=None,
        residual_limit_um=float(phys.residual_limit_um.value),
        notes=["single published neuron, not a reduced connectome"],
    )
    recipe = MaleCNSRecipe(bundle, phys)
    cell = recipe.cell_description(0)
    assert cell is not None
    assert Path(table_path("syn_partners")).is_file()
