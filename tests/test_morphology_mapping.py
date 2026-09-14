"""SWC coordinates must convert to µm before Arbor mapping; synapses use the same space."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from malecns.morphology.convert import load_scaled_morphology
from malecns.synapses.mapping import map_points_to_cable, mapping_flags
from malecns.units import MALE_CNS_UNITS_TO_UM, native_to_um


def _write_swc(path: Path) -> None:
    # Straight cable along +x in native 8 nm units, length 1000 units = 8 µm.
    path.write_text("1 1 0 0 0 10 -1\n2 3 1000 0 0 10 1\n", encoding="utf-8")


def test_scaled_morphology_is_in_micrometers(tmp_path: Path) -> None:
    swc = tmp_path / "1.swc"
    _write_swc(swc)
    morph = load_scaled_morphology(swc)
    prox = morph.segment_prox[0]
    dist = morph.segment_dist[0]
    np.testing.assert_allclose(prox, [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(dist, [8.0, 0.0, 0.0], atol=1e-9)
    assert morph.num_branches >= 1


def test_synapse_midpoint_maps_to_branch_center(tmp_path: Path) -> None:
    swc = tmp_path / "1.swc"
    _write_swc(swc)
    morph = load_scaled_morphology(swc)
    native_xyz = np.array([[500.0, 0.0, 0.0]])
    xyz_um = native_to_um(native_xyz)
    mapped = map_points_to_cable(morph, xyz_um)
    assert mapped.branch[0] == 0
    assert abs(mapped.pos[0] - 0.5) < 1e-6
    assert mapped.residual_um[0] < 1e-6
    assert mapping_flags(mapped, residual_limit_um=10.0).pathological[0] == 0


def test_morphology_and_synapse_share_conversion(tmp_path: Path) -> None:
    swc = tmp_path / "1.swc"
    _write_swc(swc)
    morph = load_scaled_morphology(swc)
    native = np.array([250.0, 0.0, 0.0])
    syn_um = native * MALE_CNS_UNITS_TO_UM
    mapped = map_points_to_cable(morph, syn_um.reshape(1, 3))
    assert mapped.residual_um[0] < 1e-6


def test_far_point_is_flagged_pathological_not_dropped(tmp_path: Path) -> None:
    swc = tmp_path / "1.swc"
    _write_swc(swc)
    morph = load_scaled_morphology(swc)
    xyz = np.array([[4.0, 50.0, 0.0]])  # already µm, 50 µm off the cable
    mapped = map_points_to_cable(morph, xyz)
    flags = mapping_flags(mapped, residual_limit_um=10.0)
    assert flags.pathological[0] == 1
    assert flags.unmapped[0] == 0
    assert mapped.residual_um[0] > 10.0


def test_extra_swc_roots_are_kept_not_dropped(tmp_path: Path) -> None:
    swc = tmp_path / "1.swc"
    swc.write_text(
        "1 1 0 0 0 10 -1\n"
        "2 3 1000 0 0 10 1\n"
        "3 3 0 1000 0 10 -1\n"
        "4 3 0 2000 0 10 3\n",
        encoding="utf-8",
    )
    morph = load_scaled_morphology(swc)
    assert morph.num_branches >= 2
    assert morph.segment_prox.shape[0] >= 2
    assert any("extra SWC roots" in n for n in morph.notes)
