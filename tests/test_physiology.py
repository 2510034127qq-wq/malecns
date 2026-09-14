"""Provenance-tagged physiology must not be presented as MaleCNS measurements."""

from malecns.parameters.physiology import ALLOWED_PROVENANCE, load_physiology


def test_physiology_yaml_tags_every_parameter() -> None:
    phys = load_physiology()
    assert phys.cv_max_extent_um.value == 10.0
    assert phys.cv_max_extent_um.provenance == "ASSUMED"
    assert phys.cm_F_per_m2.provenance == "LITERATURE_DERIVED"
    assert phys.spike_mechanism.value == "threshold_detector"
    for item in phys.syn_e_mV.values():
        assert item.provenance in ALLOWED_PROVENANCE
        assert item.source
    assert "not a measured" in phys.spike_mechanism.source.lower() or "assumption" in phys.spike_mechanism.source.lower()
