"""Native MaleCNS coordinates must convert to micrometers at the ingestion boundary."""

from malecns.units import (
    MALE_CNS_UNITS_TO_UM,
    MALE_CNS_VOXEL_NM,
    assert_same_physical_space,
    native_to_um,
)


def test_published_voxel_size_is_eight_nanometers() -> None:
    assert MALE_CNS_VOXEL_NM == 8.0
    assert MALE_CNS_UNITS_TO_UM == 0.008


def test_native_xyz_and_radius_scale_identically() -> None:
    native = (86080.0, 25408.0, 35776.0, 96.0)
    um = native_to_um(native)
    assert um == (
        86080.0 * 0.008,
        25408.0 * 0.008,
        35776.0 * 0.008,
        96.0 * 0.008,
    )


def test_morphology_and_synapse_conversion_share_one_factor() -> None:
    morph = native_to_um((100.0, 50.0, 25.0))
    syn = native_to_um((100.0, 50.0, 25.0))
    assert_same_physical_space(morph, syn)
