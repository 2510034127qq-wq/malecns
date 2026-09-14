"""CNS LOD uses the same 8 nm → µm conversion as morphology."""

import numpy as np

from malecns.units import MALE_CNS_UNITS_TO_UM
from malecns.viewer.cns_lod import soma_cloud_um


def test_soma_cloud_shares_morphology_units() -> None:
    native = np.array([[1000.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    xyz, rgb = soma_cloud_um(native, ["vnc_motor", "ol_sensory"])
    assert xyz.shape == (1, 3)
    np.testing.assert_allclose(xyz[0, 0], 1000.0 * MALE_CNS_UNITS_TO_UM)
    assert rgb.shape == (1, 3)
    np.testing.assert_allclose(rgb[0], (0.2, 0.4, 0.95), atol=1e-5)
