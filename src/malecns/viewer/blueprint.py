"""Rerun Viewer blueprint: world, MaleCNS, eyes, telemetry on one timeline."""

from __future__ import annotations


def malecns_blueprint():
    from rerun.blueprint import (
        Blueprint,
        Horizontal,
        Spatial2DView,
        Spatial3DView,
        TimeSeriesView,
        Vertical,
    )

    return Blueprint(
        Vertical(
            Horizontal(
                Spatial3DView(name="MuJoCo world", origin="world"),
                Spatial3DView(name="MaleCNS", origin="cns"),
            ),
            Horizontal(
                Spatial2DView(name="Left eye", origin="eyes/left"),
                Spatial2DView(name="Right eye", origin="eyes/right"),
                TimeSeriesView(name="Telemetry", origin="telemetry"),
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=False,
    )
