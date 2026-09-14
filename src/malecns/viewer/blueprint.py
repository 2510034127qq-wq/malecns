"""Rerun Viewer blueprint: world, MaleCNS, eyes, telemetry, inspect on one timeline."""

from __future__ import annotations


def malecns_blueprint():
    from rerun.blueprint import (
        Blueprint,
        Horizontal,
        Spatial2DView,
        Spatial3DView,
        TextDocumentView,
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
                Spatial2DView(name="Left compound", origin="eyes/compound_left"),
                Spatial2DView(name="Right compound", origin="eyes/compound_right"),
            ),
            Horizontal(
                TimeSeriesView(name="Telemetry", origin="telemetry"),
                TextDocumentView(name="Inspect", origin="inspect"),
            ),
            row_shares=[3, 2, 2],
        ),
        collapse_panels=False,
    )
