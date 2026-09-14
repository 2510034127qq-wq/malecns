"""Project-level acceptance report against GOAL.md / AGENTS.md criteria."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from malecns.paths import LOGS_DIR


@dataclass
class AcceptanceReport:
    env: dict[str, Any]
    manifest: dict[str, Any]
    mapping: dict[str, Any]
    simulate: dict[str, Any]
    loop: dict[str, Any]
    dropped: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_acceptance(
    *,
    t_final_ms: float = 0.25,
    loop_ms: float = 1.0,
    do_map: bool = True,
    do_simulate: bool = True,
    do_loop: bool = True,
    spawn_viewer: bool = False,
) -> AcceptanceReport:
    from malecns.data.manifest import build_dataset_manifest
    from malecns.envcheck import validate_environment
    from malecns.loop import run_closed_loop
    from malecns.simulation.runner import run_network
    from malecns.synapses.mapping import load_mapping_stats, map_synapses

    t0 = time.perf_counter()
    notes: list[str] = []
    env = validate_environment(require_gpu=True).to_dict()
    manifest = build_dataset_manifest(full=True).to_dict()
    mapping: dict[str, Any] = {}
    if do_map:
        cached = load_mapping_stats()
        if cached is not None and cached.n_mapped > 0:
            mapping = cached.to_dict()
            notes.append("reused data/cache/mapping_stats.json")
        else:
            mapping = map_synapses(full=True, resume=True).to_dict()
    simulate: dict[str, Any] = {}
    if do_simulate:
        simulate = run_network(t_final_ms=float(t_final_ms), full=True).to_dict()
    loop: dict[str, Any] = {}
    if do_loop:
        loop = run_closed_loop(
            experiment="sandbox",
            t_final_ms=float(loop_ms),
            spawn_viewer=spawn_viewer,
            full=True,
        ).to_dict()

    dropped = {
        "partner_post_not_in_annotations": (mapping.get("n_unmapped") if mapping else None),
        "simulate_unmapped_sites": simulate.get("n_unmapped"),
        "simulate_stub_morphologies": simulate.get("n_stub"),
        "simulate_pathological": simulate.get("n_pathological"),
        "note": "Unmapped/pathological/stub records are counted and kept in the model; they are not silently dropped.",
    }
    report = AcceptanceReport(
        env=env,
        manifest=manifest,
        mapping=mapping,
        simulate=simulate,
        loop=loop,
        dropped=dropped,
        notes=notes,
        wall_s=time.perf_counter() - t0,
    )
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / "acceptance.json"
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    notes.append(f"wrote {out}")
    report.notes = notes
    out.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return report
