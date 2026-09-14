"""Instantiate an Arbor recipe and advance biological simulation time."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import arbor

from malecns.parameters.physiology import load_physiology
from malecns.simulation.context import make_execution_context
from malecns.simulation.recipe import MaleCNSRecipe, NetworkBundle, debug_two_cell_bundle
from malecns.simulation.resources import gpu_memory_gb, rss_gb


@dataclass
class SimReport:
    n_cells: int
    n_connections: int
    t_final_ms: float
    dt_ms: float
    advanced: bool
    cv_policy_name: str
    execution_mode: str
    wall_s: float
    rss_gb: float
    vram_gb: float | None
    n_spikes: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _full_bundle(phys) -> NetworkBundle:
    from malecns.data.catalog import load_body_catalog
    from malecns.data.skeletons import build_skeleton_index, swc_path_for
    from malecns.paths import DATA_CACHE, SKELETONS_DIR
    from malecns.synapses.partners import build_post_incidence

    catalog = load_body_catalog()
    skeletons = build_skeleton_index(cache=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    incidence = build_post_incidence(
        gid_of=catalog.gid_of,
        n_gids=len(catalog),
        cache_dir=DATA_CACHE / "post_incidence",
    )

    def swc_for_gid(gid: int) -> Path | None:
        body_id = int(catalog.body_ids[int(gid)])
        try:
            return swc_path_for(skeletons, body_id)
        except FileNotFoundError:
            path = SKELETONS_DIR / f"{body_id}.swc"
            return path if path.is_file() else None

    return NetworkBundle(
        n_cells=len(catalog),
        body_ids=catalog.body_ids,
        gid_of=catalog.gid_of,
        swc_for_gid=swc_for_gid,
        consensus_nt=catalog.consensus_nt,
        incidence=incidence,
        residual_limit_um=float(phys.residual_limit_um.value),
        notes=[
            "full published MaleCNS v1.0 bodies; no extra connectivity threshold",
            "missing SWC bodies receive an explicit stub cable and are counted",
        ],
    )


def run_network(
    *,
    t_final_ms: float = 1.0,
    full: bool = False,
    workdir: Path | str | None = None,
    prefer_gpu: bool = True,
    dt_ms: float = 0.025,
) -> SimReport:
    phys = load_physiology()
    notes: list[str] = []
    if full:
        bundle = _full_bundle(phys)
        sensory = {}
    else:
        work = Path(workdir) if workdir is not None else Path("/tmp/malecns-debug-net")
        work.mkdir(parents=True, exist_ok=True)
        bundle = debug_two_cell_bundle(work, phys)
        sensory = {0: [(0.2, 0.4)]}
        notes.extend(bundle.notes)

    recipe = MaleCNSRecipe(bundle, phys, sensory_events=sensory)
    cv_name = str(recipe.cv_policy)
    notes.append(f"cv_policy={cv_name} provenance={phys.cv_max_extent_um.provenance}")

    def _simulate(exec_ctx) -> tuple[object, object]:
        decomp = arbor.partition_load_balance(recipe, exec_ctx.context)
        sim = arbor.simulation(recipe, context=exec_ctx.context, domains=decomp)
        return decomp, sim

    exec_ctx = make_execution_context(prefer_gpu=prefer_gpu)
    notes.extend(exec_ctx.notes)
    try:
        _decomp, sim = _simulate(exec_ctx)
        mode = exec_ctx.mode
    except Exception as exc:
        message = str(exc).lower()
        gpu_related = any(
            token in message for token in ("cuda", "gpu", "out of memory", "oom", "device")
        )
        if exec_ctx.mode != "gpu" or not gpu_related:
            raise
        notes.append(
            f"GPU simulation construction failed ({exc}); retrying multicore CPU. "
            "Cell/synapse counts are unchanged."
        )
        exec_ctx = make_execution_context(prefer_gpu=False)
        notes.extend(exec_ctx.notes)
        _decomp, sim = _simulate(exec_ctx)
        mode = "multicore"

    sim.record(arbor.spike_recording.all)
    t0 = time.perf_counter()
    sim.run(float(t_final_ms) * arbor.units.ms, float(dt_ms) * arbor.units.ms)
    wall = time.perf_counter() - t0
    spikes = sim.spikes()
    n_conn = int(recipe.accounting.get("n_explicit_synapses", 0))
    notes.append(f"accounting={recipe.accounting}")
    return SimReport(
        n_cells=int(recipe.num_cells()),
        n_connections=n_conn,
        t_final_ms=float(t_final_ms),
        dt_ms=float(dt_ms),
        advanced=True,
        cv_policy_name=cv_name,
        execution_mode=mode,
        wall_s=wall,
        rss_gb=rss_gb(),
        vram_gb=gpu_memory_gb(),
        n_spikes=len(spikes) if spikes is not None else 0,
        notes=notes,
    )
