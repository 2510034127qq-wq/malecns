"""Closed sensory → MaleCNS → motor → MuJoCo → sensory loop."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from malecns.environment.scenes import load_scene
from malecns.motor.aerodynamics import apply_aerodynamics
from malecns.motor.mapping import motor_from_spikes
from malecns.parameters.physiology import load_physiology
from malecns.sensory.encoder import encode_sensors
from malecns.simulation.recipe import MaleCNSRecipe, debug_two_cell_bundle
from malecns.units import MALE_CNS_UNITS_TO_UM
from malecns.viewer.logger import Workbench


@dataclass
class LoopReport:
    experiment: str
    n_steps: int
    t_final_ms: float
    sensory_changed: bool
    body_moved: bool
    used_scripted_behavior: bool
    n_cells: int
    n_connections: int = 0
    execution_mode: str = ""
    rss_gb: float = 0.0
    vram_gb: float | None = None
    construction_s: float = 0.0
    execution_notes: list[str] = field(default_factory=list)
    wall_s: float = 0.0
    accounting: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _actuator_index(model, name: str) -> int:
    return int(model.actuator(name).id)


def _thorax_pos(model, data) -> np.ndarray:
    return np.array(data.xpos[int(model.body("thorax").id)], dtype=np.float64)


def _spike_rates(spikes, t0_ms: float, t1_ms: float) -> dict[int, float]:
    rates: dict[int, float] = {}
    if spikes is None:
        return rates
    dt = max(t1_ms - t0_ms, 1e-6)
    for item in spikes:
        try:
            time_ms = float(item[0])
            gid = int(item[1][0]) if not np.isscalar(item[1]) else int(item[1])
        except Exception:
            continue
        if t0_ms <= time_ms * 1000.0 <= t1_ms or t0_ms <= time_ms <= t1_ms:
            rates[gid] = rates.get(gid, 0.0) + 1.0
    for gid in list(rates):
        rates[gid] = rates[gid] / dt * 1000.0
    return rates


def _gpu_related(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("cuda", "gpu", "out of memory", "oom", "device"))


def _dump_motor_map(path: Path, catalog, motor_gids: dict, model) -> None:
    names = [str(model.actuator(i).name) for i in range(model.nu)]
    rows = []
    for gid, spec in motor_gids.items():
        idx, sign = spec if isinstance(spec, tuple) else (int(spec), 1.0)
        rows.append(
            {
                "gid": int(gid),
                "body_id": int(catalog.body_ids[int(gid)]),
                "type": catalog.cell_type[int(gid)],
                "superclass": catalog.superclass[int(gid)],
                "soma_side": catalog.soma_side[int(gid)],
                "soma_neuromere": catalog.soma_neuromere[int(gid)],
                "actuator": names[int(idx)] if 0 <= int(idx) < len(names) else str(idx),
                "sign": float(sign),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n": len(rows), "map": rows}, indent=2), encoding="utf-8")


def run_closed_loop(
    *,
    experiment: str = "sandbox",
    t_final_ms: float = 100.0,
    spawn_viewer: bool = True,
    full: bool = False,
    workdir: Path | str | None = None,
    physics_dt_ms: float = 0.2,
) -> LoopReport:
    import arbor
    import mujoco

    from malecns.paths import DATA_CACHE
    from malecns.simulation.context import make_execution_context
    from malecns.simulation.resources import gpu_memory_gb, rss_gb
    from malecns.simulation.runner import _full_bundle

    phys = load_physiology()
    model, data = load_scene(experiment)
    work = Path(workdir) if workdir is not None else Path("/tmp/malecns-loop")
    work.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    io_plan = None
    catalog = None
    sensory_gids: dict[str, int] = {}
    motor_gids: dict = {}
    if full:
        from malecns.data.catalog import load_body_catalog
        from malecns.sensory.channels import build_io_plan

        bundle = _full_bundle(phys)
        catalog = load_body_catalog()
        io_plan = build_io_plan(catalog, model)
        motor_gids = io_plan.motor_gids
        _dump_motor_map(DATA_CACHE / "motor_map.json", catalog, motor_gids, model)
        notes.append("full MaleCNS recipe in the closed loop")
        notes.extend(io_plan.notes)
        notes.append(f"wrote {DATA_CACHE / 'motor_map.json'}")
    else:
        bundle = debug_two_cell_bundle(work, phys)
        sensory_gids = {"vision_l": 0, "vision_r": 1}
        motor_gids = {1: _actuator_index(model, "head_yaw")}
        notes.extend(bundle.notes)

    sensory_state: dict[int, list[tuple[float, float]]] = {}
    recipe = MaleCNSRecipe(bundle, phys, sensory_events=sensory_state)
    exec_ctx = make_execution_context(prefer_gpu=True)
    notes.extend(exec_ctx.notes)
    t_build = time.perf_counter()
    try:
        decomp = arbor.partition_load_balance(recipe, exec_ctx.context)
        sim = arbor.simulation(recipe, context=exec_ctx.context, domains=decomp)
        mode = exec_ctx.mode
    except Exception as exc:
        if exec_ctx.mode != "gpu" or not _gpu_related(exc):
            raise
        notes.append(
            f"GPU simulation construction failed ({exc}); retrying multicore CPU. "
            "Cell/synapse counts are unchanged."
        )
        exec_ctx = make_execution_context(prefer_gpu=False)
        notes.extend(exec_ctx.notes)
        decomp = arbor.partition_load_balance(recipe, exec_ctx.context)
        sim = arbor.simulation(recipe, context=exec_ctx.context, domains=decomp)
        mode = "multicore"
    construction_s = time.perf_counter() - t_build
    sim.record(arbor.spike_recording.all)
    notes.append(f"construction_s={construction_s:.1f} accounting={recipe.accounting}")

    viewer = Workbench(spawn=spawn_viewer)
    viewer.log_static_world(model)
    soma_xyz = None
    if full and catalog is not None:
        try:
            from malecns.viewer.cns_lod import (
                downsample_line_strips,
                morphology_line_strips,
                soma_cloud_um,
            )

            xyz, rgb = soma_cloud_um(catalog.soma_xyz_native, catalog.superclass)
            viewer.log_cns_somas(xyz, rgb)
            soma_xyz = catalog.soma_xyz_native * MALE_CNS_UNITS_TO_UM
            notes.append(f"logged {len(xyz)} soma points in µm to Rerun")
            from malecns.data.skeletons import build_skeleton_index, swc_path_for
            from malecns.morphology.convert import load_scaled_morphology

            skeletons = build_skeleton_index(cache=True)
            lod_strips = []
            n_lod = 0
            for gid in io_plan.dn_gids.tolist()[:48]:
                body_id = int(catalog.body_ids[int(gid)])
                try:
                    path = swc_path_for(skeletons, body_id)
                    morph = load_scaled_morphology(path)
                    lod_strips.append(downsample_line_strips(morphology_line_strips(morph), 800))
                    n_lod += 1
                except Exception:
                    continue
            if lod_strips:
                viewer.log_cns_morphology_lod(np.concatenate(lod_strips, axis=0))
                notes.append(f"logged {n_lod} DN morphologies as CNS line LOD")
        except Exception as exc:
            notes.append(f"CNS soma/morph LOD skipped: {exc}")

    n_steps = 0
    first_senses = None
    last_senses = None
    pos0 = _thorax_pos(model, data).copy()
    t_bio = 0.0
    wall0 = time.perf_counter()
    prev_spike_n = 0
    inspect_id = None
    while t_bio < t_final_ms - 1e-12:
        ctl = viewer.poll_control()
        if ctl.paused:
            time.sleep(0.02)
            continue
        if ctl.speed < 1.0:
            time.sleep(max(0.0, (1.0 - ctl.speed) * physics_dt_ms / 1000.0))
        packet = encode_sensors(
            model,
            data,
            experiment=experiment,
            t_ms=t_bio,
            sensory_gids=sensory_gids,
            io_plan=io_plan,
        )
        vec = np.concatenate(
            [packet.left_eye, packet.right_eye, [packet.odor_l, packet.odor_r]]
        )
        if first_senses is None:
            first_senses = vec.copy()
        last_senses = vec
        sensory_state.clear()
        sensory_state.update(packet.events)
        sim.update(recipe)
        t_next = t_bio + physics_dt_ms
        sim.run(t_next * arbor.units.ms, 0.025 * arbor.units.ms)
        spikes = sim.spikes()
        n_spikes = len(spikes) if spikes is not None else 0
        new_n = n_spikes - prev_spike_n
        prev_spike_n = n_spikes
        rates = _spike_rates(spikes, t_bio, t_next)
        ctrl = motor_from_spikes(rates, n_actuator=model.nu, motor_gids=motor_gids)
        data.ctrl[:] = ctrl
        apply_aerodynamics(model, data)
        mujoco.mj_step(model, data)
        wall = time.perf_counter() - wall0
        ratio = (t_next / 1000.0) / max(wall, 1e-9)
        dn_drive = 0.0
        if io_plan is not None and io_plan.dn_gids.size:
            dn_drive = float(sum(rates.get(int(g), 0.0) for g in io_plan.dn_gids[:64]))
        packet.dn_drive = dn_drive
        act_xyz = act_rgb = None
        if soma_xyz is not None and rates:
            gids = np.fromiter(rates.keys(), dtype=np.int64)
            gids = gids[(gids >= 0) & (gids < soma_xyz.shape[0])]
            if gids.size:
                pts = soma_xyz[gids]
                ok = np.isfinite(pts).all(axis=1)
                act_xyz = pts[ok].astype(np.float32)
                act_rgb = np.tile(np.array([1.0, 0.15, 0.1], dtype=np.float32), (int(ok.sum()), 1))
        wing_l = float(ctrl[_actuator_index(model, "wing_L_stroke")]) if model.nu else 0.0
        wing_r = float(ctrl[_actuator_index(model, "wing_R_stroke")]) if model.nu else 0.0
        speed = float(np.linalg.norm(data.cvel[int(model.body("thorax").id)][3:6]))
        viewer.log_step(
            t_ms=t_next,
            thorax_pos=_thorax_pos(model, data),
            left_image=packet.left_image,
            right_image=packet.right_image,
            odor_l=packet.odor_l,
            odor_r=packet.odor_r,
            ctrl=ctrl,
            n_spikes=new_n,
            realtime_ratio=ratio,
            left_compound=packet.left_compound,
            right_compound=packet.right_compound,
            vision_mean=packet.vision_mean,
            mechano_mean=packet.mechano_mean,
            proprio_mean=packet.proprio_mean,
            dn_drive=dn_drive,
            wing_l=wing_l,
            wing_r=wing_r,
            body_speed=speed,
            model=model,
            data=data,
            activity_xyz=act_xyz,
            activity_rgb=act_rgb,
        )
        want = ctl.inspect_body_id
        if want is not None and want != inspect_id and catalog is not None:
            try:
                from malecns.inspect import inspect_body

                report = inspect_body(int(want))
                viewer.log_inspect(
                    body_id=int(want),
                    markdown=report.to_markdown(),
                    strips_um=report.__dict__.get("_strips"),
                    synapse_xyz_um=report.__dict__.get("_syn_xyz"),
                )
                inspect_id = int(want)
                notes.append(f"inspected body {want}")
            except Exception as exc:
                notes.append(f"inspect {want} failed: {exc}")
        t_bio = t_next
        n_steps += 1

    pos1 = _thorax_pos(model, data)
    sensory_changed = first_senses is None or last_senses is None
    if first_senses is not None and last_senses is not None:
        sensory_changed = not np.allclose(first_senses, last_senses, atol=1e-8)
    acc = dict(recipe.accounting)
    n_conn = int(acc.get("n_explicit_synapses", 0))
    return LoopReport(
        experiment=experiment,
        n_steps=n_steps,
        t_final_ms=float(t_bio),
        sensory_changed=bool(sensory_changed),
        body_moved=bool(np.linalg.norm(pos1 - pos0) > 1e-9 or np.any(np.abs(data.qvel) > 1e-8)),
        used_scripted_behavior=False,
        n_cells=int(recipe.num_cells()),
        n_connections=n_conn,
        execution_mode=mode,
        rss_gb=rss_gb(),
        vram_gb=gpu_memory_gb(),
        construction_s=construction_s,
        execution_notes=notes,
        wall_s=time.perf_counter() - wall0,
        accounting=acc,
    )
