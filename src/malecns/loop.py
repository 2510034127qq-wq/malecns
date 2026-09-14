"""Closed sensory → MaleCNS → motor → MuJoCo → sensory loop."""

from __future__ import annotations

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
    execution_notes: list[str] = field(default_factory=list)
    wall_s: float = 0.0

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
        # Arbor python spikes: (time, (gid, lid)) or structured
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

    from malecns.simulation.context import make_execution_context
    from malecns.simulation.runner import _full_bundle

    phys = load_physiology()
    model, data = load_scene(experiment)
    work = Path(workdir) if workdir is not None else Path("/tmp/malecns-loop")
    work.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    if full:
        from malecns.data.catalog import load_body_catalog
        from malecns.sensory.channels import select_channel_gids

        bundle = _full_bundle(phys)
        catalog = load_body_catalog()
        sensory_gids, motor_roles = select_channel_gids(catalog)
        if not sensory_gids:
            sensory_gids = {"vision_l": 0, "vision_r": 1}
        motor_gids = {}
        role_to_joint = {
            "tibia_flexor": "L1_knee",
            "trochanter_flexor": "L1_pitch",
            "wing": "wing_L_stroke",
            "descending": "head_yaw",
            "leg": "L1_yaw",
        }
        claimed: set[str] = set()
        for gid, role in motor_roles.items():
            joint = role_to_joint.get(role)
            if joint and joint not in claimed:
                motor_gids[gid] = _actuator_index(model, joint)
                claimed.add(joint)
        if not motor_gids:
            motor_gids = {1: _actuator_index(model, "head_yaw")}
        notes.append("full MaleCNS recipe in the closed loop")
        notes.append(f"sensory_gids={sensory_gids} n_motor_roles={len(motor_roles)}")
    else:
        bundle = debug_two_cell_bundle(work, phys)
        sensory_gids = {"vision_l": 0, "vision_r": 1}
        motor_gids = {1: _actuator_index(model, "head_yaw")}
        notes.extend(bundle.notes)

    sensory_state: dict[int, list[tuple[float, float]]] = {}
    recipe = MaleCNSRecipe(bundle, phys, sensory_events=sensory_state)
    exec_ctx = make_execution_context(prefer_gpu=True)
    notes.extend(exec_ctx.notes)
    try:
        decomp = arbor.partition_load_balance(recipe, exec_ctx.context)
        sim = arbor.simulation(recipe, context=exec_ctx.context, domains=decomp)
    except Exception as exc:
        notes.append(f"GPU sim failed ({exc}); CPU fallback, model size unchanged")
        exec_ctx = make_execution_context(prefer_gpu=False)
        decomp = arbor.partition_load_balance(recipe, exec_ctx.context)
        sim = arbor.simulation(recipe, context=exec_ctx.context, domains=decomp)
    sim.record(arbor.spike_recording.all)

    viewer = Workbench(spawn=spawn_viewer)
    viewer.log_static_world(model)

    n_steps = int(max(t_final_ms / physics_dt_ms, 1))
    first_senses = None
    last_senses = None
    pos0 = _thorax_pos(model, data).copy()
    t_bio = 0.0
    wall0 = time.perf_counter()
    prev_spike_n = 0
    for step in range(n_steps):
        if viewer.control.paused:
            time.sleep(0.01)
            continue
        packet = encode_sensors(
            model,
            data,
            experiment=experiment,
            t_ms=t_bio,
            sensory_gids=sensory_gids,
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
        )
        t_bio = t_next

    pos1 = _thorax_pos(model, data)
    sensory_changed = first_senses is None or last_senses is None
    if first_senses is not None and last_senses is not None:
        sensory_changed = not np.allclose(first_senses, last_senses, atol=1e-8)
    return LoopReport(
        experiment=experiment,
        n_steps=n_steps,
        t_final_ms=float(t_bio),
        sensory_changed=bool(sensory_changed),
        body_moved=bool(np.linalg.norm(pos1 - pos0) > 1e-9 or np.any(np.abs(data.qvel) > 1e-8)),
        used_scripted_behavior=False,
        n_cells=int(recipe.num_cells()),
        execution_notes=notes,
        wall_s=time.perf_counter() - wall0,
    )
