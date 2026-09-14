"""Experiment scenes provide stimuli only; they never encode a correct response."""

from __future__ import annotations

import mujoco

from malecns.body.fly import build_fly_xml

SCENE_NAMES = ("sandbox", "phototaxis", "ymaze", "obstacle", "looming")


def _inject_worldbody(fly_xml: str, extra: str) -> str:
    marker = "<body name=\"thorax\""
    if marker not in fly_xml:
        raise ValueError("fly xml missing thorax")
    return fly_xml.replace(marker, extra + "\n    " + marker, 1)


def scene_worldbody(name: str) -> str:
    if name == "sandbox":
        return """
    <geom name="wall_n" type="box" pos="0 0.04 0.005" size="0.04 0.0004 0.005" rgba="0.6 0.55 0.5 1"/>
    <geom name="wall_s" type="box" pos="0 -0.04 0.005" size="0.04 0.0004 0.005" rgba="0.6 0.55 0.5 1"/>
    <geom name="food" type="sphere" pos="0.015 0.0 0.0008" size="0.0008" rgba="0.9 0.2 0.1 1"/>
"""
    if name == "phototaxis":
        return """
    <light name="bright_x" directional="false" pos="0.03 0 0.02" dir="0 0 -1" diffuse="1.5 1.5 1.2" specular="0.4 0.4 0.3"/>
    <geom name="dark_wall" type="box" pos="-0.02 0 0.006" size="0.0005 0.02 0.006" rgba="0.05 0.05 0.08 1"/>
    <geom name="bright_panel" type="box" pos="0.025 0 0.006" size="0.0004 0.015 0.006" rgba="1 1 0.85 1"/>
"""
    if name == "ymaze":
        return """
    <geom name="stem" type="box" pos="0 0 0.003" size="0.0012 0.012 0.003" rgba="0.55 0.5 0.45 1"/>
    <geom name="arm_l" type="box" pos="0.012 0.012 0.003" size="0.010 0.0012 0.003" euler="0 0 40" rgba="0.55 0.5 0.45 1"/>
    <geom name="arm_r" type="box" pos="0.012 -0.012 0.003" size="0.010 0.0012 0.003" euler="0 0 -40" rgba="0.55 0.5 0.45 1"/>
    <geom name="odor_l" type="sphere" pos="0.022 0.018 0.001" size="0.0007" rgba="0.2 0.8 0.3 1"/>
    <geom name="odor_r" type="sphere" pos="0.022 -0.018 0.001" size="0.0007" rgba="0.8 0.2 0.8 1"/>
"""
    if name == "obstacle":
        return """
    <geom name="block_a" type="box" pos="0.008 0.002 0.002" size="0.0015 0.003 0.002" rgba="0.5 0.4 0.3 1"/>
    <geom name="block_b" type="box" pos="0.014 -0.003 0.002" size="0.0015 0.004 0.002" rgba="0.45 0.35 0.25 1"/>
"""
    if name == "looming":
        return """
    <body name="loom_disk" pos="0.02 0 0.008">
      <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-12 1e-12 1e-12"/>
      <joint name="loom_slide" type="slide" axis="1 0 0" range="-0.02 0.02"/>
      <geom name="loom_g" type="sphere" size="0.003" rgba="0.1 0.1 0.1 1"/>
    </body>
"""
    raise ValueError(f"unknown scene {name}")


def odor_sources_for(name: str) -> list[tuple[float, float, float, float]]:
    """World-frame odor sources (x, y, z, strength). CNS never receives these coordinates."""
    if name == "sandbox":
        return [(0.015, 0.0, 0.0008, 1.0)]
    if name == "ymaze":
        return [(0.022, 0.018, 0.001, 1.0), (0.022, -0.018, 0.001, 0.6)]
    return []


def load_scene(name: str = "sandbox") -> tuple[mujoco.MjModel, mujoco.MjData]:
    if name not in SCENE_NAMES:
        raise ValueError(f"unknown experiment {name}")
    xml = _inject_worldbody(build_fly_xml(), scene_worldbody(name))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
