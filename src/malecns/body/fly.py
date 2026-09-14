"""Drosophila melanogaster rigid-body model in MuJoCo (SI units).

Segment lengths and mass are literature-scale, not MaleCNS measurements.
Sources: fly body mass ~1 mg (Lehmann & Dickinson 1997); body length ~2.5 mm;
leg segmentation follows the NeuroMechFly morphological plan (Lobato-Rios et al.
2022 Nature Methods) without copying that model's assets.
"""

from __future__ import annotations

from collections.abc import Iterable

import mujoco

FLY_BODY_NAMES = (
    "thorax",
    "head",
    "abdomen",
    "L1_coxa",
    "L2_coxa",
    "L3_coxa",
    "R1_coxa",
    "R2_coxa",
    "R3_coxa",
    "wing_L",
    "wing_R",
)

_LEGS = (
    ("L1", 0.00028, 0.00032, 1),
    ("L2", 0.00000, 0.00038, 1),
    ("L3", -0.00030, 0.00030, 1),
    ("R1", 0.00028, -0.00032, -1),
    ("R2", 0.00000, -0.00038, -1),
    ("R3", -0.00030, -0.00030, -1),
)


def _leg_xml(name: str, x: float, y: float, side: int) -> str:
    sy = 1 if side > 0 else -1
    coxa_len = 0.00022
    femur_len = 0.00040
    tibia_len = 0.00042
    return f"""
      <body name="{name}_coxa" pos="{x} {y} -0.00012">
        <inertial pos="0 {sy * coxa_len * 0.5} -0.00002" mass="8e-9" diaginertia="2e-14 2e-14 1e-14"/>
        <joint name="{name}_yaw" type="hinge" axis="0 0 1" range="-50 50" damping="2e-9"/>
        <geom name="{name}_coxa_g" type="capsule" fromto="0 0 0 0 {sy * coxa_len} -0.00005"
              size="7e-5" rgba="0.45 0.25 0.12 1"/>
        <body name="{name}_femur" pos="0 {sy * coxa_len} -0.00005">
          <inertial pos="0 0 {-femur_len * 0.5}" mass="1.2e-8" diaginertia="3e-14 3e-14 1.5e-14"/>
          <joint name="{name}_pitch" type="hinge" axis="1 0 0" range="-10 80" damping="2e-9"/>
          <geom name="{name}_femur_g" type="capsule" fromto="0 0 0 0 {sy * 0.00005} {-femur_len}"
                size="6e-5" rgba="0.45 0.25 0.12 1"/>
          <body name="{name}_tibia" pos="0 {sy * 0.00005} {-femur_len}">
            <inertial pos="0 0 {-tibia_len * 0.5}" mass="1.2e-8" diaginertia="3e-14 3e-14 1.5e-14"/>
            <joint name="{name}_knee" type="hinge" axis="1 0 0" range="-90 10" damping="2e-9"/>
            <geom name="{name}_tibia_g" type="capsule" fromto="0 0 0 0 0 {-tibia_len}"
                  size="4.5e-5" rgba="0.4 0.22 0.1 1"/>
            <geom name="{name}_tarsus" type="capsule" fromto="0 0 {-tibia_len} 0 0 {-tibia_len - 0.00018}"
                  size="3e-5" rgba="0.35 0.18 0.08 1"/>
          </body>
        </body>
      </body>"""


def _actuator_xml(joints: Iterable[str]) -> str:
    lines = []
    for joint in joints:
        lines.append(
            f'    <motor name="{joint}" joint="{joint}" gear="2e-8" ctrllimited="true" ctrlrange="-1 1"/>'
        )
    return "\n".join(lines)


def build_fly_xml() -> str:
    legs = "\n".join(_leg_xml(*row) for row in _LEGS)
    joints = []
    for name, *_rest in _LEGS:
        joints.extend([f"{name}_yaw", f"{name}_pitch", f"{name}_knee"])
    joints.extend(["head_yaw", "head_pitch", "abdomen_pitch", "wing_L_stroke", "wing_R_stroke"])
    return f"""
<mujoco model="malecns_drosophila">
  <compiler angle="degree" inertiafromgeom="false" autolimits="true"/>
  <option timestep="0.0002" gravity="0 0 -9.81" iterations="40" solver="Newton" cone="pyramidal"/>
  <default>
    <joint limited="true" armature="2e-12"/>
    <geom condim="3" friction="1.0 0.01 0.001" solref="0.002 1" solimp="0.9 0.95 0.001"/>
  </default>
  <asset>
    <material name="cuticle" rgba="0.35 0.18 0.08 1"/>
    <material name="floor" rgba="0.45 0.5 0.35 1"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 1" dir="0 0 -1" diffuse="0.8 0.8 0.75" specular="0.2 0.2 0.2"/>
    <camera name="world_cam" pos="0.008 -0.012 0.006" xyaxes="1 0.4 0 0 0.3 1"/>
    <geom name="floor" type="plane" size="0.08 0.08 0.001" material="floor"/>
    <body name="thorax" pos="0 0 0.00065">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="4e-7" diaginertia="4e-13 4e-13 3e-13"/>
      <geom name="thorax_g" type="ellipsoid" size="0.00045 0.00038 0.00032" material="cuticle"/>
      <site name="thorax_site" pos="0 0 0" size="5e-5"/>
      <body name="head" pos="0.00055 0 0.00005">
        <inertial pos="0 0 0" mass="2e-7" diaginertia="2e-13 2e-13 1.5e-13"/>
        <joint name="head_yaw" type="hinge" axis="0 0 1" range="-40 40" damping="3e-9"/>
        <joint name="head_pitch" type="hinge" axis="0 1 0" range="-30 30" damping="3e-9"/>
        <geom name="head_g" type="ellipsoid" size="0.00032 0.00030 0.00028" material="cuticle"/>
        <site name="antenna_L" pos="0.00022 0.00012 0.00005" size="4e-5" rgba="0.2 0.8 0.2 1"/>
        <site name="antenna_R" pos="0.00022 -0.00012 0.00005" size="4e-5" rgba="0.2 0.8 0.2 1"/>
        <camera name="eye_L" pos="0.00005 0.00022 0.00004" xyaxes="0.2 -1 0 0 0 1" fovy="90"/>
        <camera name="eye_R" pos="0.00005 -0.00022 0.00004" xyaxes="-0.2 -1 0 0 0 1" fovy="90"/>
      </body>
      <body name="abdomen" pos="-0.00055 0 -0.00005">
        <inertial pos="0 0 0" mass="3e-7" diaginertia="3e-13 3e-13 2e-13"/>
        <joint name="abdomen_pitch" type="hinge" axis="0 1 0" range="-20 20" damping="4e-9"/>
        <geom name="abdomen_g" type="ellipsoid" size="0.00055 0.00032 0.00028" material="cuticle"/>
      </body>
      <body name="wing_L" pos="0.00005 0.00020 0.00012">
        <inertial pos="0 0.0004 0" mass="2e-9" diaginertia="2e-14 1e-14 2e-14"/>
        <joint name="wing_L_stroke" type="hinge" axis="1 0 0" range="-90 90" damping="1e-10"/>
        <geom name="wing_L_g" type="box" size="0.00008 0.0010 0.000015" rgba="0.7 0.75 0.8 0.4"/>
      </body>
      <body name="wing_R" pos="0.00005 -0.00020 0.00012">
        <inertial pos="0 -0.0004 0" mass="2e-9" diaginertia="2e-14 1e-14 2e-14"/>
        <joint name="wing_R_stroke" type="hinge" axis="1 0 0" range="-90 90" damping="1e-10"/>
        <geom name="wing_R_g" type="box" size="0.00008 0.0010 0.000015" rgba="0.7 0.75 0.8 0.4"/>
      </body>
      {legs}
    </body>
  </worldbody>
  <actuator>
{_actuator_xml(joints)}
  </actuator>
  <sensor>
    <accelerometer name="thorax_acc" site="thorax_site"/>
    <gyro name="thorax_gyro" site="thorax_site"/>
  </sensor>
</mujoco>
"""


def load_fly_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(build_fly_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
