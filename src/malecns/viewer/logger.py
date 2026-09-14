"""Log static geometry once and dynamic state at a bounded rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malecns.viewer.blueprint import malecns_blueprint


@dataclass
class SimControl:
    paused: bool = False
    speed: float = 1.0


class Workbench:
    def __init__(self, *, spawn: bool = False) -> None:
        import rerun as rr

        self.rr = rr
        rr.init("malecns", spawn=spawn)
        rr.send_blueprint(malecns_blueprint())
        self._static_logged = False
        self.control = SimControl()

    def log_static_world(self, model) -> None:
        if self._static_logged:
            return
        rr = self.rr
        # Floor as a thin box; fly geoms logged dynamically via transforms.
        rr.log(
            "world/floor",
            rr.Boxes3D(centers=[[0.0, 0.0, -0.0005]], half_sizes=[[0.08, 0.08, 0.0005]]),
            static=True,
        )
        rr.log("cns", rr.TextLog("MaleCNS morphology logged on demand / LOD"), static=True)
        self._static_logged = True

    def log_step(
        self,
        *,
        t_ms: float,
        thorax_pos: np.ndarray,
        left_image,
        right_image,
        odor_l: float,
        odor_r: float,
        ctrl: np.ndarray,
        n_spikes: int,
        realtime_ratio: float,
    ) -> None:
        rr = self.rr
        rr.set_time("sim_time", duration=float(t_ms) / 1000.0)
        rr.log(
            "world/fly/thorax",
            rr.Transform3D(translation=np.asarray(thorax_pos, dtype=np.float32)),
        )
        rr.log("eyes/left", rr.Image(left_image))
        rr.log("eyes/right", rr.Image(right_image))
        rr.log("telemetry/odor_L", rr.Scalars(float(odor_l)))
        rr.log("telemetry/odor_R", rr.Scalars(float(odor_r)))
        rr.log("telemetry/n_spikes", rr.Scalars(float(n_spikes)))
        rr.log("telemetry/realtime_ratio", rr.Scalars(float(realtime_ratio)))
        if ctrl.size:
            rr.log("telemetry/motor_rms", rr.Scalars(float(np.sqrt(np.mean(ctrl * ctrl)))))

    def log_cns_somas(self, xyz_um, rgb) -> None:
        if xyz_um is None or len(xyz_um) == 0:
            return
        self.rr.log(
            "cns/somas",
            self.rr.Points3D(positions=xyz_um, colors=rgb, radii=2.0),
            static=True,
        )
