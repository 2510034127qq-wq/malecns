"""Log static geometry once and dynamic state at a bounded rate."""

from __future__ import annotations

import numpy as np

from malecns.viewer.blueprint import malecns_blueprint
from malecns.viewer.control import SimControl, read_control, write_control


class Workbench:
    def __init__(self, *, spawn: bool = False) -> None:
        import rerun as rr

        self.rr = rr
        rr.init("malecns", spawn=spawn)
        rr.send_blueprint(malecns_blueprint())
        self._static_logged = False
        self._inspect_logged_id: int | None = None
        self.control = SimControl()
        write_control(self.control)

    def poll_control(self) -> SimControl:
        self.control = read_control()
        return self.control

    def log_static_world(self, model) -> None:
        if self._static_logged:
            return
        rr = self.rr
        rr.log(
            "world/floor",
            rr.Boxes3D(centers=[[0.0, 0.0, -0.0005]], half_sizes=[[0.08, 0.08, 0.0005]]),
            static=True,
        )
        try:
            for i in range(int(model.ngeom)):
                name = str(model.geom(i).name) or f"geom_{i}"
                if name in {"floor", "thorax_g"}:
                    continue
                size = np.asarray(model.geom_size[i], dtype=np.float32)
                pos = np.asarray(model.geom_pos[i], dtype=np.float32)
                gtype = int(model.geom_type[i])
                # 0=plane 2=sphere 3=capsule 6=box. Static env geoms only (bodyid 0).
                if int(model.geom_bodyid[i]) != 0:
                    continue
                if gtype == 6:
                    rr.log(
                        f"world/env/{name}",
                        rr.Boxes3D(centers=[pos], half_sizes=[size]),
                        static=True,
                    )
                elif gtype == 2:
                    rr.log(
                        f"world/env/{name}",
                        rr.Points3D(positions=[pos], radii=[float(size[0])]),
                        static=True,
                    )
        except Exception:
            pass
        rr.log("cns", rr.TextLog("MaleCNS morphology: somas static; inspect on demand"), static=True)
        self._static_logged = True

    def log_fly(self, model, data) -> None:
        rr = self.rr
        nbody = int(model.nbody)
        positions = np.asarray(data.xpos[1:nbody], dtype=np.float32)
        if positions.size:
            rr.log("world/fly/bodies", rr.Points3D(positions=positions, radii=0.00012))
        try:
            thorax = np.asarray(data.xpos[int(model.body("thorax").id)], dtype=np.float32)
            rr.log("world/fly/thorax", rr.Transform3D(translation=thorax))
        except Exception:
            pass

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
        left_compound=None,
        right_compound=None,
        vision_mean: float = 0.0,
        mechano_mean: float = 0.0,
        proprio_mean: float = 0.0,
        dn_drive: float = 0.0,
        wing_l: float = 0.0,
        wing_r: float = 0.0,
        body_speed: float = 0.0,
        model=None,
        data=None,
        activity_xyz=None,
        activity_rgb=None,
    ) -> None:
        rr = self.rr
        rr.set_time("sim_time", duration=float(t_ms) / 1000.0)
        rr.log(
            "world/fly/thorax",
            rr.Transform3D(translation=np.asarray(thorax_pos, dtype=np.float32)),
        )
        if model is not None and data is not None:
            self.log_fly(model, data)
        rr.log("eyes/left", rr.Image(left_image))
        rr.log("eyes/right", rr.Image(right_image))
        if left_compound is not None:
            rr.log("eyes/compound_left", rr.Image(left_compound))
        if right_compound is not None:
            rr.log("eyes/compound_right", rr.Image(right_compound))
        rr.log("telemetry/odor_L", rr.Scalars(float(odor_l)))
        rr.log("telemetry/odor_R", rr.Scalars(float(odor_r)))
        rr.log("telemetry/vision", rr.Scalars(float(vision_mean)))
        rr.log("telemetry/mechanosensory", rr.Scalars(float(mechano_mean)))
        rr.log("telemetry/proprio", rr.Scalars(float(proprio_mean)))
        rr.log("telemetry/DN", rr.Scalars(float(dn_drive)))
        rr.log("telemetry/wing_L", rr.Scalars(float(wing_l)))
        rr.log("telemetry/wing_R", rr.Scalars(float(wing_r)))
        rr.log("telemetry/body_speed", rr.Scalars(float(body_speed)))
        rr.log("telemetry/n_spikes", rr.Scalars(float(n_spikes)))
        rr.log("telemetry/realtime_ratio", rr.Scalars(float(realtime_ratio)))
        if ctrl.size:
            rr.log("telemetry/motor_rms", rr.Scalars(float(np.sqrt(np.mean(ctrl * ctrl)))))
        if activity_xyz is not None and len(activity_xyz):
            rr.log("cns/activity", rr.Points3D(positions=activity_xyz, colors=activity_rgb, radii=4.0))

    def log_cns_somas(self, xyz_um, rgb) -> None:
        if xyz_um is None or len(xyz_um) == 0:
            return
        self.rr.log(
            "cns/somas",
            self.rr.Points3D(positions=xyz_um, colors=rgb, radii=2.0),
            static=True,
        )

    def log_cns_morphology_lod(self, strips_um, path: str = "cns/morph_lod") -> None:
        if strips_um is None or len(strips_um) == 0:
            return
        self.rr.log(path, self.rr.LineStrips3D(strips_um), static=True)

    def log_inspect(
        self,
        *,
        body_id: int,
        markdown: str,
        strips_um=None,
        synapse_xyz_um=None,
    ) -> None:
        rr = self.rr
        rr.log("inspect/neuron", rr.TextDocument(markdown, media_type="text/markdown"))
        if self._inspect_logged_id != body_id:
            if strips_um is not None and len(strips_um):
                rr.log("cns/inspect/morphology", rr.LineStrips3D(strips_um), static=True)
            if synapse_xyz_um is not None and len(synapse_xyz_um):
                rr.log(
                    "cns/inspect/synapses",
                    rr.Points3D(positions=synapse_xyz_um, radii=0.8, colors=(1.0, 0.2, 0.2)),
                    static=True,
                )
            self._inspect_logged_id = body_id
