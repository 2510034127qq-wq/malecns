"""Arbor cable-cell recipe over explicit MaleCNS morphologies and synaptic partners."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from malecns.morphology.convert import ScaledMorphology, load_scaled_morphology
from malecns.parameters.physiology import Physiology
from malecns.synapses.mapping import map_points_to_cable, mapping_flags
from malecns.synapses.partners import PostIncidence, build_post_incidence, iter_partners_for_gid
from malecns.units import MALE_CNS_UNITS_TO_UM

NT_KEYS = ("acetylcholine", "gaba", "glutamate", "histamine", "unknown")


@dataclass
class NetworkBundle:
    n_cells: int
    body_ids: np.ndarray
    gid_of: dict[int, int]
    swc_for_gid: Callable[[int], Path | None]
    consensus_nt: Sequence[str | None]
    incidence: PostIncidence | None
    residual_limit_um: float
    notes: list[str] = field(default_factory=list)


@dataclass
class CellWiring:
    morph: ScaledMorphology
    locsets: dict[str, str]
    connections: list[object]
    n_post: int
    n_pathological: int
    n_unmapped: int
    stub: bool


def nt_key(name: str | None) -> str:
    if not name:
        return "unknown"
    key = str(name).strip().lower()
    if key in NT_KEYS:
        return key
    if key in {"ach", "aCh", "choline"}:
        return "acetylcholine"
    return "unknown"


def _locset(branches: np.ndarray, pos: np.ndarray) -> str:
    terms = [f"(location {int(b)} {float(p):.9f})" for b, p in zip(branches, pos, strict=True)]
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    return "(sum " + " ".join(terms) + ")"


def _delay_ms(
    xyz_pre_native: np.ndarray,
    xyz_post_native: np.ndarray,
    phys: Physiology,
) -> float:
    delta = (xyz_pre_native - xyz_post_native) * MALE_CNS_UNITS_TO_UM
    dist_um = float(np.linalg.norm(delta))
    dist_m = dist_um * 1e-6
    v = float(phys.conduction_m_per_s.value)
    t_cond_ms = (dist_m / max(v, 1e-9)) * 1000.0
    delay = t_cond_ms + float(phys.synaptic_delay_ms.value)
    return max(delay, float(phys.min_delay_ms.value))


def debug_two_cell_bundle(workdir: Path, phys: Physiology) -> NetworkBundle:
    """Synthetic two-cell cable network for tests. Not a reduced MaleCNS product."""
    workdir = Path(workdir)
    skeletons = workdir / "swc"
    skeletons.mkdir(parents=True, exist_ok=True)
    cable = "1 1 0 0 0 10 -1\n2 1 1000 0 0 10 1\n"
    (skeletons / "10.swc").write_text(cable, encoding="utf-8")
    (skeletons / "11.swc").write_text(cable, encoding="utf-8")
    partners = workdir / "partners.feather"
    table = pa.table(
        {
            "x_pre": pa.array([500], type=pa.int32()),
            "y_pre": pa.array([0], type=pa.int32()),
            "z_pre": pa.array([0], type=pa.int32()),
            "body_pre": pa.array([10], type=pa.int64()),
            "conf_pre": pa.array([1.0], type=pa.float32()),
            "x_post": pa.array([500], type=pa.int32()),
            "y_post": pa.array([0], type=pa.int32()),
            "z_post": pa.array([0], type=pa.int32()),
            "body_post": pa.array([11], type=pa.int64()),
            "conf_post": pa.array([1.0], type=pa.float32()),
        }
    )
    feather.write_feather(table, partners)
    gid_of = {10: 0, 11: 1}
    incidence = build_post_incidence(
        partners, gid_of, n_gids=2, cache_dir=workdir / "incidence"
    )
    paths = {0: skeletons / "10.swc", 1: skeletons / "11.swc"}
    return NetworkBundle(
        n_cells=2,
        body_ids=np.array([10, 11], dtype=np.int64),
        gid_of=gid_of,
        swc_for_gid=lambda gid: paths[int(gid)],
        consensus_nt=["acetylcholine", "acetylcholine"],
        incidence=incidence,
        residual_limit_um=float(phys.residual_limit_um.value),
        notes=["debug two-cell network; not the published MaleCNS connectome"],
    )


class MaleCNSRecipe:
    """Arbor recipe wrapper; subclassed at runtime because arbor.recipe is a C++ type."""

    def __new__(cls, bundle: NetworkBundle, phys: Physiology, sensory_events=None):
        import arbor

        phys_ref = phys
        bundle_ref = bundle
        cache: dict[int, CellWiring] = {}
        sensory = sensory_events or {}

        class _Recipe(arbor.recipe):
            def __init__(self) -> None:
                super().__init__()
                self.bundle = bundle_ref
                self.phys = phys_ref
                self.cv_policy = arbor.cv_policy_max_extent_um(
                    float(phys_ref.cv_max_extent_um.value)
                )
                self.accounting = {
                    "n_stub_morphologies": 0,
                    "n_explicit_synapses": 0,
                    "n_pathological": 0,
                    "n_unmapped_sites": 0,
                }

            def num_cells(self) -> int:
                return int(self.bundle.n_cells)

            def cell_kind(self, gid: int):
                return arbor.cell_kind.cable

            def global_properties(self, kind):
                return arbor.neuron_cable_properties()

            def probes(self, gid: int):
                if int(gid) >= min(32, self.bundle.n_cells):
                    return []
                return [arbor.cable_probe_membrane_voltage("(location 0 0)", "Um")]

            def event_generators(self, gid: int):
                events = sensory.get(int(gid), [])
                if not events:
                    return []
                times = [t * arbor.units.ms for t, _w in events]
                weight = float(events[0][1]) if events else 0.0
                return [
                    arbor.event_generator(
                        "sensory_in",
                        weight,
                        arbor.explicit_schedule(times),
                    )
                ]

            def connections_on(self, gid: int):
                wiring = self._wiring(int(gid))
                return list(wiring.connections)

            def cell_description(self, gid: int):
                wiring = self._wiring(int(gid))
                return self._cable_cell(wiring)

            def _cable_cell(self, wiring: CellWiring):
                U = arbor.units
                p = self.phys
                decor = arbor.decor()
                decor.set_property(
                    Vm=float(p.vm0_mV.value) * U.mV,
                    cm=float(p.cm_F_per_m2.value) * U.F / U.m2,
                    rL=float(p.ra_ohm_cm.value) * U.Ohm * U.cm,
                    tempK=float(p.temperature_K.value) * U.Kelvin,
                )
                e_pas = float(p.leak_e_mV.value)
                g_pas = float(p.leak_g_S_per_cm2.value)
                decor.paint("(all)", arbor.density(f"pas/e={e_pas}", {"g": g_pas}))
                decor.place(
                    "(location 0 0)",
                    arbor.threshold_detector(float(p.spike_threshold_mV.value) * U.mV),
                    "src",
                )
                unk = nt_key("unknown")
                decor.place("(location 0 0)", self._synapse(unk), "sensory_in")
                for nt, locset in wiring.locsets.items():
                    if locset:
                        decor.place(locset, self._synapse(nt), f"post_{nt}")
                return arbor.cable_cell(
                    wiring.morph.morphology,
                    decor,
                    discretization=self.cv_policy,
                )

            def _synapse(self, nt: str):
                key = nt if nt in self.phys.syn_e_mV else "unknown"
                tau = float(self.phys.syn_tau_ms[key].value)
                e_rev = float(self.phys.syn_e_mV[key].value)
                return arbor.synapse("expsyn", {"tau": tau, "e": e_rev})

            def _wiring(self, gid: int) -> CellWiring:
                if gid in cache:
                    return cache[gid]
                path = self.bundle.swc_for_gid(gid)
                stub = False
                if path is None or not Path(path).is_file():
                    stub = True
                    self.accounting["n_stub_morphologies"] += 1
                    morph = _stub_morphology()
                else:
                    morph = load_scaled_morphology(path)
                locsets: dict[str, str] = {}
                connections: list[object] = []
                n_post = 0
                n_path = 0
                n_unmap = 0
                if self.bundle.incidence is not None:
                    rows = iter_partners_for_gid(self.bundle.incidence, gid)
                    n_post = int(rows["body_pre"].size)
                    if n_post:
                        xyz = np.stack(
                            [rows["x_post"], rows["y_post"], rows["z_post"]], axis=1
                        ).astype(np.float64)
                        mapped = map_points_to_cable(morph, xyz * MALE_CNS_UNITS_TO_UM)
                        flags = mapping_flags(
                            mapped, residual_limit_um=self.bundle.residual_limit_um
                        )
                        n_path = int(flags.pathological.sum())
                        n_unmap = int(flags.unmapped.sum())
                        pre_nt = []
                        for b in rows["body_pre"].tolist():
                            pre_gid = self.bundle.gid_of.get(int(b))
                            if pre_gid is None:
                                pre_nt.append("unknown")
                            else:
                                pre_nt.append(nt_key(self.bundle.consensus_nt[pre_gid]))
                        xyz_pre = np.stack(
                            [rows["x_pre"], rows["y_pre"], rows["z_pre"]], axis=1
                        ).astype(np.float64)
                        for nt in NT_KEYS:
                            idx = np.array(
                                [i for i, name in enumerate(pre_nt) if name == nt],
                                dtype=np.int64,
                            )
                            if idx.size == 0:
                                continue
                            locsets[nt] = _locset(mapped.branch[idx], mapped.pos[idx])
                            for i in idx.tolist():
                                pre_body = int(rows["body_pre"][i])
                                if pre_body not in self.bundle.gid_of:
                                    continue
                                pre_gid = self.bundle.gid_of[pre_body]
                                delay = _delay_ms(xyz_pre[i], xyz[i], self.phys)
                                connections.append(
                                    arbor.connection(
                                        arbor.cell_global_label(pre_gid, "src"),
                                        arbor.cell_local_label(
                                            f"post_{nt}",
                                            arbor.selection_policy.round_robin,
                                        ),
                                        float(self.phys.unitary_weight.value),
                                        delay * arbor.units.ms,
                                    )
                                )
                        self.accounting["n_explicit_synapses"] += len(connections)
                        self.accounting["n_pathological"] += n_path
                        self.accounting["n_unmapped_sites"] += n_unmap
                wiring = CellWiring(
                    morph=morph,
                    locsets=locsets,
                    connections=connections,
                    n_post=n_post,
                    n_pathological=n_path,
                    n_unmapped=n_unmap,
                    stub=stub,
                )
                cache[gid] = wiring
                return wiring

        recipe = _Recipe()
        recipe.__dict__["_malecns_bundle"] = bundle
        return recipe


def _stub_morphology() -> ScaledMorphology:
    """Placeholder cable when an annotation has no SWC. Counted, not silent."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".swc", delete=False) as handle:
        handle.write("1 1 0 0 0 10 -1\n2 1 100 0 0 10 1\n")
        path = Path(handle.name)
    try:
        morph = load_scaled_morphology(path)
        morph.notes = morph.notes + ("missing SWC; stub cable used and counted",)
        return morph
    finally:
        path.unlink(missing_ok=True)
