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
_LOCSET_CHUNK = 256


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
    nt_codes: np.ndarray | None = None
    sorted_bodies: np.ndarray | None = None
    sorted_gids: np.ndarray | None = None


@dataclass
class CellWiring:
    morph: ScaledMorphology
    locsets: list[tuple[str, str, str]]
    connections: list[object]
    n_post: int
    n_pathological: int
    n_unmapped: int
    n_dropped_pre: int
    residual_sum: float
    residual_n: int
    stub: bool
    swc_error: bool = False


def nt_key(name: str | None) -> str:
    if not name:
        return "unknown"
    key = str(name).strip().lower()
    if key in NT_KEYS:
        return key
    if key in {"ach", "acholine", "choline"}:
        return "acetylcholine"
    return "unknown"


def nt_code(name: str | None) -> int:
    key = nt_key(name)
    try:
        return NT_KEYS.index(key)
    except ValueError:
        return NT_KEYS.index("unknown")


def encode_nt_codes(names: Sequence[str | None]) -> np.ndarray:
    return np.fromiter((nt_code(n) for n in names), dtype=np.int8, count=len(names))


def _prepare_lookups(bundle: NetworkBundle) -> None:
    if bundle.nt_codes is None:
        bundle.nt_codes = encode_nt_codes(bundle.consensus_nt)
    if bundle.sorted_bodies is None:
        bodies = np.fromiter(bundle.gid_of.keys(), dtype=np.int64, count=len(bundle.gid_of))
        gids = np.fromiter(bundle.gid_of.values(), dtype=np.int64, count=len(bundle.gid_of))
        order = np.argsort(bodies)
        bundle.sorted_bodies = bodies[order]
        bundle.sorted_gids = gids[order]


def _lookup_gids(bodies: np.ndarray, sorted_bodies: np.ndarray, sorted_gids: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_bodies, bodies)
    n = int(sorted_bodies.size)
    ok = pos < n
    clipped = np.minimum(pos, max(n - 1, 0))
    if n:
        ok &= sorted_bodies[clipped] == bodies
    gids = np.full(bodies.shape[0], -1, dtype=np.int64)
    if n:
        gids[ok] = sorted_gids[pos[ok]]
    return gids


def _locset(branches: np.ndarray, pos: np.ndarray) -> str:
    terms = [f"(location {int(b)} {float(p):.6f})" for b, p in zip(branches, pos, strict=True)]
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    return "(sum " + " ".join(terms) + ")"


def _delays_ms(xyz_pre_native: np.ndarray, xyz_post_native: np.ndarray, phys: Physiology) -> np.ndarray:
    delta = (xyz_pre_native - xyz_post_native) * MALE_CNS_UNITS_TO_UM
    dist_um = np.linalg.norm(delta, axis=1)
    dist_m = dist_um * 1e-6
    v = float(phys.conduction_m_per_s.value)
    delay = (dist_m / max(v, 1e-9)) * 1000.0 + float(phys.synaptic_delay_ms.value)
    return np.maximum(delay, float(phys.min_delay_ms.value))


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
        nt_codes=np.array([0, 0], dtype=np.int8),
        sorted_bodies=np.array([10, 11], dtype=np.int64),
        sorted_gids=np.array([0, 1], dtype=np.int64),
    )


class MaleCNSRecipe:
    """Arbor recipe wrapper; subclassed at runtime because arbor.recipe is a C++ type."""

    def __new__(cls, bundle: NetworkBundle, phys: Physiology, sensory_events=None):
        import arbor

        from malecns.simulation.resources import rss_gb

        phys_ref = phys
        bundle_ref = bundle
        _prepare_lookups(bundle_ref)
        sensory = sensory_events or {}
        last: dict[str, object] = {"gid": None, "wiring": None}

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
                    "n_swc_errors": 0,
                    "n_explicit_synapses": 0,
                    "n_pathological": 0,
                    "n_unmapped_sites": 0,
                    "n_dropped_pre": 0,
                    "n_cells_wired": 0,
                    "residual_sum_um": 0.0,
                    "residual_n": 0,
                }
                self._counted: set[int] = set()

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
                return list(self._wiring(int(gid)).connections)

            def cell_description(self, gid: int):
                return self._cable_cell(self._wiring(int(gid)))

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
                for label, locset, nt in wiring.locsets:
                    if locset:
                        decor.place(locset, self._synapse(nt), label)
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
                if last["gid"] == gid and last["wiring"] is not None:
                    return last["wiring"]  # type: ignore[return-value]
                wiring = self._build_wiring(gid)
                last["gid"] = gid
                last["wiring"] = wiring
                if gid not in self._counted:
                    self._counted.add(gid)
                    acc = self.accounting
                    acc["n_cells_wired"] += 1
                    acc["n_explicit_synapses"] += len(wiring.connections)
                    acc["n_pathological"] += wiring.n_pathological
                    acc["n_unmapped_sites"] += wiring.n_unmapped
                    acc["n_dropped_pre"] += wiring.n_dropped_pre
                    acc["residual_sum_um"] += wiring.residual_sum
                    acc["residual_n"] += wiring.residual_n
                    if wiring.stub:
                        acc["n_stub_morphologies"] += 1
                    if wiring.swc_error:
                        acc["n_swc_errors"] += 1
                    n_wired = acc["n_cells_wired"]
                    if n_wired % 200 == 0 or n_wired == self.bundle.n_cells:
                        print(
                            f"[malecns] recipe cells {n_wired}/{self.bundle.n_cells} "
                            f"syn={acc['n_explicit_synapses']} rss={rss_gb():.2f}GB",
                            flush=True,
                        )
                return wiring

            def _build_wiring(self, gid: int) -> CellWiring:
                path = self.bundle.swc_for_gid(gid)
                stub = False
                swc_error = False
                if path is None or not Path(path).is_file():
                    stub = True
                    morph = _stub_morphology()
                else:
                    try:
                        morph = load_scaled_morphology(path)
                    except Exception:
                        stub = True
                        swc_error = True
                        morph = _stub_morphology()
                locsets: list[tuple[str, str, str]] = []
                connections: list[object] = []
                n_post = 0
                n_path = 0
                n_unmap = 0
                n_dropped_pre = 0
                residual_sum = 0.0
                residual_n = 0
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
                        finite = np.isfinite(mapped.residual_um) & ~flags.unmapped.astype(bool)
                        if np.any(finite):
                            vals = mapped.residual_um[finite]
                            residual_sum = float(vals.sum())
                            residual_n = int(vals.size)
                        branch = mapped.branch.copy()
                        pos = mapped.pos.copy()
                        bad = flags.unmapped.astype(bool) | (branch < 0) | ~np.isfinite(pos)
                        branch[bad] = 0
                        pos[bad] = 0.0
                        np.clip(pos, 0.0, 1.0, out=pos)
                        pre_gids = _lookup_gids(
                            np.asarray(rows["body_pre"], dtype=np.int64),
                            self.bundle.sorted_bodies,
                            self.bundle.sorted_gids,
                        )
                        n_dropped_pre = int(np.count_nonzero(pre_gids < 0))
                        codes = self.bundle.nt_codes
                        pre_nt = np.full(n_post, NT_KEYS.index("unknown"), dtype=np.int8)
                        ok_pre = pre_gids >= 0
                        if codes is not None and np.any(ok_pre):
                            pre_nt[ok_pre] = codes[pre_gids[ok_pre]]
                        xyz_pre = np.stack(
                            [rows["x_pre"], rows["y_pre"], rows["z_pre"]], axis=1
                        ).astype(np.float64)
                        delays = _delays_ms(xyz_pre, xyz, self.phys)
                        weight = float(self.phys.unitary_weight.value)
                        Ums = arbor.units.ms
                        for nt_i, nt_name in enumerate(NT_KEYS):
                            known = np.flatnonzero((pre_nt == nt_i) & (pre_gids >= 0))
                            orphan = np.flatnonzero((pre_nt == nt_i) & (pre_gids < 0))
                            if orphan.size:
                                locsets.append(
                                    (
                                        f"orphan_{nt_name}",
                                        _locset(branch[orphan], pos[orphan]),
                                        nt_name,
                                    )
                                )
                            for c0 in range(0, known.size, _LOCSET_CHUNK):
                                sl = known[c0 : c0 + _LOCSET_CHUNK]
                                label = f"post_{nt_name}_{c0}"
                                locsets.append((label, _locset(branch[sl], pos[sl]), nt_name))
                                for i in sl.tolist():
                                    connections.append(
                                        arbor.connection(
                                            arbor.cell_global_label(int(pre_gids[i]), "src"),
                                            arbor.cell_local_label(
                                                label, arbor.selection_policy.round_robin
                                            ),
                                            weight,
                                            float(delays[i]) * Ums,
                                        )
                                    )
                return CellWiring(
                    morph=morph,
                    locsets=locsets,
                    connections=connections,
                    n_post=n_post,
                    n_pathological=n_path,
                    n_unmapped=n_unmap,
                    n_dropped_pre=n_dropped_pre,
                    residual_sum=residual_sum,
                    residual_n=residual_n,
                    stub=stub,
                    swc_error=swc_error,
                )

        recipe = _Recipe()
        recipe.__dict__["_malecns_bundle"] = bundle
        return recipe


_STUB: ScaledMorphology | None = None


def _stub_morphology() -> ScaledMorphology:
    """Placeholder cable when an annotation has no SWC. Counted, not silent."""
    global _STUB
    if _STUB is not None:
        return _STUB
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".swc", delete=False) as handle:
        handle.write("1 1 0 0 0 10 -1\n2 1 100 0 0 10 1\n")
        path = Path(handle.name)
    try:
        morph = load_scaled_morphology(path)
        morph.notes = morph.notes + ("missing SWC; stub cable used and counted",)
        _STUB = morph
        return morph
    finally:
        path.unlink(missing_ok=True)
