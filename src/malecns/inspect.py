"""Inspect one MaleCNS neuron: annotations, morphology, synapses, NT, mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from malecns.units import MALE_CNS_UNITS_TO_UM


@dataclass
class NeuronInspect:
    body_id: int
    gid: int
    superclass: str | None
    cell_type: str | None
    cell_class: str | None
    subclass: str | None
    instance: str | None
    soma_side: str | None
    soma_neuromere: str | None
    consensus_nt: str | None
    soma_xyz_um: list[float]
    n_incoming: int
    n_outgoing: int
    n_pathological: int
    n_unmapped: int
    residual_mean_um: float
    n_branches: int
    n_segments: int
    swc_path: str | None
    synapses: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        soma = ", ".join(f"{v:.2f}" for v in self.soma_xyz_um)
        syn_lines = []
        for row in self.synapses[:16]:
            syn_lines.append(
                f"- pre `{row['body_pre']}` → post `{row['body_post']}`  "
                f"xyz_um=({row['x_um']:.2f}, {row['y_um']:.2f}, {row['z_um']:.2f})  "
                f"branch={row['branch']} pos={row['pos']:.3f} residual={row['residual_um']:.2f} µm"
            )
        extra = "\n".join(syn_lines) if syn_lines else "_no incoming synapses in sample_"
        return (
            f"# Body {self.body_id} (gid {self.gid})\n\n"
            f"- type: `{self.cell_type}`\n"
            f"- superclass: `{self.superclass}` class: `{self.cell_class}` subclass: `{self.subclass}`\n"
            f"- side/neuromere: `{self.soma_side}` / `{self.soma_neuromere}`\n"
            f"- NT: `{self.consensus_nt}`\n"
            f"- soma xyz µm: {soma}\n"
            f"- incoming synapses: {self.n_incoming}\n"
            f"- outgoing synapses: {self.n_outgoing}\n"
            f"- mapped residual mean: {self.residual_mean_um:.3f} µm; "
            f"pathological={self.n_pathological}; unmapped={self.n_unmapped}\n"
            f"- morphology branches={self.n_branches} segments={self.n_segments}\n"
            f"- SWC: `{self.swc_path}`\n\n"
            f"## Sample incoming synapses\n{extra}\n"
        )


def inspect_body(body_id: int, *, max_synapses: int = 64) -> NeuronInspect:
    from malecns.data.catalog import load_body_catalog
    from malecns.data.skeletons import build_skeleton_index, swc_path_for
    from malecns.morphology.convert import load_scaled_morphology
    from malecns.parameters.physiology import load_physiology
    from malecns.paths import DATA_CACHE, SKELETONS_DIR
    from malecns.synapses.mapping import map_points_to_cable, mapping_flags
    from malecns.synapses.partners import build_post_incidence, iter_partners_for_gid

    catalog = load_body_catalog()
    if int(body_id) not in catalog.gid_of:
        raise KeyError(f"body {body_id} is not in body annotations")
    gid = catalog.gid(int(body_id))
    skeletons = build_skeleton_index(cache=True)
    try:
        swc = swc_path_for(skeletons, int(body_id))
    except FileNotFoundError:
        swc = SKELETONS_DIR / f"{int(body_id)}.swc"
        if not swc.is_file():
            swc = None
    phys = load_physiology()
    incidence = build_post_incidence(
        gid_of=catalog.gid_of,
        n_gids=len(catalog),
        cache_dir=DATA_CACHE / "post_incidence",
    )
    n_in = int(incidence.offsets[gid + 1] - incidence.offsets[gid])
    n_out = int(incidence.pre_counts[gid]) if incidence.pre_counts is not None else -1
    notes = []
    n_branches = 0
    n_seg = 0
    n_path = 0
    n_unmap = 0
    residual_mean = 0.0
    synapses: list[dict[str, Any]] = []
    strips = None
    syn_xyz = None
    if swc is not None:
        morph = load_scaled_morphology(swc)
        n_branches = int(morph.num_branches)
        n_seg = int(morph.segment_prox.shape[0])
        from malecns.viewer.cns_lod import morphology_line_strips

        strips = morphology_line_strips(morph)
        if n_in:
            rows = iter_partners_for_gid(incidence, gid)
            n_show = min(int(max_synapses), int(rows["body_pre"].size))
            xyz_native = np.stack(
                [rows["x_post"][:n_show], rows["y_post"][:n_show], rows["z_post"][:n_show]],
                axis=1,
            ).astype(np.float64)
            mapped = map_points_to_cable(morph, xyz_native * MALE_CNS_UNITS_TO_UM)
            flags = mapping_flags(mapped, residual_limit_um=float(phys.residual_limit_um.value))
            n_path = int(flags.pathological.sum())
            n_unmap = int(flags.unmapped.sum())
            finite = np.isfinite(mapped.residual_um)
            if np.any(finite):
                residual_mean = float(mapped.residual_um[finite].mean())
            syn_xyz = (xyz_native * MALE_CNS_UNITS_TO_UM).astype(np.float32)
            for i in range(n_show):
                synapses.append(
                    {
                        "body_pre": int(rows["body_pre"][i]),
                        "body_post": int(rows["body_post"][i]),
                        "x_um": float(syn_xyz[i, 0]),
                        "y_um": float(syn_xyz[i, 1]),
                        "z_um": float(syn_xyz[i, 2]),
                        "branch": int(mapped.branch[i]),
                        "pos": float(mapped.pos[i]),
                        "residual_um": float(mapped.residual_um[i]),
                    }
                )
            if n_in > n_show:
                notes.append(f"showing {n_show} of {n_in} incoming synapses")
    else:
        notes.append("no SWC; morphology unavailable")
        morph = None
        strips = None

    soma = catalog.soma_xyz_native[gid] * MALE_CNS_UNITS_TO_UM
    report = NeuronInspect(
        body_id=int(body_id),
        gid=gid,
        superclass=catalog.superclass[gid],
        cell_type=catalog.cell_type[gid],
        cell_class=catalog.cell_class[gid],
        subclass=catalog.subclass[gid],
        instance=catalog.instance[gid],
        soma_side=catalog.soma_side[gid],
        soma_neuromere=catalog.soma_neuromere[gid],
        consensus_nt=catalog.consensus_nt[gid],
        soma_xyz_um=[float(x) for x in soma.tolist()],
        n_incoming=n_in,
        n_outgoing=n_out,
        n_pathological=n_path,
        n_unmapped=n_unmap,
        residual_mean_um=residual_mean,
        n_branches=n_branches,
        n_segments=n_seg,
        swc_path=None if swc is None else str(swc),
        synapses=synapses,
        notes=notes,
    )
    report.__dict__["_strips"] = strips
    report.__dict__["_syn_xyz"] = syn_xyz
    return report
