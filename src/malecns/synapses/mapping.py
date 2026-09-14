"""Map synapse coordinates onto cable (branch, pos) using nearest-segment projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from malecns.morphology.convert import ScaledMorphology

_CHUNK = 2048


@dataclass
class MappedSites:
    branch: NDArray[np.int32]
    pos: NDArray[np.float64]
    residual_um: NDArray[np.float64]
    segment_index: NDArray[np.int32]


@dataclass
class MappingFlags:
    pathological: NDArray[np.uint8]
    unmapped: NDArray[np.uint8]


@dataclass
class MappingStats:
    n_partner_rows: int = 0
    n_mapped: int = 0
    n_unmapped: int = 0
    n_pathological: int = 0
    residual_mean_um: float = 0.0
    residual_max_um: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def map_points_to_cable(
    morph: ScaledMorphology,
    xyz_um: NDArray[np.float64],
) -> MappedSites:
    points = np.asarray(xyz_um, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    n = int(points.shape[0])
    if morph.segment_prox.shape[0] == 0:
        nan = np.full(n, np.nan, dtype=np.float64)
        return MappedSites(
            branch=np.full(n, -1, dtype=np.int32),
            pos=nan.copy(),
            residual_um=np.full(n, np.inf, dtype=np.float64),
            segment_index=np.full(n, -1, dtype=np.int32),
        )

    prox = morph.segment_prox
    vec = morph.segment_dist - prox
    length2 = np.einsum("ij,ij->i", vec, vec)
    length2 = np.maximum(length2, 1e-18)

    best_resid = np.full(n, np.inf, dtype=np.float64)
    best_seg = np.full(n, -1, dtype=np.int32)
    best_t = np.zeros(n, dtype=np.float64)

    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        chunk = points[start:stop]
        delta = chunk[:, None, :] - prox[None, :, :]
        t = np.einsum("nsi,si->ns", delta, vec) / length2[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = prox[None, :, :] + t[:, :, None] * vec[None, :, :]
        resid = np.linalg.norm(chunk[:, None, :] - closest, axis=-1)
        seg = np.argmin(resid, axis=1)
        rows = np.arange(stop - start)
        best_resid[start:stop] = resid[rows, seg]
        best_seg[start:stop] = seg.astype(np.int32, copy=False)
        best_t[start:stop] = t[rows, seg]

    t0 = morph.segment_t0[best_seg]
    t1 = morph.segment_t1[best_seg]
    pos = t0 + best_t * (t1 - t0)
    np.clip(pos, 0.0, 1.0, out=pos)
    return MappedSites(
        branch=morph.segment_branch[best_seg].astype(np.int32, copy=False),
        pos=pos,
        residual_um=best_resid,
        segment_index=best_seg,
    )


def mapping_flags(mapped: MappedSites, *, residual_limit_um: float) -> MappingFlags:
    unmapped = (
        (mapped.branch < 0)
        | ~np.isfinite(mapped.residual_um)
        | (mapped.residual_um == np.inf)
    )
    pathological = (~unmapped) & (mapped.residual_um > residual_limit_um)
    return MappingFlags(
        pathological=pathological.astype(np.uint8),
        unmapped=unmapped.astype(np.uint8),
    )


def map_synapses(*, full: bool = False) -> MappingStats:
    """CLI entry. Full mapping quality is accumulated lazily per cell in the recipe.

    Building a dense 312M-row mapping table would be a multi-GB derived copy of
    syn-partners; we keep incidence indexes under 2 GB and map onto cables at
    cell-construction time.
    """
    notes = [
        "Individual synaptic partner rows are preserved; mapping is nearest-segment projection.",
        "No extra connectivity threshold is applied beyond the published minconf-0.5 tables.",
    ]
    if not full:
        notes.append("full=False does not map the published connectome; pass --full.")
        return MappingStats(notes=notes)

    from malecns.data.catalog import load_body_catalog
    from malecns.data.skeletons import build_skeleton_index
    from malecns.paths import DATA_CACHE
    from malecns.synapses.partners import build_post_incidence

    catalog = load_body_catalog()
    skeletons = build_skeleton_index(cache=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    index = build_post_incidence(
        cache_dir=DATA_CACHE / "post_incidence",
        gid_of=catalog.gid_of,
        n_gids=len(catalog),
    )
    n_unmapped_bodies = int(np.setdiff1d(catalog.body_ids, skeletons.body_ids).size)
    return MappingStats(
        n_partner_rows=index.n_rows,
        n_mapped=0,
        n_unmapped=n_unmapped_bodies,
        notes=notes
        + [
            f"post incidence rows={index.n_rows} gids={index.n_gids}",
            f"annotation bodies missing SWC={n_unmapped_bodies}",
            "Per-synapse cable locations are computed in MaleCNSRecipe.cell_description/connections_on.",
        ],
    )
