"""Map synapse coordinates onto cable (branch, pos) using nearest-segment projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from malecns.morphology.convert import ScaledMorphology

_POINT_CHUNK = 2048
_SEG_CHUNK = 2048
_BRUTE_LIMIT = 8_000_000
_KDTREE_K = 48


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


def _project_candidates(
    points: NDArray[np.float64],
    prox: NDArray[np.float64],
    vec: NDArray[np.float64],
    length2: NDArray[np.float64],
    cand: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    """Exact nearest-segment projection onto per-point candidate indices (n, k)."""
    prox_c = prox[cand]
    vec_c = vec[cand]
    len_c = length2[cand]
    delta = points[:, None, :] - prox_c
    t = np.einsum("nki,nki->nk", delta, vec_c) / len_c
    t = np.clip(t, 0.0, 1.0)
    closest = prox_c + t[:, :, None] * vec_c
    resid = np.linalg.norm(points[:, None, :] - closest, axis=-1)
    best = np.argmin(resid, axis=1)
    rows = np.arange(points.shape[0])
    return resid[rows, best], cand[rows, best].astype(np.int32, copy=False), t[rows, best]


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
    nseg = int(prox.shape[0])
    length2 = np.einsum("ij,ij->i", vec, vec)
    length2 = np.maximum(length2, 1e-18)

    if n * nseg > _BRUTE_LIMIT and nseg > _KDTREE_K:
        from scipy.spatial import cKDTree

        samples = np.concatenate([prox, morph.segment_dist, 0.5 * (prox + morph.segment_dist)])
        seg_of = np.tile(np.arange(nseg, dtype=np.int32), 3)
        tree = cKDTree(samples)
        k = int(min(_KDTREE_K, samples.shape[0]))
        _, nn = tree.query(points, k=k, workers=1)
        nn = np.asarray(nn, dtype=np.int64)
        if nn.ndim == 1:
            nn = nn.reshape(-1, 1)
        cand = seg_of[nn]
        best_resid, best_seg, best_t = _project_candidates(points, prox, vec, length2, cand)
    else:
        best_resid = np.full(n, np.inf, dtype=np.float64)
        best_seg = np.full(n, -1, dtype=np.int32)
        best_t = np.zeros(n, dtype=np.float64)
        for start in range(0, n, _POINT_CHUNK):
            stop = min(start + _POINT_CHUNK, n)
            chunk = points[start:stop]
            n_pt = stop - start
            chunk_best = np.full(n_pt, np.inf, dtype=np.float64)
            chunk_seg = np.full(n_pt, -1, dtype=np.int32)
            chunk_t = np.zeros(n_pt, dtype=np.float64)
            for seg0 in range(0, nseg, _SEG_CHUNK):
                seg1 = min(seg0 + _SEG_CHUNK, nseg)
                prox_c = prox[seg0:seg1]
                vec_c = vec[seg0:seg1]
                len_c = length2[seg0:seg1]
                delta = chunk[:, None, :] - prox_c[None, :, :]
                t = np.einsum("nsi,si->ns", delta, vec_c) / len_c[None, :]
                t = np.clip(t, 0.0, 1.0)
                closest = prox_c[None, :, :] + t[:, :, None] * vec_c[None, :, :]
                resid = np.linalg.norm(chunk[:, None, :] - closest, axis=-1)
                seg_local = np.argmin(resid, axis=1)
                rows = np.arange(n_pt)
                resid_min = resid[rows, seg_local]
                better = resid_min < chunk_best
                chunk_best[better] = resid_min[better]
                chunk_seg[better] = (seg0 + seg_local[better]).astype(np.int32)
                chunk_t[better] = t[rows, seg_local][better]
            best_resid[start:stop] = chunk_best
            best_seg[start:stop] = chunk_seg
            best_t[start:stop] = chunk_t

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


def _stats_path():
    from malecns.paths import DATA_CACHE

    return DATA_CACHE / "mapping_stats.json"


def save_mapping_stats(stats: MappingStats) -> None:
    import json

    from malecns.paths import DATA_CACHE

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    _stats_path().write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")


def load_mapping_stats() -> MappingStats | None:
    import json

    path = _stats_path()
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return MappingStats(**raw)


def map_synapses(*, full: bool = False, resume: bool = True) -> MappingStats:
    """Map every explicit partner row onto the postsynaptic cable. No per-row cache.

    Residuals and flags are accumulated; pathological/unmapped sites are counted
    and still placed at the nearest cable location in the recipe. A dense 312M-row
    (branch, pos) table is not written.
    """
    notes = [
        "Individual synaptic partner rows are preserved; mapping is nearest-segment projection.",
        "No extra connectivity threshold is applied beyond the published minconf-0.5 tables.",
    ]
    if not full:
        notes.append("full=False does not map the published connectome; pass --full.")
        return MappingStats(notes=notes)

    from malecns.data.catalog import load_body_catalog
    from malecns.morphology.convert import load_scaled_morphology
    from malecns.parameters.physiology import load_physiology
    from malecns.paths import DATA_CACHE, SKELETONS_DIR
    from malecns.synapses.partners import build_post_incidence, ensure_partner_store
    from malecns.units import MALE_CNS_UNITS_TO_UM

    catalog = load_body_catalog()
    phys = load_physiology()
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    index = build_post_incidence(
        cache_dir=DATA_CACHE / "post_incidence",
        gid_of=catalog.gid_of,
        n_gids=len(catalog),
    )
    store = ensure_partner_store(index)
    progress_path = DATA_CACHE / "mapping_progress.npz"
    start_gid = 0
    n_mapped = 0
    n_unmapped = 0
    n_path = 0
    residual_sum = 0.0
    residual_max = 0.0
    n_resid = 0
    n_missing_swc = 0
    n_swc_error = 0
    if resume and progress_path.is_file():
        prog = np.load(progress_path)
        start_gid = int(prog["gid"])
        n_mapped = int(prog["n_mapped"])
        n_unmapped = int(prog["n_unmapped"])
        n_path = int(prog["n_path"])
        residual_sum = float(prog["residual_sum"])
        residual_max = float(prog["residual_max"])
        n_resid = int(prog["n_resid"])
        n_missing_swc = int(prog["n_missing_swc"])
        n_swc_error = int(prog["n_swc_error"])
        print(f"[malecns] resume mapping at gid={start_gid}", flush=True)

    residual_limit = float(phys.residual_limit_um.value)
    n_gids = len(catalog)
    for gid in range(start_gid, n_gids):
        start = int(index.offsets[gid])
        stop = int(index.offsets[gid + 1])
        n_post = stop - start
        body_id = int(catalog.body_ids[gid])
        swc = SKELETONS_DIR / f"{body_id}.swc"
        if n_post == 0:
            if not swc.is_file():
                n_missing_swc += 1
        elif not swc.is_file():
            n_missing_swc += 1
            n_unmapped += n_post
        else:
            try:
                morph = load_scaled_morphology(swc)
            except Exception:
                n_swc_error += 1
                n_unmapped += n_post
            else:
                rows = store.gather(index.row_ids[start:stop])
                xyz = np.stack(
                    [rows["x_post"], rows["y_post"], rows["z_post"]], axis=1
                ).astype(np.float64)
                mapped = map_points_to_cable(morph, xyz * MALE_CNS_UNITS_TO_UM)
                flags = mapping_flags(mapped, residual_limit_um=residual_limit)
                unmapped = flags.unmapped.astype(bool)
                n_unmapped += int(unmapped.sum())
                n_path += int(flags.pathological.sum())
                n_mapped += int((~unmapped).sum())
                finite = np.isfinite(mapped.residual_um) & ~unmapped
                if np.any(finite):
                    vals = mapped.residual_um[finite]
                    residual_sum += float(vals.sum())
                    residual_max = max(residual_max, float(vals.max()))
                    n_resid += int(vals.size)
        if gid % 200 == 0 or gid + 1 == n_gids:
            print(
                f"[malecns] map gid={gid + 1}/{n_gids} mapped={n_mapped} "
                f"unmapped={n_unmapped} path={n_path}",
                flush=True,
            )
            np.savez(
                progress_path,
                gid=np.int64(gid + 1),
                n_mapped=n_mapped,
                n_unmapped=n_unmapped,
                n_path=n_path,
                residual_sum=residual_sum,
                residual_max=residual_max,
                n_resid=n_resid,
                n_missing_swc=n_missing_swc,
                n_swc_error=n_swc_error,
            )

    stats = MappingStats(
        n_partner_rows=index.n_rows,
        n_mapped=n_mapped,
        n_unmapped=n_unmapped + int(index.n_missing_post),
        n_pathological=n_path,
        residual_mean_um=(residual_sum / n_resid) if n_resid else 0.0,
        residual_max_um=residual_max,
        notes=notes
        + [
            f"post incidence rows={index.n_rows} indexed={int(index.offsets[-1])}",
            f"partner rows with post body not in annotations={index.n_missing_post} (counted unmapped)",
            f"annotation bodies missing SWC={n_missing_swc}",
            f"SWC load errors={n_swc_error}",
            "Pathological sites are placed at the nearest cable and counted, not dropped.",
        ],
    )
    save_mapping_stats(stats)
    return stats
