"""Load native SWC skeletons into Arbor after converting 8 nm units to micrometers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from malecns.units import MALE_CNS_UNITS_TO_UM


@dataclass(frozen=True)
class _Sample:
    sid: int
    tag: int
    x: float
    y: float
    z: float
    r: float
    parent: int


@dataclass
class ScaledMorphology:
    morphology: object
    labels: object | None
    num_branches: int
    segment_prox: NDArray[np.float64]
    segment_dist: NDArray[np.float64]
    segment_branch: NDArray[np.int32]
    segment_t0: NDArray[np.float64]
    segment_t1: NDArray[np.float64]
    n_samples: int
    notes: tuple[str, ...] = ()


def parse_scaled_swc(path: Path | str) -> list[_Sample]:
    samples: list[_Sample] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            sid, tag, x, y, z, radius, parent = parts[:7]
            samples.append(
                _Sample(
                    sid=int(float(sid)),
                    tag=int(float(tag)),
                    x=float(x) * MALE_CNS_UNITS_TO_UM,
                    y=float(y) * MALE_CNS_UNITS_TO_UM,
                    z=float(z) * MALE_CNS_UNITS_TO_UM,
                    r=float(radius) * MALE_CNS_UNITS_TO_UM,
                    parent=int(float(parent)),
                )
            )
    if not samples:
        raise ValueError(f"no SWC samples in {path}")
    samples.sort(key=lambda s: s.sid)
    return samples


def _mpoint(arbor: object, sample: _Sample):
    return arbor.mpoint(sample.x, sample.y, sample.z, sample.r)


def _short_cable(arbor, tree, sample: _Sample, parent_seg):
    half = max(float(sample.r), 0.005)
    return tree.append(
        parent_seg,
        arbor.mpoint(sample.x, sample.y, sample.z - half, sample.r),
        arbor.mpoint(sample.x, sample.y, sample.z + half, sample.r),
        sample.tag,
    )


def build_segment_tree(samples: list[_Sample]):
    """SWC parent–child segments in µm. Mixed tags and extra roots are kept."""
    import arbor

    tree = arbor.segment_tree()
    notes: list[str] = []
    if len(samples) == 1:
        _short_cable(arbor, tree, samples[0], arbor.mnpos)
        notes.append("single-sample SWC expanded to a short soma cable (ASSUMED geometry)")
        return tree, tuple(notes)

    by_id = {s.sid: s for s in samples}
    n_children = {}
    for sample in samples:
        if sample.parent != -1 and sample.parent in by_id:
            n_children[sample.parent] = n_children.get(sample.parent, 0) + 1

    if samples[0].parent != -1:
        notes.append("first SWC sample is not a root; treating it as an extra root (not dropped)")

    seg_of: dict[int, int] = {}
    n_extra_roots = 0
    n_missing = 0
    n_isolated = 0
    for sample in samples:
        if sample.parent == -1 or sample.parent not in by_id:
            if sample.parent != -1:
                n_missing += 1
            elif sample.sid != samples[0].sid:
                n_extra_roots += 1
            if n_children.get(sample.sid, 0) == 0:
                seg_of[sample.sid] = _short_cable(arbor, tree, sample, arbor.mnpos)
                n_isolated += 1
            continue
        parent = by_id[sample.parent]
        parent_seg = seg_of.get(sample.parent, arbor.mnpos)
        idx = tree.append(parent_seg, _mpoint(arbor, parent), _mpoint(arbor, sample), sample.tag)
        seg_of[sample.sid] = idx

    if n_extra_roots:
        notes.append(f"{n_extra_roots} extra SWC roots kept as additional trees (not dropped)")
    if n_missing:
        notes.append(f"{n_missing} samples with missing parent kept as extra roots (not dropped)")
    if n_isolated:
        notes.append(f"{n_isolated} isolated roots expanded to short cables (ASSUMED geometry)")
    return tree, tuple(notes)


def _extract_segments(morphology: object) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    prox: list[list[float]] = []
    dist: list[list[float]] = []
    branch: list[int] = []
    t0: list[float] = []
    t1: list[float] = []
    n_branches = int(morphology.num_branches)
    for b in range(n_branches):
        segs = morphology.branch_segments(b)
        lengths = []
        xyz = []
        for seg in segs:
            p = (float(seg.prox.x), float(seg.prox.y), float(seg.prox.z))
            d = (float(seg.dist.x), float(seg.dist.y), float(seg.dist.z))
            xyz.append((p, d))
            lengths.append(float(np.linalg.norm(np.subtract(d, p))))
        total = float(sum(lengths))
        if total <= 0.0:
            total = 1e-12
        acc = 0.0
        for (p, d), length in zip(xyz, lengths, strict=True):
            start = acc / total
            acc += length
            prox.append(list(p))
            dist.append(list(d))
            branch.append(b)
            t0.append(start)
            t1.append(acc / total)
    if not prox:
        empty = np.zeros((0, 3), dtype=np.float64)
        z = np.zeros((0,), dtype=np.float64)
        return empty, empty.copy(), np.zeros((0,), dtype=np.int32), z, z.copy()
    return (
        np.asarray(prox, dtype=np.float64),
        np.asarray(dist, dtype=np.float64),
        np.asarray(branch, dtype=np.int32),
        np.asarray(t0, dtype=np.float64),
        np.asarray(t1, dtype=np.float64),
    )


def load_scaled_morphology(path: Path | str) -> ScaledMorphology:
    import arbor

    samples = parse_scaled_swc(path)
    tree, notes = build_segment_tree(samples)
    morphology = arbor.morphology(tree)
    prox, dist, branch, t0, t1 = _extract_segments(morphology)
    return ScaledMorphology(
        morphology=morphology,
        labels=None,
        num_branches=int(morphology.num_branches),
        segment_prox=prox,
        segment_dist=dist,
        segment_branch=branch,
        segment_t0=t0,
        segment_t1=t1,
        n_samples=len(samples),
        notes=notes,
    )
