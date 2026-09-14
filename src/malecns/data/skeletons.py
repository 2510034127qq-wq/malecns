"""Index native MaleCNS SWC skeletons by body ID without copying files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from malecns.paths import DATA_CACHE, SKELETONS_DIR


@dataclass(frozen=True)
class SkeletonIndex:
    root: Path
    body_ids: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.body_ids.size)

    def has(self, body_id: int) -> bool:
        gid = int(np.searchsorted(self.body_ids, body_id))
        return gid < self.body_ids.size and int(self.body_ids[gid]) == int(body_id)


def build_skeleton_index(
    root: Path | None = None,
    *,
    cache: bool = False,
) -> SkeletonIndex:
    root = Path(root) if root is not None else SKELETONS_DIR
    cache_path = DATA_CACHE / "skeleton_body_ids.npy"
    if cache and cache_path.is_file() and root == SKELETONS_DIR:
        body_ids = np.load(cache_path)
        return SkeletonIndex(root=root, body_ids=np.asarray(body_ids, dtype=np.int64))

    ids: list[int] = []
    for path in root.iterdir():
        if path.suffix.lower() != ".swc":
            continue
        ids.append(int(path.stem))
    body_ids = np.array(sorted(ids), dtype=np.int64)
    if cache and root == SKELETONS_DIR:
        DATA_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, body_ids)
    return SkeletonIndex(root=root, body_ids=body_ids)


def swc_path_for(index: SkeletonIndex, body_id: int) -> Path:
    path = index.root / f"{int(body_id)}.swc"
    if not path.is_file():
        raise FileNotFoundError(f"no native SWC for body {body_id}: {path}")
    return path
