"""Index syn-partners by postsynaptic body without duplicating the Feather table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from malecns.data.tables import iter_record_batches
from malecns.paths import DATA_CACHE, table_path

_PARTNER_COLUMNS = (
    "body_pre",
    "body_post",
    "x_pre",
    "y_pre",
    "z_pre",
    "x_post",
    "y_post",
    "z_post",
    "conf_pre",
    "conf_post",
)


@dataclass
class PostIncidence:
    partners_path: Path
    offsets: NDArray[np.int64]
    row_ids: NDArray[np.uint32]
    n_rows: int
    n_gids: int
    n_missing_post: int


def _lookup_gids(
    body_post: NDArray[np.int64],
    sorted_bodies: NDArray[np.int64],
    sorted_gids: NDArray[np.int64],
) -> NDArray[np.int64]:
    pos = np.searchsorted(sorted_bodies, body_post)
    n = int(sorted_bodies.size)
    ok = pos < n
    clipped = np.minimum(pos, max(n - 1, 0))
    if n:
        ok &= sorted_bodies[clipped] == body_post
    gids = np.full(body_post.shape[0], -1, dtype=np.int64)
    if n:
        gids[ok] = sorted_gids[pos[ok]]
    return gids


def build_post_incidence(
    partners_path: Path | str | None = None,
    gid_of: dict[int, int] | None = None,
    *,
    n_gids: int,
    cache_dir: Path | str | None = None,
) -> PostIncidence:
    path = Path(partners_path) if partners_path is not None else table_path("syn_partners")
    cache = Path(cache_dir) if cache_dir is not None else DATA_CACHE / "post_incidence"
    cache.mkdir(parents=True, exist_ok=True)
    offsets_path = cache / "offsets.npy"
    rows_path = cache / "row_ids.npy"
    meta_path = cache / "meta.npy"
    if offsets_path.is_file() and rows_path.is_file() and meta_path.is_file():
        meta = np.load(meta_path)
        return PostIncidence(
            partners_path=path,
            offsets=np.load(offsets_path),
            row_ids=np.load(rows_path),
            n_rows=int(meta[0]),
            n_gids=int(meta[1]),
            n_missing_post=int(meta[2]),
        )
    if gid_of is None:
        raise ValueError("gid_of is required when incidence cache is absent")

    bodies = np.fromiter(gid_of.keys(), dtype=np.int64)
    gids = np.fromiter(gid_of.values(), dtype=np.int64)
    order = np.argsort(bodies)
    sorted_bodies = bodies[order]
    sorted_gids = gids[order]

    counts = np.zeros(n_gids, dtype=np.int64)
    missing = 0
    n_rows = 0
    for batch in iter_record_batches(path):
        body_post = np.asarray(batch.column("body_post").to_numpy(), dtype=np.int64)
        mapped = _lookup_gids(body_post, sorted_bodies, sorted_gids)
        missing += int(np.count_nonzero(mapped < 0))
        valid = mapped >= 0
        np.add.at(counts, mapped[valid], 1)
        n_rows += int(batch.num_rows)

    offsets = np.zeros(n_gids + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    row_ids = np.empty(int(offsets[-1]), dtype=np.uint32)
    write_at = offsets[:-1].copy()
    global_row = 0
    for batch in iter_record_batches(path):
        body_post = np.asarray(batch.column("body_post").to_numpy(), dtype=np.int64)
        mapped = _lookup_gids(body_post, sorted_bodies, sorted_gids)
        local = np.arange(batch.num_rows, dtype=np.int64)
        valid = mapped >= 0
        g = mapped[valid]
        rows = (global_row + local[valid]).astype(np.uint32)
        if g.size:
            order = np.argsort(g, kind="stable")
            gs = g[order]
            rs = rows[order]
            change = np.ones(gs.size, dtype=bool)
            change[1:] = gs[1:] != gs[:-1]
            run_start = np.flatnonzero(change)
            run_sizes = np.diff(np.r_[run_start, gs.size])
            ranks = np.arange(gs.size, dtype=np.int64) - np.repeat(run_start, run_sizes)
            dest = write_at[gs] + ranks
            row_ids[dest] = rs
            np.add.at(write_at, g, 1)
        global_row += int(batch.num_rows)

    np.save(offsets_path, offsets)
    np.save(rows_path, row_ids)
    np.save(meta_path, np.array([n_rows, n_gids, missing], dtype=np.int64))
    return PostIncidence(
        partners_path=path,
        offsets=offsets,
        row_ids=row_ids,
        n_rows=n_rows,
        n_gids=n_gids,
        n_missing_post=missing,
    )


def iter_partners_for_gid(index: PostIncidence, gid: int) -> dict[str, np.ndarray]:
    start = int(index.offsets[gid])
    stop = int(index.offsets[gid + 1])
    row_ids = index.row_ids[start:stop]
    return fetch_partner_rows(index.partners_path, row_ids)


def fetch_partner_rows(path: Path | str, row_ids: NDArray[np.uint32]) -> dict[str, np.ndarray]:
    wanted = np.asarray(row_ids, dtype=np.int64)
    empty = {
        "body_pre": np.zeros(0, dtype=np.int64),
        "body_post": np.zeros(0, dtype=np.int64),
        "x_pre": np.zeros(0, dtype=np.int32),
        "y_pre": np.zeros(0, dtype=np.int32),
        "z_pre": np.zeros(0, dtype=np.int32),
        "x_post": np.zeros(0, dtype=np.int32),
        "y_post": np.zeros(0, dtype=np.int32),
        "z_post": np.zeros(0, dtype=np.int32),
        "conf_pre": np.zeros(0, dtype=np.float32),
        "conf_post": np.zeros(0, dtype=np.float32),
    }
    if wanted.size == 0:
        return empty

    order = np.argsort(wanted)
    sorted_ids = wanted[order]
    collected: dict[str, list[np.ndarray]] = {name: [] for name in _PARTNER_COLUMNS}
    cursor = 0
    start = 0
    for batch in iter_record_batches(path):
        end = start + int(batch.num_rows)
        lo = int(np.searchsorted(sorted_ids, start, side="left"))
        hi = int(np.searchsorted(sorted_ids, end, side="left"))
        if hi > lo:
            local = (sorted_ids[lo:hi] - start).astype(np.int64)
            for name in _PARTNER_COLUMNS:
                col = batch.column(name).to_numpy()
                collected[name].append(np.asarray(col)[local])
            cursor += hi - lo
        start = end
        if cursor >= sorted_ids.size:
            break

    out: dict[str, np.ndarray] = {}
    for name in _PARTNER_COLUMNS:
        stacked = np.concatenate(collected[name])
        restored = np.empty_like(stacked)
        restored[order] = stacked
        out[name] = restored
    return out
