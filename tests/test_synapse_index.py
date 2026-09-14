"""Partner rows are indexed by post body without duplicating the Feather table."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

from malecns.synapses.partners import build_post_incidence, iter_partners_for_gid


def test_post_incidence_lists_partner_rows_per_cell(tmp_path: Path) -> None:
    path = tmp_path / "partners.feather"
    table = pa.table(
        {
            "x_pre": pa.array([0, 1, 2], type=pa.int32()),
            "y_pre": pa.array([0, 0, 0], type=pa.int32()),
            "z_pre": pa.array([0, 0, 0], type=pa.int32()),
            "body_pre": pa.array([10, 10, 11], type=pa.int64()),
            "conf_pre": pa.array([1.0, 1.0, 1.0], type=pa.float32()),
            "x_post": pa.array([0, 1, 2], type=pa.int32()),
            "y_post": pa.array([0, 0, 0], type=pa.int32()),
            "z_post": pa.array([0, 0, 0], type=pa.int32()),
            "body_post": pa.array([20, 21, 20], type=pa.int64()),
            "conf_post": pa.array([1.0, 1.0, 1.0], type=pa.float32()),
        }
    )
    feather.write_feather(table, path)
    gid_of = {20: 0, 21: 1}
    index = build_post_incidence(path, gid_of, n_gids=2, cache_dir=tmp_path / "cache")
    assert index.n_rows == 3
    g0 = iter_partners_for_gid(index, 0)
    assert set(g0["body_pre"].tolist()) == {10, 11}
    g1 = iter_partners_for_gid(index, 1)
    assert g1["body_pre"].tolist() == [10]
    # cache is an index, not a copy of the source table
    cached = list((tmp_path / "cache").glob("*"))
    assert cached
    assert all(p.stat().st_size < path.stat().st_size * 2 for p in cached)
