"""Dataset ingest must memory-map published Feather tables and index native SWCs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from malecns.data.manifest import build_dataset_manifest
from malecns.data.skeletons import build_skeleton_index, swc_path_for
from malecns.data.tables import iter_record_batches, table_num_rows, table_schema
from malecns.paths import SKELETONS_DIR, TABLES_DIR, table_path


def _write_feather(path: Path, table: pa.Table) -> None:
    import pyarrow.feather as feather

    feather.write_feather(table, path)


def test_table_schema_and_row_count_use_memory_map(tmp_path: Path) -> None:
    path = tmp_path / "tiny.feather"
    table = pa.table({"bodyId": pa.array([1, 2, 3], type=pa.int64()), "x": [10, 20, 30]})
    _write_feather(path, table)

    schema = table_schema(path)
    assert schema.names == ["bodyId", "x"]
    assert table_num_rows(path) == 3
    batches = list(iter_record_batches(path))
    assert sum(b.num_rows for b in batches) == 3


def test_skeleton_index_maps_body_id_to_swc(tmp_path: Path) -> None:
    (tmp_path / "10001.swc").write_text(
        "1 1 0 0 0 1 -1\n2 3 10 0 0 0.5 1\n", encoding="utf-8"
    )
    (tmp_path / "10002.swc").write_text(
        "1 1 1 1 1 1 -1\n2 3 2 2 2 0.5 1\n", encoding="utf-8"
    )
    index = build_skeleton_index(tmp_path)
    assert len(index) == 2
    assert index.body_ids.tolist() == [10001, 10002]
    assert swc_path_for(index, 10001) == tmp_path / "10001.swc"


def test_manifest_records_schema_size_and_skeleton_count(tmp_path: Path) -> None:
    tables = tmp_path / "tables"
    skeletons = tmp_path / "skeletons-swc"
    tables.mkdir()
    skeletons.mkdir()
    _write_feather(
        tables / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        pa.table({"bodyId": pa.array([11, 22], type=pa.int64())}),
    )
    (skeletons / "11.swc").write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")
    (skeletons / "22.swc").write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")

    manifest = build_dataset_manifest(
        full=True,
        tables_dir=tables,
        skeletons_dir=skeletons,
    )
    assert manifest.n_skeletons == 2
    assert manifest.n_annotation_rows == 2
    rec = manifest.table("body_annotations")
    assert rec.n_rows == 2
    assert rec.size_bytes > 0
    assert "bodyId" in rec.columns
    dumped = manifest.to_dict()
    assert dumped["n_skeletons"] == 2


@pytest.mark.data
def test_full_local_dataset_is_readable() -> None:
    assert TABLES_DIR.is_dir()
    assert SKELETONS_DIR.is_dir()
    ann = table_path("body_annotations")
    assert ann.is_file()
    assert "bodyId" in table_schema(ann).names
    partners = table_path("syn_partners")
    names = table_schema(partners).names
    for col in ("body_pre", "body_post", "x_pre", "y_pre", "z_pre", "x_post", "y_post", "z_post"):
        assert col in names
    index = build_skeleton_index(SKELETONS_DIR)
    assert len(index) >= 200_000
    assert np.issubdtype(index.body_ids.dtype, np.integer)
