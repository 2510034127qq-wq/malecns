"""Memory-map published Feather tables. Do not convert or duplicate bulk files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa


def _open_ipc(path: Path) -> tuple[pa.memory_map, pa.ipc.RecordBatchFileReader]:
    src = pa.memory_map(str(path), "r")
    reader = pa.ipc.open_file(src)
    return src, reader


def table_schema(path: Path | str) -> pa.Schema:
    src, reader = _open_ipc(Path(path))
    try:
        return reader.schema
    finally:
        src.close()


def table_num_rows(path: Path | str) -> int:
    src, reader = _open_ipc(Path(path))
    try:
        return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
    finally:
        src.close()


def iter_record_batches(path: Path | str) -> Iterator[pa.RecordBatch]:
    src, reader = _open_ipc(Path(path))
    try:
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)
    finally:
        src.close()


def read_table_mmap(path: Path | str) -> pa.Table:
    """Read a Feather file via memory mapping. Caller must not retain huge tables."""
    import pyarrow.feather as feather

    return feather.read_table(path, memory_map=True)
