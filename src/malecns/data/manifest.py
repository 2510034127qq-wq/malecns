"""Dataset manifest: size, schema, and counts of published MaleCNS files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from malecns.data.skeletons import build_skeleton_index
from malecns.data.tables import table_num_rows, table_schema
from malecns.paths import SKELETONS_DIR, TABLE_FILES, TABLES_DIR

# Row counts for the published minconf-0.5 bulk tables, measured by streaming
# IPC batches without holding the full table. Used when full=False to avoid a
# multi-second scan of 10+ GB files during unit tests.
KNOWN_FULL_ROW_COUNTS = {
    "body_annotations": 211_577,
    "body_neurotransmitters": 1_835_518,
    "body_stats": 88_384_522,
    "connectome_weights": 151_856_684,
    "syn_partners": 311_833_243,
    "syn_points": 357_489_383,
    "tbar_neurotransmitters": 45_656_140,
}


@dataclass
class TableRecord:
    key: str
    filename: str
    path: str
    size_bytes: int
    n_rows: int | None
    columns: list[str]
    types: list[str]


@dataclass
class DatasetManifest:
    tables: list[TableRecord]
    n_skeletons: int
    n_annotation_rows: int
    notes: list[str] = field(default_factory=list)

    def table(self, key: str) -> TableRecord:
        for rec in self.tables:
            if rec.key == key:
                return rec
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dataset_manifest(
    *,
    full: bool = False,
    tables_dir: Path | None = None,
    skeletons_dir: Path | None = None,
) -> DatasetManifest:
    tables_dir = Path(tables_dir) if tables_dir is not None else TABLES_DIR
    skeletons_dir = Path(skeletons_dir) if skeletons_dir is not None else SKELETONS_DIR
    notes: list[str] = [
        "Raw Feather/SWC files are recorded in place; this manifest is not a data copy.",
        "syn-partners is the explicit synapse table; connectome-weights is graph-level only.",
    ]
    records: list[TableRecord] = []
    annotation_rows = 0
    for key, filename in TABLE_FILES.items():
        path = tables_dir / filename
        if not path.is_file():
            notes.append(f"missing table {filename}")
            continue
        schema = table_schema(path)
        if full:
            n_rows = table_num_rows(path)
        else:
            n_rows = KNOWN_FULL_ROW_COUNTS.get(key)
            if n_rows is None:
                n_rows = table_num_rows(path)
        if key == "body_annotations":
            annotation_rows = int(n_rows or 0)
        records.append(
            TableRecord(
                key=key,
                filename=filename,
                path=str(path),
                size_bytes=int(path.stat().st_size),
                n_rows=n_rows,
                columns=list(schema.names),
                types=[str(f.type) for f in schema],
            )
        )
    index = build_skeleton_index(skeletons_dir)
    return DatasetManifest(
        tables=records,
        n_skeletons=len(index),
        n_annotation_rows=annotation_rows,
        notes=notes,
    )
