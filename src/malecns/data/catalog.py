"""Body annotations and consensus neurotransmitter lookup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from malecns.data.tables import read_table_mmap
from malecns.paths import table_path


@dataclass(frozen=True)
class BodyCatalog:
    body_ids: NDArray[np.int64]
    gid_of: dict[int, int]
    superclass: list[str | None]
    cell_type: list[str | None]
    cell_class: list[str | None]
    subclass: list[str | None]
    instance: list[str | None]
    soma_side: list[str | None]
    receptor_type: list[str | None]
    consensus_nt: list[str | None]
    assigned_ol_hex1: NDArray[np.float64]
    assigned_ol_hex2: NDArray[np.float64]
    status: list[str | None]
    soma_neuromere: list[str | None]
    root_side: list[str | None]
    soma_xyz_native: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.body_ids.size)

    def gid(self, body_id: int) -> int:
        return self.gid_of[int(body_id)]


def _string_col(table: pa.Table, name: str, n: int) -> list[str | None]:
    if name not in table.column_names:
        return [None] * n
    return [None if v is None else str(v) for v in table[name].to_pylist()]


def _soma_xyz(table: pa.Table, n: int) -> np.ndarray:
    out = np.full((n, 3), np.nan, dtype=np.float64)
    if "somaLocation" not in table.column_names:
        return out
    for i, val in enumerate(table["somaLocation"].to_pylist()):
        if val is not None and len(val) >= 3:
            out[i, 0] = float(val[0])
            out[i, 1] = float(val[1])
            out[i, 2] = float(val[2])
    return out


def load_body_catalog() -> BodyCatalog:
    ann = read_table_mmap(table_path("body_annotations"))
    body_ids = np.asarray(ann["bodyId"].to_numpy(), dtype=np.int64)
    n = int(body_ids.size)
    gid_of = {int(b): i for i, b in enumerate(body_ids.tolist())}

    nt_map: dict[int, str | None] = {}
    nt_path = table_path("body_neurotransmitters")
    if nt_path.is_file():
        nt = read_table_mmap(nt_path)
        # Table has one or more rows per body; last consensus wins (stable enough).
        bodies = nt["body"].to_numpy()
        consensus = nt["consensus_nt"].to_pylist()
        for body, value in zip(bodies.tolist(), consensus, strict=False):
            nt_map[int(body)] = None if value is None else str(value)

    consensus = [nt_map.get(int(b)) for b in body_ids.tolist()]
    hex1 = (
        np.asarray(ann["assignedOlHex1"].to_numpy(zero_copy_only=False), dtype=np.float64)
        if "assignedOlHex1" in ann.column_names
        else np.full(n, np.nan)
    )
    hex2 = (
        np.asarray(ann["assignedOlHex2"].to_numpy(zero_copy_only=False), dtype=np.float64)
        if "assignedOlHex2" in ann.column_names
        else np.full(n, np.nan)
    )
    return BodyCatalog(
        body_ids=body_ids,
        gid_of=gid_of,
        superclass=_string_col(ann, "superclass", n),
        cell_type=_string_col(ann, "type", n),
        cell_class=_string_col(ann, "class", n),
        subclass=_string_col(ann, "subclass", n),
        instance=_string_col(ann, "instance", n),
        soma_side=_string_col(ann, "somaSide", n),
        receptor_type=_string_col(ann, "receptorType", n),
        consensus_nt=consensus,
        assigned_ol_hex1=hex1,
        assigned_ol_hex2=hex2,
        status=_string_col(ann, "status", n),
        soma_neuromere=_string_col(ann, "somaNeuromere", n),
        root_side=_string_col(ann, "rootSide", n),
        soma_xyz_native=_soma_xyz(ann, n),
    )
