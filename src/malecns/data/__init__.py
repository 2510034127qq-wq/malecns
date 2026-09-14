"""MaleCNS bulk-data access: memory-mapped tables, SWC index, manifest."""

from malecns.data.manifest import DatasetManifest, build_dataset_manifest
from malecns.data.skeletons import SkeletonIndex, build_skeleton_index, swc_path_for
from malecns.data.tables import iter_record_batches, table_num_rows, table_schema

__all__ = [
    "DatasetManifest",
    "SkeletonIndex",
    "build_dataset_manifest",
    "build_skeleton_index",
    "iter_record_batches",
    "swc_path_for",
    "table_num_rows",
    "table_schema",
]
