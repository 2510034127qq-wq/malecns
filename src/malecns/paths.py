"""Repository and dataset locations. Raw data is never copied into Git."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_CACHE = DATA_ROOT / "cache"
TABLES_DIR = DATA_RAW / "tables"
SKELETONS_DIR = DATA_RAW / "skeletons-swc"
CONFIGS_DIR = REPO_ROOT / "configs"
LOGS_DIR = REPO_ROOT / "logs"

TABLE_FILES = {
    "body_annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body_neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "body_stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
    "connectome_weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "syn_partners": "syn-partners-male-cns-v1.0-minconf-0.5.feather",
    "syn_points": "syn-points-male-cns-v1.0-minconf-0.5.feather",
    "tbar_neurotransmitters": "tbar-neurotransmitters-male-cns-v1.0.feather",
}


def table_path(key: str) -> Path:
    return TABLES_DIR / TABLE_FILES[key]
