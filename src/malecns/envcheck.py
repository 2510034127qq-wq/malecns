"""Validate the working MaleCNS runtime: Arbor CUDA, MuJoCo, Rerun, data paths."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from malecns.paths import DATA_CACHE, DATA_RAW, SKELETONS_DIR, TABLES_DIR


@dataclass
class EnvironmentReport:
    python_version: str
    arbor_version: str
    arbor_gpu_build: str | None
    arbor_mpi_build: bool
    mujoco_version: str | None
    rerun_available: bool
    gpu_context_ok: bool
    gpu_name: str | None
    threads: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gpu_name() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()
    return line[0] if line else None


def validate_environment(*, require_gpu: bool = False) -> EnvironmentReport:
    notes: list[str] = []
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    try:
        import arbor
    except Exception as exc:  # pragma: no cover - environment must provide Arbor
        raise RuntimeError(f"Arbor import failed: {exc}") from exc

    cfg = arbor.config()
    arbor_gpu = cfg.get("gpu")
    arbor_mpi = bool(cfg.get("mpi"))

    mujoco_version = None
    try:
        import mujoco

        mujoco_version = getattr(mujoco, "__version__", "unknown")
    except Exception as exc:
        notes.append(f"mujoco unavailable: {exc}")

    rerun_available = False
    try:
        import rerun as rr  # noqa: F401

        rerun_available = True
    except Exception as exc:
        notes.append(f"rerun-sdk unavailable: {exc}")

    gpu_name = _gpu_name()
    gpu_context_ok = False
    threads = None
    try:
        n_threads = os.cpu_count() or 1
        ctx = arbor.context(gpu_id=0, threads=min(n_threads, 8))
        gpu_context_ok = bool(getattr(ctx, "has_gpu", False)) or "has_gpu True" in str(ctx)
        threads = getattr(ctx, "threads", None)
        notes.append(f"arbor.context={ctx}")
    except Exception as exc:
        notes.append(f"Arbor GPU context failed: {exc}")
        if require_gpu:
            raise RuntimeError(f"Arbor CUDA context on GPU 0 is required: {exc}") from exc

    if require_gpu and not gpu_context_ok:
        raise RuntimeError("Arbor CUDA context on GPU 0 was not created")

    for path, label in (
        (DATA_RAW, "data/raw"),
        (TABLES_DIR, "data/raw/tables"),
        (SKELETONS_DIR, "data/raw/skeletons-swc"),
        (DATA_CACHE, "data/cache"),
    ):
        if not path.exists():
            notes.append(f"missing {label} at {path}")

    return EnvironmentReport(
        python_version=python_version,
        arbor_version=str(arbor.__version__),
        arbor_gpu_build=str(arbor_gpu) if arbor_gpu is not None else None,
        arbor_mpi_build=arbor_mpi,
        mujoco_version=mujoco_version,
        rerun_available=rerun_available,
        gpu_context_ok=gpu_context_ok,
        gpu_name=gpu_name,
        threads=threads,
        notes=notes,
    )
