"""Hardware context selection. GPU OOM falls back to multicore CPU; the model is not shrunk."""

from __future__ import annotations

import os
from dataclasses import dataclass

import arbor


@dataclass
class ExecutionContext:
    context: object
    mode: str
    notes: list[str]


def make_execution_context(*, prefer_gpu: bool = True, threads: int | None = None) -> ExecutionContext:
    n_threads = int(threads or os.cpu_count() or 1)
    notes: list[str] = []
    if prefer_gpu:
        try:
            ctx = arbor.context(gpu_id=0, threads=max(1, min(n_threads, 8)))
            notes.append(f"requested GPU 0; context={ctx}")
            return ExecutionContext(context=ctx, mode="gpu", notes=notes)
        except Exception as exc:
            notes.append(f"GPU context failed ({exc}); using multicore CPU. Model size unchanged.")
    ctx = arbor.context(threads=n_threads)
    notes.append(f"multicore CPU context={ctx}")
    return ExecutionContext(context=ctx, mode="multicore", notes=notes)
