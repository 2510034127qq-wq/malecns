"""Memory and VRAM snapshots for simulation accounting."""

from __future__ import annotations

from pathlib import Path


def rss_gb() -> float:
    status = Path("/proc/self/status")
    if not status.is_file():
        return float("nan")
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            kb = float(line.split()[1])
            return kb / 1e6
    return float("nan")


def gpu_memory_gb() -> float | None:
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    proc = subprocess.run(
        [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return float(proc.stdout.strip().splitlines()[0]) / 1024.0
    except ValueError:
        return None
