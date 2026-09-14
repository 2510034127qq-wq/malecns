"""Minimal live-sim control bridge: pause / speed / inspect body ID via a JSON file.

Rerun already provides timeline pause/seek/replay for recorded data. This file is
only for the running simulator, which Rerun cannot pause by itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from malecns.paths import DATA_CACHE

CONTROL_PATH = DATA_CACHE / "control.json"


@dataclass
class SimControl:
    paused: bool = False
    speed: float = 1.0
    inspect_body_id: int | None = None


def control_path() -> Path:
    return CONTROL_PATH


def write_control(control: SimControl, path: Path | None = None) -> None:
    dest = path or CONTROL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(asdict(control), indent=2), encoding="utf-8")


def read_control(path: Path | None = None) -> SimControl:
    src = path or CONTROL_PATH
    if not src.is_file():
        return SimControl()
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SimControl()
    inspect = raw.get("inspect_body_id")
    return SimControl(
        paused=bool(raw.get("paused", False)),
        speed=float(raw.get("speed", 1.0) or 1.0),
        inspect_body_id=None if inspect in (None, "", 0) else int(inspect),
    )
