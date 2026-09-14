"""Unified MaleCNS entry point: validate, ingest, simulate, and run the closed loop."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malecns",
        description="Embodied MaleCNS v1.0 sandbox (Arbor + MuJoCo + Rerun)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-env", help="Check Arbor CUDA, MuJoCo, Rerun, and data paths")
    v.add_argument("--require-gpu", action="store_true", default=True)
    v.add_argument("--allow-cpu", action="store_true", help="Do not fail if GPU context is missing")
    v.set_defaults(func=_cmd_validate_env)

    ing = sub.add_parser("ingest", help="Build dataset manifest and skeleton index (no raw copies)")
    ing.add_argument(
        "--full",
        action="store_true",
        help="Scan all local MaleCNS tables and SWC files",
    )
    ing.set_defaults(func=_cmd_ingest)

    mp = sub.add_parser("map-synapses", help="Map synaptic partners onto cable locations")
    mp.add_argument("--full", action="store_true")
    mp.set_defaults(func=_cmd_map_synapses)

    sim = sub.add_parser("simulate", help="Instantiate the Arbor recipe and advance simulation time")
    sim.add_argument("--t-final-ms", type=float, default=1.0)
    sim.add_argument("--full", action="store_true", help="Use the complete local connectome")
    sim.set_defaults(func=_cmd_simulate)

    run = sub.add_parser("run", help="Closed sensory–CNS–motor–body loop with Rerun workbench")
    run.add_argument(
        "--experiment",
        default="sandbox",
        choices=("sandbox", "phototaxis", "ymaze", "obstacle", "looming"),
    )
    run.add_argument("--t-final-ms", type=float, default=100.0)
    run.add_argument("--spawn-viewer", action="store_true", default=True)
    run.add_argument("--no-spawn-viewer", action="store_false", dest="spawn_viewer")
    run.add_argument("--full", action="store_true", help="Use the complete local MaleCNS connectome")
    run.set_defaults(func=_cmd_run)

    return parser


def _cmd_validate_env(args: argparse.Namespace) -> int:
    from malecns.envcheck import validate_environment

    require_gpu = bool(args.require_gpu) and not bool(args.allow_cpu)
    report = validate_environment(require_gpu=require_gpu)
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from malecns.data.manifest import build_dataset_manifest

    manifest = build_dataset_manifest(full=bool(args.full))
    print(json.dumps(manifest.to_dict(), indent=2, default=str))
    return 0


def _cmd_map_synapses(args: argparse.Namespace) -> int:
    from malecns.synapses.mapping import map_synapses

    stats = map_synapses(full=bool(args.full))
    print(json.dumps(stats.to_dict(), indent=2, default=str))
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from malecns.simulation.runner import run_network

    report = run_network(t_final_ms=float(args.t_final_ms), full=bool(args.full))
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from malecns.loop import run_closed_loop

    report = run_closed_loop(
        experiment=str(args.experiment),
        t_final_ms=float(args.t_final_ms),
        spawn_viewer=bool(args.spawn_viewer),
        full=bool(getattr(args, "full", False)),
    )
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
