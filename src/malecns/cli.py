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
    mp.add_argument("--no-resume", action="store_true")
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

    insp = sub.add_parser("inspect", help="Inspect one neuron by body ID")
    insp.add_argument("--body-id", type=int, required=True)
    insp.add_argument("--max-synapses", type=int, default=64)
    insp.set_defaults(func=_cmd_inspect)

    ctl = sub.add_parser("control", help="Live pause/resume/speed/inspect bridge (control.json)")
    ctl.add_argument("--pause", action="store_true")
    ctl.add_argument("--resume", action="store_true")
    ctl.add_argument("--speed", type=float, default=None)
    ctl.add_argument("--inspect-body-id", type=int, default=None)
    ctl.set_defaults(func=_cmd_control)

    acc = sub.add_parser("accept", help="Full-data GOAL acceptance: env, ingest, map, simulate, loop")
    acc.add_argument("--t-final-ms", type=float, default=0.25)
    acc.add_argument("--loop-ms", type=float, default=1.0)
    acc.add_argument("--skip-map", action="store_true")
    acc.add_argument("--skip-simulate", action="store_true")
    acc.add_argument("--skip-loop", action="store_true")
    acc.add_argument("--spawn-viewer", action="store_true")
    acc.set_defaults(func=_cmd_accept)

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

    stats = map_synapses(full=bool(args.full), resume=not bool(args.no_resume))
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


def _cmd_inspect(args: argparse.Namespace) -> int:
    from malecns.inspect import inspect_body

    report = inspect_body(int(args.body_id), max_synapses=int(args.max_synapses))
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def _cmd_control(args: argparse.Namespace) -> int:
    from malecns.viewer.control import read_control, write_control

    ctl = read_control()
    if args.pause:
        ctl.paused = True
    if args.resume:
        ctl.paused = False
    if args.speed is not None:
        ctl.speed = float(args.speed)
    if args.inspect_body_id is not None:
        ctl.inspect_body_id = int(args.inspect_body_id)
    write_control(ctl)
    print(json.dumps({"paused": ctl.paused, "speed": ctl.speed, "inspect_body_id": ctl.inspect_body_id}))
    return 0


def _cmd_accept(args: argparse.Namespace) -> int:
    from malecns.acceptance import run_acceptance

    report = run_acceptance(
        t_final_ms=float(args.t_final_ms),
        loop_ms=float(args.loop_ms),
        do_map=not bool(args.skip_map),
        do_simulate=not bool(args.skip_simulate),
        do_loop=not bool(args.skip_loop),
        spawn_viewer=bool(args.spawn_viewer),
    )
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
