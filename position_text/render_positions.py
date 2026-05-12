#!/usr/bin/env python3
"""Render a Ludii .lud position as text (fact list + ASCII board)."""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ludii_build as lb  # noqa: E402

DEFAULT_SCAN_SOURCES = [
    lb.LUDII_ROOT / "Common/res/lud/board",
    PROJECT_ROOT / "dataset_gen/resources/gavel_games",
]


def _build_classpath() -> str:
    ludii_build = lb.compile_ludii()
    own_sources = sorted(str(p) for p in HERE.glob("*.java"))
    own_build = HERE / "build"
    lb.compile_extras(own_build, own_sources, [ludii_build])
    return lb.runtime_classpath([own_build, ludii_build])


def cmd_render(args: argparse.Namespace) -> int:
    cp = _build_classpath()
    cmd = ["java", "-cp", cp,
           "RenderPositions", str(args.lud.resolve()),
           "--plies", str(args.plies),
           "--seed", str(args.seed),
           "--num", str(args.num)]
    if args.out is None:
        return subprocess.call(cmd)
    with open(args.out, "w") as f:
        return subprocess.call(cmd, stdout=f)


def cmd_scan(args: argparse.Namespace) -> int:
    cp = _build_classpath()
    sources = args.source if args.source else DEFAULT_SCAN_SOURCES
    luds = lb.collect_lud_files(sources)
    if args.limit:
        luds = luds[: args.limit]
    if not luds:
        print("no .lud files", file=sys.stderr)
        return 2

    with lb.manifest_file(luds) as manifest:
        cmd = ["java", "-cp", cp,
               "ScanPositions",
               "--manifest", manifest,
               "--plies", str(args.plies),
               "--seed", str(args.seed),
               "--out", str(args.out),
               "--threads", str(args.threads)]
        if args.samples_dir is not None:
            cmd += ["--samples-dir", str(args.samples_dir)]
        return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="Print a single position.")
    r.add_argument("lud", type=pathlib.Path, help="Path to a .lud file.")
    r.add_argument("--plies", type=int, default=0)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--num", type=int, default=1)
    r.add_argument("--out", type=pathlib.Path, default=None)
    r.set_defaults(func=cmd_render)

    s = sub.add_parser("scan", help="Try the serializer on every game.")
    s.add_argument("--source", action="append", type=pathlib.Path, default=None,
                   help="Override source dir/file (repeatable). Defaults: "
                        "Ludii/Common/res/lud/board + dataset_gen/resources/gavel_games.")
    s.add_argument("--plies", type=int, default=4)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", type=pathlib.Path, default=HERE / "scan.tsv")
    s.add_argument("--samples-dir", type=pathlib.Path, default=None,
                   help="If set, write the rendered position per game here.")
    s.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
