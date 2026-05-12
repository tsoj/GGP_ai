#!/usr/bin/env python3
"""Generate a UCT self-play trial dataset for every .lud file in the
configured source directories.

Each kept game writes a single <gameId>.txt file under --out-dir with all
trials concatenated, delimited by ---TRIAL--- markers. A trial replays
deterministically given the original .lud and the saved RNG state, so it
captures the full sequence of moves needed later for position/outcome
supervision.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ludii_build as lb  # noqa: E402

DEFAULT_SOURCE_DIRS = [
    lb.LUDII_ROOT / "Common/res/lud/board",
    HERE / "resources" / "gavel_games",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "dataset"),
                    help="Dataset root. Each kept game writes a single "
                         "<gameId>.txt file containing all trials for that "
                         "game type, delimited by ---TRIAL--- markers.")
    ap.add_argument("--failures-file", default=None,
                    help="TSV log of skipped games and reasons. "
                         "Defaults to <out-dir>/logs/failures.tsv.")
    ap.add_argument("--num-games", type=int, default=50,
                    help="Number of trials to play per .lud file.")
    ap.add_argument("--num-playouts", type=int, default=1000,
                    help="MCTS iterations per move (UCT playouts).")
    ap.add_argument("--max-seconds", type=float, default=-1.0,
                    help="Optional wall-clock cap per move (seconds). "
                         "Negative = no cap (iterations only).")
    ap.add_argument("--move-limit", type=int, default=500,
                    help="Max plies per trial; longer = forced draw.")
    ap.add_argument("--max-game-seconds", type=float, default=0.3,
                    help="Skip a game if at any point the average trial "
                         "length grows above this limit (wall clock).")
    ap.add_argument("--drawish-check-after", type=int, default=50,
                    help="After this many trials, skip a game if a single "
                         "outcome dominates (--drawish-threshold).")
    ap.add_argument("--drawish-threshold", type=float, default=0.9,
                    help="Outcome dominance fraction at which a game is "
                         "considered too drawish/biased to keep.")
    ap.add_argument("--opening-max-depth", type=int, default=250,
                    help="Cap on opening depth: random sampler grows the "
                         "opening length until it has enough unique start "
                         "positions, but never beyond this.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base RNG seed for opening enumeration.")
    ap.add_argument("--threads", type=int, default=None,
                    help="Worker threads on the Java side (one game per "
                         "task). Defaults to the JVM's available processors.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-game / per-trial logs above the progress "
                         "bar (default: bar only).")
    ap.add_argument("--append", action="store_true",
                    help="Keep existing --out-file / --failures-file and "
                         "append to them (default: truncate first).")
    ap.add_argument("--source", action="append", type=pathlib.Path, default=None,
                    help="Override source dir/file (repeatable). "
                         "Defaults to the two configured directories.")
    ap.add_argument("--ludii-root", type=pathlib.Path, default=lb.LUDII_ROOT,
                    help="Path to the Ludii repository (with built bin/ dirs).")
    ap.add_argument("--jvm-arg", action="append", default=[],
                    help="Extra JVM flag, e.g. --jvm-arg=-Xmx8g")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N games (debug).")
    args = ap.parse_args()

    sources = args.source if args.source else DEFAULT_SOURCE_DIRS
    lud_files = lb.collect_lud_files(sources)
    if args.limit:
        lud_files = lud_files[: args.limit]
    if not lud_files:
        print("No .lud files found.", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(args.out_dir).resolve()
    failures_file = pathlib.Path(
        args.failures_file if args.failures_file
        else out_dir / "logs" / "failures.tsv"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_file.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        failures_file.write_text("")

    print(f"[info] {len(lud_files)} games -> {out_dir}")
    print(f"[info] failures log -> {failures_file}")
    print(f"[info] {args.num_games} trials/game, {args.num_playouts} playouts/move,"
          f" move_limit={args.move_limit}, max_game_seconds={args.max_game_seconds}")

    ludii_build = lb.compile_ludii(ludii_root=args.ludii_root)
    own_sources = sorted(str(p) for p in HERE.glob("*.java"))
    own_build = HERE / "build"
    lb.compile_extras(own_build, own_sources, [ludii_build],
                      ludii_root=args.ludii_root)
    full_cp = lb.runtime_classpath([own_build, ludii_build],
                                   ludii_root=args.ludii_root)

    with lb.manifest_file(lud_files) as manifest_path:
        cmd = [
            "java",
            *args.jvm_arg,
            "-cp", full_cp,
            "GenerateDataset",
            "--out-dir", str(out_dir),
            "--failures-file", str(failures_file),
            "--num-games", str(args.num_games),
            "--num-playouts", str(args.num_playouts),
            "--max-seconds", str(args.max_seconds),
            "--move-limit", str(args.move_limit),
            "--max-game-seconds", str(args.max_game_seconds),
            "--drawish-check-after", str(args.drawish_check_after),
            "--drawish-threshold", str(args.drawish_threshold),
            "--opening-max-depth", str(args.opening_max_depth),
            "--seed", str(args.seed),
            "--manifest", manifest_path,
        ]
        if args.threads is not None:
            cmd += ["--threads", str(args.threads)]
        if args.verbose:
            cmd += ["--verbose"]
        print("[run]", " ".join(cmd))
        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
