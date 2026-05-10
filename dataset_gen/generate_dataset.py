#!/usr/bin/env python3
"""Generate a UCT self-play trial dataset for every .lud file in the
configured source directories.

Each game gets its own subdirectory under --out-dir containing trial_*.txt
files (Ludii's native trial format). A trial replays deterministically
given the original .lud and the saved RNG state, so it captures the full
sequence of moves needed later for position/outcome supervision.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
LUDII_ROOT = PROJECT_ROOT / "Ludii"

# Modules whose src/ trees we feed to javac and whose res/ trees we add to the
# runtime classpath (Ludii looks up .lud assets via getResourceAsStream).
# Headless modules only — Player/PlayerDesktop drag in Apache Batik / javax.mail
# UI deps that aren't needed for trial generation. Manager/ViewController are
# Swing-only so we skip them too.
LUDII_MODULES = ["Common", "Core", "Language", "Features", "AI"]

# Skip files/dirs that drag in unwanted deps (JUnit, Player UI, mail, batik...).
JAVA_PATH_SKIPS = (
    "/test/",
    "/junit/",
)

LUDII_LIB_JARS = [
    "Common/lib/json-20180813.jar",
    "Common/lib/Trove4j_ApacheCommonsRNG.jar",
    "Common/lib/jfreesvg-3.4.jar",
]

DEFAULT_SOURCE_DIRS = [
    LUDII_ROOT / "Common/res/lud/board",
    PROJECT_ROOT / "gavel/selected_games_cos098",
]


def build_runtime_classpath(ludii_root: pathlib.Path,
                            out_class_dir: pathlib.Path) -> str:
    """Classpath used to launch the JVM. Includes Ludii res/ dirs because
    Ludii pulls some assets via getResourceAsStream."""
    parts = [str(out_class_dir)]
    for m in LUDII_MODULES:
        res_dir = ludii_root / m / "res"
        if res_dir.is_dir():
            parts.append(str(res_dir))
    for jar in LUDII_LIB_JARS:
        parts.append(str(ludii_root / jar))
    return os.pathsep.join(parts)


def build_sourcepath(ludii_root: pathlib.Path) -> str:
    parts = []
    for m in LUDII_MODULES:
        src_dir = ludii_root / m / "src"
        if src_dir.is_dir():
            parts.append(str(src_dir))
    return os.pathsep.join(parts)


def _collect_java_sources(ludii_root: pathlib.Path) -> list[str]:
    files: list[str] = []
    for m in LUDII_MODULES:
        src_dir = ludii_root / m / "src"
        if src_dir.is_dir():
            for p in src_dir.rglob("*.java"):
                s = str(p)
                if any(skip in s for skip in JAVA_PATH_SKIPS):
                    continue
                files.append(s)
    return files


def compile_java(out_class_dir: pathlib.Path, ludii_root: pathlib.Path) -> None:
    """Compile our launcher together with the entire Ludii source tree.

    Ludii's grammar relies on reflection over `game.*` classes that aren't
    transitively reached from GenerateDataset.java, so a sourcepath-only
    compile produces a runtime NullPointerException. Compiling the whole
    tree once into out_class_dir avoids that and matches what ant would
    have produced.
    """
    out_class_dir.mkdir(parents=True, exist_ok=True)
    launcher_src = HERE / "GenerateDataset.java"
    sentinel = out_class_dir / ".compiled_ok"
    if sentinel.exists() and sentinel.stat().st_mtime >= launcher_src.stat().st_mtime:
        return

    java_release = _detect_java_release()
    classpath = os.pathsep.join(str(ludii_root / j) for j in LUDII_LIB_JARS)
    sources = _collect_java_sources(ludii_root) + [str(launcher_src)]
    print(f"[compile] {len(sources)} .java files (release {java_release}) "
          f"-> {out_class_dir}")

    # Pass the file list via @argfile to dodge ARG_MAX.
    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False) as af:
        argfile = af.name
        for s in sources:
            af.write('"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"\n')
    try:
        subprocess.run(
            ["javac",
             "--release", java_release,
             "-encoding", "UTF-8",
             "-nowarn",
             "-Xlint:none",
             "-proc:none",
             # Ludii's compiler reflects on constructor parameter names.
             "-parameters",
             "-d", str(out_class_dir),
             "-cp", classpath,
             "@" + argfile],
            check=True,
        )
    finally:
        try:
            os.unlink(argfile)
        except OSError:
            pass

    sentinel.write_text("ok\n")


def _detect_java_release() -> str:
    try:
        out = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return "17"
    # e.g. 'openjdk version "25.0.3" 2026-04-21'
    for tok in out.split():
        tok = tok.strip('"')
        if tok and tok[0].isdigit():
            major = tok.split(".", 1)[0]
            if major.isdigit():
                return major
    return "17"


def collect_lud_files(sources: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for src in sources:
        if not src.exists():
            print(f"[warn] source missing: {src}", file=sys.stderr)
            continue
        if src.is_file() and src.suffix == ".lud":
            files.append(src.resolve())
        else:
            files.extend(sorted(p.resolve() for p in src.rglob("*.lud")))
    # Dedup while preserving order.
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "dataset"),
                    help="Output root directory.")
    ap.add_argument("--num-games", type=int, default=10,
                    help="Number of trials to play per .lud file.")
    ap.add_argument("--num-playouts", type=int, default=1000,
                    help="MCTS iterations per move (UCT playouts).")
    ap.add_argument("--max-seconds", type=float, default=-1.0,
                    help="Optional wall-clock cap per move (seconds). "
                         "Negative = no cap (iterations only).")
    ap.add_argument("--move-limit", type=int, default=1000,
                    help="Max moves per trial before forced draw.")
    ap.add_argument("--source", action="append", type=pathlib.Path, default=None,
                    help="Override source dir/file (repeatable). "
                         "Defaults to the two configured directories.")
    ap.add_argument("--ludii-root", type=pathlib.Path, default=LUDII_ROOT,
                    help="Path to the Ludii repository (with built bin/ dirs).")
    ap.add_argument("--jvm-arg", action="append", default=[],
                    help="Extra JVM flag, e.g. --jvm-arg=-Xmx8g")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N games (debug).")
    args = ap.parse_args()

    sources = args.source if args.source else DEFAULT_SOURCE_DIRS
    lud_files = collect_lud_files(sources)
    if args.limit:
        lud_files = lud_files[: args.limit]
    if not lud_files:
        print("No .lud files found.", file=sys.stderr)
        return 2

    print(f"[info] {len(lud_files)} games -> {args.out_dir}")
    print(f"[info] {args.num_games} trials/game, {args.num_playouts} playouts/move")

    out_class_dir = HERE / "build"
    compile_java(out_class_dir, args.ludii_root)
    full_cp = build_runtime_classpath(args.ludii_root, out_class_dir)

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".manifest", delete=False) as mf:
        manifest_path = mf.name
        for f in lud_files:
            mf.write(str(f) + "\n")

    try:
        cmd = [
            "java",
            *args.jvm_arg,
            "-cp", full_cp,
            "GenerateDataset",
            "--out-dir", str(out_dir),
            "--num-games", str(args.num_games),
            "--num-playouts", str(args.num_playouts),
            "--max-seconds", str(args.max_seconds),
            "--move-limit", str(args.move_limit),
            "--manifest", manifest_path,
        ]
        print("[run]", " ".join(cmd))
        return subprocess.call(cmd)
    finally:
        try:
            os.unlink(manifest_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
