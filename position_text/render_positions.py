#!/usr/bin/env python3
"""Render a Ludii .lud position as text (fact list + ASCII board).

Reuses the compiled Ludii classes produced by dataset_gen/generate_dataset.py,
and compiles this folder's .java files on top.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
LUDII_ROOT = PROJECT_ROOT / "Ludii"
DATASET_GEN = PROJECT_ROOT / "dataset_gen"
DATASET_BUILD = DATASET_GEN / "build"

LUDII_MODULES = ["Common", "Core", "Language", "Features", "AI"]
LUDII_LIB_JARS = [
    "Common/lib/json-20180813.jar",
    "Common/lib/Trove4j_ApacheCommonsRNG.jar",
    "Common/lib/jfreesvg-3.4.jar",
]


def ensure_ludii_compiled() -> None:
    """Run the dataset_gen compile step (idempotent) so we get Ludii classes."""
    sentinel = DATASET_BUILD / ".compiled_ok"
    if sentinel.exists():
        return
    print("[info] compiling Ludii via dataset_gen/generate_dataset.py ...")
    subprocess.run(
        [sys.executable, str(DATASET_GEN / "generate_dataset.py"), "--limit", "0"],
        check=False,
    )
    if not sentinel.exists():
        # Fall back to a minimal compile if the dataset script bailed early.
        raise SystemExit("Ludii compilation failed; run dataset_gen/generate_dataset.py once first.")


def compile_own(out_dir: pathlib.Path) -> None:
    own = sorted(str(p) for p in HERE.glob("*.java"))
    sentinel = out_dir / ".compiled_ok"
    if sentinel.exists() and all(
        sentinel.stat().st_mtime >= os.path.getmtime(s) for s in own
    ):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cp = os.pathsep.join([str(DATASET_BUILD)] + [
        str(LUDII_ROOT / j) for j in LUDII_LIB_JARS
    ])
    print(f"[compile] {len(own)} files -> {out_dir}")
    release = _detect_java_release()
    subprocess.run(
        ["javac", "--release", release, "-encoding", "UTF-8", "-nowarn",
         "-Xlint:none", "-proc:none", "-parameters", "-d", str(out_dir),
         "-cp", cp, *own],
        check=True,
    )


def _detect_java_release() -> str:
    try:
        out = subprocess.check_output(
            ["java", "-version"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return "17"
    for tok in out.split():
        tok = tok.strip('"')
        if tok and tok[0].isdigit():
            major = tok.split(".", 1)[0]
            if major.isdigit():
                return major
    return "17"
    sentinel.write_text("ok\n")


def runtime_classpath(out_dir: pathlib.Path) -> str:
    parts = [str(out_dir), str(DATASET_BUILD)]
    for m in LUDII_MODULES:
        res = LUDII_ROOT / m / "res"
        if res.is_dir():
            parts.append(str(res))
    for jar in LUDII_LIB_JARS:
        parts.append(str(LUDII_ROOT / jar))
    return os.pathsep.join(parts)


DEFAULT_SCAN_SOURCES = [
    LUDII_ROOT / "Common/res/lud/board",
    PROJECT_ROOT / "dataset_gen/resources/gavel_games",
]


def cmd_render(args: argparse.Namespace) -> int:
    ensure_ludii_compiled()
    out_dir = HERE / "build"
    compile_own(out_dir)
    cmd = ["java", "-cp", runtime_classpath(out_dir),
           "RenderPositions", str(args.lud.resolve()),
           "--plies", str(args.plies),
           "--seed", str(args.seed),
           "--num", str(args.num)]
    if args.out is None:
        return subprocess.call(cmd)
    with open(args.out, "w") as f:
        return subprocess.call(cmd, stdout=f)


def cmd_scan(args: argparse.Namespace) -> int:
    ensure_ludii_compiled()
    out_dir = HERE / "build"
    compile_own(out_dir)

    sources = args.source if args.source else DEFAULT_SCAN_SOURCES
    luds: list[pathlib.Path] = []
    for s in sources:
        if not s.exists():
            print(f"[warn] missing: {s}", file=sys.stderr)
            continue
        if s.is_file() and s.suffix == ".lud":
            luds.append(s.resolve())
        else:
            luds.extend(sorted(p.resolve() for p in s.rglob("*.lud")))
    seen = set()
    uniq = []
    for f in luds:
        if f not in seen: seen.add(f); uniq.append(f)
    if args.limit:
        uniq = uniq[: args.limit]
    if not uniq:
        print("no .lud files", file=sys.stderr); return 2

    with tempfile.NamedTemporaryFile("w", suffix=".manifest", delete=False) as mf:
        manifest = mf.name
        for f in uniq: mf.write(str(f) + "\n")
    try:
        cmd = ["java", "-cp", runtime_classpath(out_dir),
               "ScanPositions",
               "--manifest", manifest,
               "--plies", str(args.plies),
               "--seed", str(args.seed),
               "--out", str(args.out),
               "--threads", str(args.threads)]
        if args.samples_dir is not None:
            cmd += ["--samples-dir", str(args.samples_dir)]
        return subprocess.call(cmd)
    finally:
        try: os.unlink(manifest)
        except OSError: pass


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
