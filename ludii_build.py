"""Shared Ludii build / classpath helpers used by the launcher scripts in
``dataset_gen/`` and ``position_text/``.

The Ludii source tree is compiled exactly once into a shared build dir
(``<project root>/build/ludii``) and reused across launchers. Each launcher
can additionally compile its own ``.java`` files into its own subdir on top
of that classpath.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
LUDII_ROOT = PROJECT_ROOT / "Ludii"
SHARED_BUILD = PROJECT_ROOT / "build" / "ludii"

# Headless Ludii modules only -- Player/PlayerDesktop drag in Apache Batik /
# javax.mail UI deps; Manager/ViewController are Swing-only.
LUDII_MODULES = ["Common", "Core", "Language", "Features", "AI"]

LUDII_LIB_JARS = [
    "Common/lib/json-20180813.jar",
    "Common/lib/Trove4j_ApacheCommonsRNG.jar",
    "Common/lib/jfreesvg-3.4.jar",
]

# Skip files/dirs that drag in unwanted deps (JUnit, Player UI, mail, batik...).
JAVA_PATH_SKIPS = ("/test/", "/junit/")


def detect_java_release() -> str:
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


def _collect_ludii_sources(ludii_root: pathlib.Path) -> list[str]:
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


def _ludii_lib_classpath(ludii_root: pathlib.Path) -> str:
    return os.pathsep.join(str(ludii_root / j) for j in LUDII_LIB_JARS)


def compile_ludii(build_dir: pathlib.Path = SHARED_BUILD,
                  ludii_root: pathlib.Path = LUDII_ROOT) -> pathlib.Path:
    """Compile the whole Ludii source tree into ``build_dir`` once.

    Ludii's grammar relies on reflection over ``game.*`` classes that aren't
    transitively reached from a small entrypoint, so a sourcepath-only
    compile produces a runtime NullPointerException. We compile the whole
    tree once, sentinel-gated, matching what ``ant`` would produce.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    sentinel = build_dir / ".compiled_ok"
    if sentinel.exists():
        return build_dir

    release = detect_java_release()
    sources = _collect_ludii_sources(ludii_root)
    if not sources:
        raise SystemExit(f"No Ludii sources found under {ludii_root}. "
                         "Did you `git submodule update --init --recursive`?")
    print(f"[compile] Ludii: {len(sources)} .java files "
          f"(release {release}) -> {build_dir}")
    _run_javac(release, build_dir, _ludii_lib_classpath(ludii_root), sources)
    sentinel.write_text("ok\n")
    return build_dir


def compile_extras(build_dir: pathlib.Path,
                   sources: list[str],
                   extra_classpath: list[pathlib.Path],
                   ludii_root: pathlib.Path = LUDII_ROOT) -> None:
    """Compile launcher-specific .java files on top of the Ludii classes."""
    if not sources:
        return
    build_dir.mkdir(parents=True, exist_ok=True)
    sentinel = build_dir / ".compiled_ok"
    if sentinel.exists() and all(
        sentinel.stat().st_mtime >= os.path.getmtime(s) for s in sources
    ):
        return
    cp = os.pathsep.join(
        [str(p) for p in extra_classpath] +
        [str(ludii_root / j) for j in LUDII_LIB_JARS]
    )
    release = detect_java_release()
    print(f"[compile] {len(sources)} files -> {build_dir}")
    _run_javac(release, build_dir, cp, sources)
    sentinel.write_text("ok\n")


def _run_javac(release: str, out_dir: pathlib.Path, classpath: str,
               sources: list[str]) -> None:
    # Pass sources via @argfile to dodge ARG_MAX on big trees.
    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False) as af:
        argfile = af.name
        for s in sources:
            af.write('"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"\n')
    try:
        subprocess.run(
            ["javac",
             "--release", release,
             "-encoding", "UTF-8",
             "-nowarn",
             "-Xlint:none",
             "-proc:none",
             # Ludii's compiler reflects on constructor parameter names.
             "-parameters",
             "-d", str(out_dir),
             "-cp", classpath,
             "@" + argfile],
            check=True,
        )
    finally:
        try:
            os.unlink(argfile)
        except OSError:
            pass


def runtime_classpath(build_dirs: list[pathlib.Path],
                      ludii_root: pathlib.Path = LUDII_ROOT) -> str:
    """Classpath for launching the JVM. Includes Ludii ``res/`` dirs because
    Ludii pulls some assets via ``getResourceAsStream``."""
    parts = [str(p) for p in build_dirs]
    for m in LUDII_MODULES:
        res_dir = ludii_root / m / "res"
        if res_dir.is_dir():
            parts.append(str(res_dir))
    for jar in LUDII_LIB_JARS:
        parts.append(str(ludii_root / jar))
    return os.pathsep.join(parts)


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
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


@contextlib.contextmanager
def manifest_file(paths: list[pathlib.Path]):
    """Write ``paths`` (one per line) to a tempfile; yield its path; clean up."""
    with tempfile.NamedTemporaryFile("w", suffix=".manifest", delete=False) as mf:
        manifest = mf.name
        for f in paths:
            mf.write(str(f) + "\n")
    try:
        yield manifest
    finally:
        try:
            os.unlink(manifest)
        except OSError:
            pass
