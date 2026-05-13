"""Library: parse trial .txt files and replay positions via the Java helper.

The trial format is the one written by ``dataset_gen/generate_dataset.py``:

    PATH=...
    LUDII_VERSION=...
    NUM_PLAYERS=N
    NUM_TRIALS=M
    ---TRIAL---
    RNG=<hex>
    OPENING_PLIES=<int>
    STATUS=<free text>
    RANKING=<float>,<float>,...
    MOVES=<int>,<int>,...

A "position" is the state after k moves (k=0 is the start state).
"""
from __future__ import annotations

import base64
import dataclasses
import pathlib
import subprocess
import sys
import tempfile
from typing import Iterable

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ludii_build as lb  # noqa: E402

POSITION_TEXT_DIR = PROJECT_ROOT / "position_text"


@dataclasses.dataclass
class TrialRecord:
    rng_hex: str
    opening_plies: int
    status: str
    ranking: list[float]
    moves: list[int]


@dataclasses.dataclass
class TrialFile:
    path: pathlib.Path
    lud_path: pathlib.Path
    num_players: int
    trials: list[TrialRecord]


@dataclasses.dataclass
class RenderedPosition:
    trial_idx: int
    ply: int
    mover: int      # 0 if no mover (terminal / no-state)
    terminal: int   # 0/1
    text: str


def parse_trial_file(path: pathlib.Path) -> TrialFile:
    """Parse a trial .txt produced by generate_dataset.py."""
    text = path.read_text()
    header: dict[str, str] = {}
    trials: list[TrialRecord] = []

    # Split: header | TRIAL | TRIAL | ...
    chunks = text.split("---TRIAL---\n")
    head = chunks[0]
    for line in head.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            header[k.strip()] = v.strip()

    for chunk in chunks[1:]:
        fields: dict[str, str] = {}
        for line in chunk.splitlines():
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
        moves_str = fields.get("MOVES", "")
        moves = [int(x) for x in moves_str.split(",") if x != ""]
        ranking_str = fields.get("RANKING", "")
        ranking = [float(x) for x in ranking_str.split(",") if x != ""]
        trials.append(TrialRecord(
            rng_hex=fields["RNG"],
            opening_plies=int(fields.get("OPENING_PLIES", "0")),
            status=fields.get("STATUS", ""),
            ranking=ranking,
            moves=moves,
        ))

    raw = pathlib.Path(header["PATH"]).expanduser()
    # Trial files written after the project-root patch store PATH= as a
    # repo-relative path. Older ones (with absolute PATH=) still work.
    lud_path = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
    return TrialFile(
        path=path,
        lud_path=lud_path,
        num_players=int(header.get("NUM_PLAYERS", "2")),
        trials=trials,
    )


def _build_classpath() -> str:
    """Compile Ludii + position_text + embeddings extras, return classpath."""
    ludii_build = lb.compile_ludii()

    # PositionSerializer lives in position_text/. We compile only the
    # serializer (not the CLIs) since those have main()s we don't need.
    pt_sources = [str(POSITION_TEXT_DIR / "PositionSerializer.java")]
    pt_build = POSITION_TEXT_DIR / "build"
    lb.compile_extras(pt_build, pt_sources, [ludii_build])

    own_sources = [str(p) for p in sorted(HERE.glob("*.java"))]
    own_build = HERE / "build"
    lb.compile_extras(own_build, own_sources, [ludii_build, pt_build])

    return lb.runtime_classpath([own_build, pt_build, ludii_build])


def render_positions(
    lud_path: pathlib.Path,
    tasks: list[tuple[int, TrialRecord, list[int]]],
    jvm_args: Iterable[str] = (),
    threads: int | None = None,
) -> list[RenderedPosition]:
    """Render positions for a list of (trial_idx, trial, plies) tasks.

    All tasks must use the same .lud (one JVM per game).
    """
    if not tasks:
        return []

    cp = _build_classpath()
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as tf:
        tasks_path = pathlib.Path(tf.name)
        for _trial_idx, trial, plies in tasks:
            tf.write(trial.rng_hex)
            tf.write("\t")
            tf.write(",".join(str(m) for m in trial.moves))
            tf.write("\t")
            tf.write(",".join(str(p) for p in plies))
            tf.write("\n")

    try:
        cmd = ["java", *jvm_args, "-cp", cp, "RenderTrials",
               "--lud", str(lud_path.resolve()),
               "--tasks", str(tasks_path)]
        if threads is not None:
            cmd += ["--threads", str(threads)]
        proc = subprocess.run(
            cmd, check=True, capture_output=True, text=False)
    finally:
        tasks_path.unlink(missing_ok=True)

    trial_idx_for_task = [t[0] for t in tasks]
    out: list[RenderedPosition] = []
    for raw in proc.stdout.splitlines():
        if not raw.startswith(b"POS "):
            continue
        parts = raw.split(b" ", 5)
        # POS <taskIdx> <ply> <mover> <terminal> <base64>
        task_idx = int(parts[1])
        ply = int(parts[2])
        mover = int(parts[3])
        terminal = int(parts[4])
        text = base64.b64decode(parts[5]).decode("utf-8")
        out.append(RenderedPosition(
            trial_idx=trial_idx_for_task[task_idx],
            ply=ply,
            mover=mover,
            terminal=terminal,
            text=text,
        ))
    if proc.stderr:
        # surface JVM warnings (e.g. Ludii startup chatter) but don't fail
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
    return out
