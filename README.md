# GGP_ai

Experiments around general game playing on top of [Ludii](https://ludii.games/).
The pipeline produces self-play datasets from Ludii `.lud` games and converts
positions into a text form suitable for downstream supervision.

## Layout

```
ludii_build.py        Shared Ludii compile / classpath helpers (imported by
                      the launcher scripts below).
build/ludii/          Compiled Ludii classes, built once and reused by every
                      launcher. Sentinel-gated; delete to force a rebuild.

dataset_gen/          UCT self-play trial generation.
                      python3 dataset_gen/generate_dataset.py
position_text/        Serialize Ludii positions to ASCII + fact lists.
                      python3 position_text/render_positions.py {render,scan}

Ludii/                Upstream Ludii submodule (compiled into build/ludii).
LudiiPythonAI/        Upstream Java<->Python bridge for Ludii agents.
gavel/                Third-party evolutionary GGP code (see gavel/README.md).
dataset/              Default output directory for dataset_gen.
```

## Prerequisites

- JDK 17+ on `PATH` (`java`, `javac`).
- The Ludii submodule checked out:
  ```
  git submodule update --init --recursive
  ```

Each launcher script compiles the Ludii source tree once into `build/ludii/`
and reuses it on subsequent runs (and across launchers).

## Components

### `dataset_gen/`

Generates UCT self-play trials for every `.lud` game in the configured
sources. Each kept game gets a directory under `--out-dir` with a
`trials.txt` holding all trials (opening prefix + move indices + starting
RNG state) so playouts can be replayed deterministically.

See [`dataset_gen/README.md`](dataset_gen/README.md) for flags and examples.

### `position_text/`

Renders a Ludii game state as a textual fact list plus an ASCII board, used
to feed positions to language-model supervision. Two subcommands:

- `render <game.lud>` — print a single starting (or playout) position.
- `scan` — try the serializer on every game in the default sources and
  write a TSV summary (used to spot games where the renderer fails).

## Adding a new launcher

If you add another Java entrypoint that depends on Ludii, follow the same
pattern as the two existing scripts:

```python
import ludii_build as lb

ludii_build = lb.compile_ludii()
own_build = HERE / "build"
lb.compile_extras(own_build, sorted(str(p) for p in HERE.glob("*.java")),
                  [ludii_build])
cp = lb.runtime_classpath([own_build, ludii_build])
subprocess.call(["java", "-cp", cp, "YourMainClass", ...])
```
