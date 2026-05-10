# Dataset generation

UCT self-play trials for one or more Ludii `.lud` games. Each kept game gets
its own subdirectory under `--out-dir` with a `trials.txt` holding all trials
(opening prefix + move indices, plus the starting RNG state) so playouts can
be replayed deterministically against the same Ludii build. Games that fail
to load, aren't supported by UCT, run too slowly, produce too few unique
openings, or whose first batch of outcomes is too lopsided get skipped and
logged to `failures.tsv`.

Generation runs across multiple worker threads (one game per task) and shows
a tqdm-style progress bar; per-game completion logs are interleaved above
the bar.

## Prerequisites

- JDK 17+ on `PATH` (`java`, `javac`).
- The `Ludii/` submodule checked out at the repo root. If you cloned without
  submodules:
  ```
  git submodule update --init --recursive
  ```

The Python launcher compiles the Ludii source tree plus
`GenerateDataset.java` / `OpeningGenerator.java` into `dataset_gen/build/` on
first run (and re-uses it on subsequent runs).

## Quick start

From the repo root:

```
python3 dataset_gen/generate_dataset.py
```

Defaults: 50 trials/game, 1000 UCT playouts/move, output to `./dataset/`,
sources are `Ludii/Common/res/lud/board/` and
`dataset_gen/resources/gavel_games/`.

## Common knobs

```
--out-dir PATH              Where to write per-game subdirs (default: ./dataset)
--failures-file PATH        TSV of skipped games (default: <out-dir>/failures.tsv)
--num-games N               Trials per game (default: 50)
--num-playouts N            UCT iterations per move (default: 1000)
--max-seconds S             Wall-clock cap per move (default: no cap)
--move-limit N              Max plies; longer = forced draw (default: 500)
--max-game-seconds S        Skip game if its first trial exceeds S (default: 0.5)
--drawish-check-after N     After N trials, skip if one outcome dominates
--drawish-threshold F       Dominance fraction for the drawish check (default: 0.9)
--opening-max-depth N       Cap on random-opening enumeration depth (default: 8)
--seed N                    Base RNG seed (default: 42)
--threads N                 Java worker threads (default: available processors)
--verbose                   Print per-game / per-trial logs above the bar
--source PATH               Override source dir/file (repeatable)
--limit N                   Only process the first N games (debug)
--jvm-arg FLAG              Extra JVM flag, e.g. --jvm-arg=-Xmx8g
--append                    Don't truncate failures.tsv at the start
```

## Examples

Single-threaded smoke test on a handful of games:

```
python3 dataset_gen/generate_dataset.py --limit 5 --threads 1
```

Bigger run with more memory and a custom output dir:

```
python3 dataset_gen/generate_dataset.py \
    --out-dir /data/ludii_trials \
    --num-games 200 \
    --num-playouts 2000 \
    --jvm-arg=-Xmx16g
```

Just the gavel games:

```
python3 dataset_gen/generate_dataset.py \
    --source dataset_gen/resources/gavel_games
```

## Output layout

```
<out-dir>/
  <game-id>/
    trials.txt        # header + one ---TRIAL--- record per playout
  failures.tsv        # <ludPath>\t<reason>\t<detail>
```

`trials.txt` header fields: `PATH`, `LUDII_VERSION`, `NUM_PLAYERS`,
`NUM_TRIALS`. Each record contains the starting `RNG` state (hex),
`OPENING_PLIES`, final `STATUS`, per-player `RANKING`, and the comma-
separated `MOVES` (indices into `game.moves(context).moves()` at each ply).
