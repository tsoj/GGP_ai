#!/usr/bin/env python3
"""Precompute Qwen3-Embedding-4B embeddings for sampled positions of selected
Ludii games. One HDF5 file per game (all trials of that game type).

Pipeline:
  1. Parse trial .txt files.
  2. For each trial, pick K positions uniformly across plies [0, len(moves)].
     K is chosen so that the total positions per game is close to
     ``--target-per-game`` (clamped to >=1 per trial).
  3. Replay positions in the JVM (RenderTrials.java) -> serialized text.
  4. Embed texts with vLLM (Qwen/Qwen3-Embedding-4B) in batches.
  5. Write <out-dir>/<gameId>.h5 with per-position embeddings + labels.

Usage on the target GPU box (single file generates everything):
    python embeddings/embed_dataset.py \
        --target-per-game 4096 \
        --out-dir embeddings/out \
        --model Qwen/Qwen3-Embedding-4B
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import render_trials as rt  # noqa: E402

DEFAULT_GAMES = [
    "game_001_fit1.000_seed_1",
    "game_003_fit1.000_seed_1",
    "game_014_fit1.000_seed_1",
    "game_019_fit1.000_seed_1",
    "game_052_fit1.000_seed_3",
]
DATASET_DIR = PROJECT_ROOT / "dataset"


def resolve_game_id(game_id: str) -> pathlib.Path:
    """Resolve a game id to its trial .txt path.

    Accepts either:
      - the exact stem (e.g. "Aboyne" or "game_001_fit1.000_seed_1"); or
      - a short numeric form for gavel games (e.g. "001" -> "game_001_*").
    """
    direct = DATASET_DIR / f"{game_id}.txt"
    if direct.exists():
        return direct
    matches = sorted(DATASET_DIR.glob(f"game_{game_id}_*.txt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous game id '{game_id}': {matches}")
    raise FileNotFoundError(f"no trial file for game id '{game_id}' in {DATASET_DIR}")


def discover_all_game_ids() -> list[str]:
    """Return every trial-file stem in dataset/."""
    return sorted(p.stem for p in DATASET_DIR.glob("*.txt"))


def sample_plies(num_moves: int, k: int) -> list[int]:
    """K uniformly spaced ply indices in [0, num_moves] (start + after-k-moves)."""
    if num_moves < 0:
        raise ValueError(num_moves)
    if k <= 0:
        return []
    if k == 1:
        return [num_moves]
    pts = np.linspace(0, num_moves, k).round().astype(int).tolist()
    # dedupe while preserving order
    seen: set[int] = set()
    out: list[int] = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def signed_outcome(ranking: list[float], mover: int, num_players: int) -> float:
    """+1 if best rank (1.0), -1 if worst rank, 0 for middle / draws.

    For 2 players: rank 1.0 -> +1 win, 1.5 -> 0 draw, 2.0 -> -1 loss.
    For N players we map by position relative to the midpoint.
    """
    if mover < 1 or mover > len(ranking) or not ranking:
        return 0.0
    r = ranking[mover - 1]
    mid = (num_players + 1) / 2.0
    # Sign so that rank=1.0 (best) -> +1, worst -> -1.
    raw = mid - r
    denom = (num_players - 1) / 2.0 if num_players > 1 else 1.0
    return float(np.clip(raw / denom, -1.0, 1.0))


def process_game(
    game_id: str,
    target_per_game: int,
    out_dir: pathlib.Path,
    model_name: str,
    embed_batch: int,
    embedder,
    debug_print_texts: int = 0,
    render_threads: int | None = None,
) -> pathlib.Path:
    trial_path = resolve_game_id(game_id)
    game_id = trial_path.stem  # canonical (in case user passed a short form)
    tf = rt.parse_trial_file(trial_path)
    print(f"[{game_id}] {trial_path.name}: {len(tf.trials)} trials, "
          f"{tf.num_players} players, lud={tf.lud_path.name}", flush=True)

    if not tf.lud_path.exists():
        raise FileNotFoundError(f"lud not found: {tf.lud_path}")

    # Decide K per trial so total ~= target_per_game.
    n_trials = len(tf.trials)
    k_per_trial = max(1, round(target_per_game / max(1, n_trials)))
    tasks: list[tuple[int, rt.TrialRecord, list[int]]] = []
    for ti, trial in enumerate(tf.trials):
        plies = sample_plies(len(trial.moves), k_per_trial)
        if plies:
            tasks.append((ti, trial, plies))
    total_positions = sum(len(t[2]) for t in tasks)
    print(f"[{game_id}] sampling {k_per_trial}/trial -> {total_positions} positions",
          flush=True)

    t0 = time.time()
    rendered = rt.render_positions(tf.lud_path, tasks, threads=render_threads)
    print(f"[{game_id}] rendered {len(rendered)} positions in "
          f"{time.time() - t0:.1f}s", flush=True)

    # Embed
    texts = [r.text for r in rendered]
    if debug_print_texts > 0:
        n_show = min(debug_print_texts, len(rendered))
        print(f"[{game_id}] --- debug: first {n_show} of {len(rendered)} "
              f"texts fed to embedder ---", flush=True)
        for i in range(n_show):
            r = rendered[i]
            print(f"===== [{game_id}] idx={i}  trial={r.trial_idx}  "
                  f"ply={r.ply}  mover={r.mover}  terminal={r.terminal} =====",
                  flush=True)
            print(r.text, flush=True)
        print(f"[{game_id}] --- end debug texts ---", flush=True)
    print(f"[{game_id}] embedding {len(texts)} texts in batches of {embed_batch}...",
          flush=True)
    t0 = time.time()
    embeddings = embedder(texts, batch_size=embed_batch)
    print(f"[{game_id}] embedded in {time.time() - t0:.1f}s, "
          f"dim={embeddings.shape[1]}", flush=True)

    # Labels
    n = len(rendered)
    trial_idx = np.array([r.trial_idx for r in rendered], dtype=np.int32)
    ply = np.array([r.ply for r in rendered], dtype=np.int32)
    mover = np.array([r.mover for r in rendered], dtype=np.int32)
    terminal = np.array([r.terminal for r in rendered], dtype=np.uint8)
    ranking = np.zeros((n, tf.num_players), dtype=np.float32)
    side_outcome = np.zeros(n, dtype=np.float32)
    for i, r in enumerate(rendered):
        rk = tf.trials[r.trial_idx].ranking
        for j in range(min(tf.num_players, len(rk))):
            ranking[i, j] = rk[j]
        side_outcome[i] = signed_outcome(rk, r.mover, tf.num_players)

    out_path = out_dir / f"{game_id}.h5"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_h5(out_path, tf, model_name, embeddings, trial_idx, ply, mover,
              terminal, ranking, side_outcome, texts)
    print(f"[{game_id}] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    return out_path


def _write_h5(path, tf, model_name, emb, trial_idx, ply, mover, terminal,
              ranking, side_outcome, texts):
    with h5py.File(path, "w") as f:
        f.attrs["model"] = model_name
        f.attrs["game_id"] = tf.path.stem
        f.attrs["lud_path"] = str(tf.lud_path)
        f.attrs["num_players"] = tf.num_players
        f.attrs["num_trials"] = len(tf.trials)
        f.create_dataset("embedding", data=emb, compression="gzip",
                         compression_opts=4)
        f.create_dataset("trial_idx", data=trial_idx)
        f.create_dataset("ply", data=ply)
        f.create_dataset("mover", data=mover)
        f.create_dataset("terminal", data=terminal)
        f.create_dataset("ranking", data=ranking)
        f.create_dataset("side_outcome", data=side_outcome)
        # Per-trial summaries (handy when training).
        trial_ranking = np.zeros((len(tf.trials), tf.num_players), dtype=np.float32)
        trial_num_moves = np.zeros(len(tf.trials), dtype=np.int32)
        statuses = []
        for i, t in enumerate(tf.trials):
            for j in range(min(tf.num_players, len(t.ranking))):
                trial_ranking[i, j] = t.ranking[j]
            trial_num_moves[i] = len(t.moves)
            statuses.append(t.status)
        f.create_dataset("trial_ranking", data=trial_ranking)
        f.create_dataset("trial_num_moves", data=trial_num_moves)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("trial_status", data=np.array(statuses, dtype=object),
                         dtype=dt)
        f.create_dataset("text", data=np.array(texts, dtype=object), dtype=dt,
                         compression="gzip", compression_opts=4)


def make_vllm_embedder(model_name: str, max_model_len: int | None,
                       gpu_memory_utilization: float, dtype: str):
    """Return a function texts -> np.ndarray[N, D] using vLLM offline embed."""
    # Lazy import: vLLM is an optional `embed` extra; module must still load
    # when --dummy-embedder is used without it installed.
    from vllm import LLM

    kwargs = dict(model=model_name,
                  gpu_memory_utilization=gpu_memory_utilization,
                  dtype=dtype, trust_remote_code=True)
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    llm = LLM(**kwargs)

    def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
        # vLLM does internal scheduling; we still chunk to bound peak memory.
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            outs = llm.embed(chunk)
            for o in outs:
                v = np.asarray(o.outputs.embedding, dtype=np.float32)
                all_vecs.append(v)
        return np.stack(all_vecs, axis=0)

    return embed


def make_dummy_embedder(dim: int = 16):
    """Hash-based fake embedder for pipeline testing without a GPU."""
    def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            rng = np.random.default_rng(h % (2**32))
            out[i] = rng.standard_normal(dim).astype(np.float32)
        return out
    return embed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+", default=None,
                    help="Game numbers (zero-padded). Default: "
                         + " ".join(DEFAULT_GAMES) + ". "
                         "Ignored if --all-games is set.")
    ap.add_argument("--all-games", action="store_true",
                    help="Process every game with a trial file in dataset/.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip games whose <out-dir>/game_<num>.h5 already "
                         "exists. Useful when resuming with --all-games.")
    ap.add_argument("--target-per-game", type=int, default=4096,
                    help="Approximate total positions per game (split evenly "
                         "across trials, min 1 per trial).")
    ap.add_argument("--out-dir", type=pathlib.Path,
                    default=HERE / "out",
                    help="Where to write <gameId>.h5 files.")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--embed-batch", type=int, default=64,
                    help="Texts per vLLM embed() call (chunking).")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="Optional override; otherwise vLLM picks from config.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dtype", default="bfloat16",
                    help="vLLM dtype (bfloat16 / float16 / auto).")
    ap.add_argument("--dummy-embedder", action="store_true",
                    help="Skip vLLM; produce random vectors. For pipeline tests.")
    ap.add_argument("--render-threads", type=int, default=None,
                    help="JVM worker threads for trial replay. "
                         "Default: max(1, available_cores // 2).")
    ap.add_argument("--debug-print-texts", type=int, default=0, metavar="N",
                    help="Print the first N rendered texts (verbatim) per game "
                         "before embedding. For inspecting what the model sees.")
    args = ap.parse_args()

    if args.dummy_embedder:
        embedder = make_dummy_embedder()
        model_name = "dummy"
    else:
        embedder = make_vllm_embedder(
            args.model, args.max_model_len,
            args.gpu_memory_utilization, args.dtype)
        model_name = args.model

    render_threads = (args.render_threads if args.render_threads is not None
                      else max(1, (os.cpu_count() or 2) // 2))

    if args.all_games:
        games = discover_all_game_ids()
        print(f"[info] --all-games: found {len(games)} games in {DATASET_DIR}",
              flush=True)
    else:
        games = args.games if args.games is not None else DEFAULT_GAMES

    for g in games:
        out_path = args.out_dir / f"{g}.h5"
        if args.skip_existing and out_path.exists():
            print(f"[{g}] skip (exists: {out_path})", flush=True)
            continue
        process_game(g, args.target_per_game, args.out_dir,
                     model_name, args.embed_batch, embedder,
                     debug_print_texts=args.debug_print_texts,
                     render_threads=render_threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
