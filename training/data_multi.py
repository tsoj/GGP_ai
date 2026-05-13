"""Load embeddings + side_outcome across many per-game HDF5 files into one
contiguous array, with a per-game stratified train/val/test split."""
from __future__ import annotations

import pathlib
import time

import h5py
import numpy as np


def load_all_games(
    h5_dir: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (emb [Ntot, D] float32, y [Ntot] float32,
                game_idx [Ntot] int32, game_ids list[str]).

    Files are sorted by name; `game_idx[i]` is the index into `game_ids` for
    the game that row `i` belongs to.
    """
    paths = sorted(h5_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no h5 files in {h5_dir}")

    embs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    game_idx_parts: list[np.ndarray] = []
    game_ids: list[str] = []

    t0 = time.time()
    for gi, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            e = f["embedding"][:].astype(np.float32, copy=False)
            y = f["side_outcome"][:].astype(np.float32, copy=False)
        embs.append(e)
        ys.append(y)
        game_idx_parts.append(np.full(e.shape[0], gi, dtype=np.int32))
        game_ids.append(p.stem)
        if (gi + 1) % 100 == 0:
            print(f"[load] {gi + 1}/{len(paths)} games "
                  f"({time.time() - t0:.1f}s elapsed)", flush=True)

    emb = np.concatenate(embs, axis=0)
    y = np.concatenate(ys, axis=0)
    game_idx = np.concatenate(game_idx_parts, axis=0)
    print(f"[load] done: N={emb.shape[0]}, D={emb.shape[1]}, "
          f"games={len(game_ids)}, {time.time() - t0:.1f}s", flush=True)
    return emb, y, game_idx, game_ids


def drop_degenerate_games(
    emb: np.ndarray, y: np.ndarray, game_idx: np.ndarray,
    game_ids: list[str], var_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Remove games whose side_outcome has (near-)zero variance.

    Returns the filtered arrays plus the list of dropped game ids. The
    `game_idx` values are reindexed to be contiguous over the surviving
    games (0..len(kept_ids)-1).
    """
    keep_game = np.ones(len(game_ids), dtype=bool)
    for g in range(len(game_ids)):
        rows = np.flatnonzero(game_idx == g)
        if rows.size == 0 or float(np.var(y[rows])) <= var_eps:
            keep_game[g] = False
    kept_ids = [gid for g, gid in enumerate(game_ids) if keep_game[g]]
    dropped_ids = [gid for g, gid in enumerate(game_ids) if not keep_game[g]]
    remap = -np.ones(len(game_ids), dtype=np.int32)
    remap[keep_game] = np.arange(keep_game.sum(), dtype=np.int32)
    keep_row = keep_game[game_idx]
    return (emb[keep_row], y[keep_row], remap[game_idx[keep_row]],
            kept_ids, dropped_ids)


def split_by_game(
    game_idx: np.ndarray, n_games: int, seed: int = 0,
    val_frac: float = 0.1, test_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split *games* (not positions). All rows of a game land in one fold.

    Tests whether the model generalizes to games it has never seen. Returns
    (train_rows, val_rows, test_rows, game_fold) where game_fold[g] is one
    of 'train'/'val'/'test'.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_games)
    n_test = int(round(n_games * test_frac))
    n_val = int(round(n_games * val_frac))
    test_games = perm[:n_test]
    val_games = perm[n_test:n_test + n_val]
    train_games = perm[n_test + n_val:]
    fold = np.empty(n_games, dtype=object)
    fold[train_games] = "train"
    fold[val_games] = "val"
    fold[test_games] = "test"
    in_set = lambda gs: np.isin(game_idx, gs)
    return (np.flatnonzero(in_set(train_games)),
            np.flatnonzero(in_set(val_games)),
            np.flatnonzero(in_set(test_games)),
            fold)


def stratified_splits(
    game_idx: np.ndarray, n_games: int, seed: int = 0,
    val_frac: float = 0.1, test_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-game random split so every game contributes to train/val/test."""
    rng = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    val: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for g in range(n_games):
        rows = np.flatnonzero(game_idx == g)
        rng.shuffle(rows)
        n = len(rows)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test.append(rows[:n_test])
        val.append(rows[n_test:n_test + n_val])
        train.append(rows[n_test + n_val:])
    return (np.concatenate(train), np.concatenate(val), np.concatenate(test))
