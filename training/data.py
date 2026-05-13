"""Load Qwen embeddings + side_outcome labels from one game's HDF5 file."""
from __future__ import annotations

import pathlib

import h5py
import numpy as np


def load_game(
    h5_path: pathlib.Path, target: str = "side_outcome",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [N, D] float32, target [N] float32).

    target:
      "side_outcome" — mover-perspective ±1/0 (as in the .h5 file).
      "p1_outcome"   — player-1 perspective ±1/0, computed from `ranking`:
                       rank 1 -> +1, rank (n+1)/2 -> 0, rank n -> -1.
                       Independent of which player is to move — recommended
                       when sampling biases `mover` toward winners.
    """
    with h5py.File(h5_path, "r") as f:
        emb = f["embedding"][:].astype(np.float32, copy=False)
        if target == "side_outcome":
            y = f["side_outcome"][:].astype(np.float32, copy=False)
        elif target == "p1_outcome":
            rk = f["ranking"][:].astype(np.float32, copy=False)
            np_ = int(f.attrs["num_players"])
            p1 = rk[:, 0]
            mid = (np_ + 1) / 2.0
            denom = (np_ - 1) / 2.0 if np_ > 1 else 1.0
            y = np.clip((mid - p1) / denom, -1.0, 1.0).astype(np.float32)
        else:
            raise ValueError(f"unknown target: {target!r}")
    return emb, y


def make_splits(
    n: int, seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random 80/10/10 row split (each position is from a distinct trial,
    so per-row shuffling does not leak across positions)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train, val, test
