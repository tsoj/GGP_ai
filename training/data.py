"""Load Qwen embeddings + side_outcome labels from one game's HDF5 file."""
from __future__ import annotations

import pathlib

import h5py
import numpy as np


def load_game(h5_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [N, D] float32, side_outcome [N] float32)."""
    with h5py.File(h5_path, "r") as f:
        emb = f["embedding"][:].astype(np.float32, copy=False)
        y = f["side_outcome"][:].astype(np.float32, copy=False)
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
