# Predicting Ludii game outcome from Qwen embeddings — experiment report

## Goal

Given precomputed `Qwen3-Embedding-4B` vectors over Ludii positions
(rules prefix + rendered position text → one pooled embedding), train a
small model to predict the eventual outcome of the game from a single
position.

## Setup

- **Embeddings:** D = 2560, L2-normalized, produced by
  `embeddings/embed_dataset.py` with `--prefix-lud` (each input embeds the
  full `.lud` rules followed by the rendered position).
- **Target:** `side_outcome ∈ {-1, 0, +1}` (mover-perspective signed
  outcome) or `p1_outcome` (player-1 perspective; selectable via
  `training/train.py --target`).
- **Model:** `training/model.py` `OutcomeMLP` — `LayerNorm → Linear(D,
  512) → GELU → Dropout → Linear(512, 256) → GELU → Dropout →
  Linear(256, 1)`. MSE loss, AdamW, early stopping on val MSE.
- **Per-game training:** `training/train.py` — one model per game.
- **Pooled training:** `training/train_all.py` — single model across all
  games, `--split-by {position,game}`.

The repo is set up for AMD/NVIDIA/CPU torch via uv extras
(`cpu` / `cuda` / `rocm`); training was run on ROCm 7.2.

## Findings

### 1. Sampling bug: K=1 always picked the terminal ply

`sample_plies` originally returned `[num_moves]` whenever K=1, so when
`target_per_game < n_trials` the sampler picked **only terminal
positions**. This produced two confounds:
- The rendered terminal board often visually reveals the winner, so the
  task degenerates to "read the winner off the final board."
- `mover` at the sampled ply correlated almost perfectly with the winner
  (`mover == winner` for ~all rows in the affected datasets), so
  `side_outcome` collapsed to near-constant +1 — not a "balanced game"
  problem, a sampling artifact.

Both are now fixed: `sample_plies` draws K plies uniformly at random
(without replacement) from `[0, num_moves]`. With the new sampler
`terminal_frac` per game is ~1–4%, both `side_outcome` and `p1_outcome`
are ~50/50, and `ply` is uniform over the trial.

### 2. Per-game results (random-ply sampling)

11 selected games, 10k positions/game (`embeddings/out/`) and again at
100k positions/game (`embeddings/out_100000/`). Target: `side_outcome`,
default MLP, AdamW lr 1e-3, wd 1e-2.

All games flatline at chance: test MSE ≈ baseline (predict-mean),
R² ≈ 0, sign-accuracy 0.50–0.55. Crucially this is **underfitting** —
train MSE also sits at ≈ var(y); the optimizer can't drive train loss
down despite ~3.1M params on 80k training rows (or 800k at 100k/game).
Bumping data 10× did not change the picture.

For reference, the same architecture trained on the **buggy
terminal-only** data reached R² 0.7–1.0 on most games — but that was the
model reading the winner off the final board, not learning evaluation.

### 3. Pooled (single model across many games)

Using `embeddings/out_all_games/` (1020 games × 1000 positions, terminal
sampling, pre-fix):

- **Split by position** (every game in train+val+test): overall R²
  +0.55, but per-game **median R² ≈ 0**. The model is learning the
  per-game mean outcome (game identity leaks through the rules prefix),
  not per-position structure.
- **Split by game** (20% games held out): overall R² +0.20, per-game
  **median R² −0.24** on unseen games. No useful generalization.

Pooled training was not re-run on the post-fix random-ply embeddings.

## Conclusion

A frozen Qwen embedding of `rules + single rendered position`, paired
with a small MLP and MSE loss, does **not** carry enough signal to
predict the eventual game outcome from a randomly chosen mid-game
position — neither per-game nor across games. The earlier high scores
were a sampling artifact (terminal-only positions).

This is a negative but informative result: the task as posed is hard
without lookahead, and a frozen general-purpose text embedder + tiny
classifier does not solve it.

## Possible next steps (not pursued here)

- **Train the embedding model end-to-end with a value head** — let the
  encoder optimize for outcome prediction directly, instead of relying
  on a frozen general-purpose embedding.
- Bias ply sampling toward late-game positions (outcome easier to
  predict; intermediate difficulty between terminal-only and uniform).
- Use a denser target (shallow MCTS rollout value at each position)
  rather than the binary final outcome.

## How to reproduce

```bash
# Per-game (one model per game in embeddings/out/)
uv run --extra rocm python -m training.train \
    --h5-dir embeddings/out --out-dir training/out \
    --target side_outcome

# Pooled (single model across all games)
uv run --extra rocm python -m training.train_all \
    --h5-dir embeddings/out_all_games --out-dir training/out_all \
    --split-by game
```

Artifacts land in `<out-dir>/<game_id>/{model.pt,metrics.json}` for
per-game and `<out-dir>/{model.pt,summary.json,per_game_metrics.json}`
for pooled.
