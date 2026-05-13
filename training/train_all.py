"""Train a single MLP across all games in embeddings/out_all_games.

The embeddings were produced with `--prefix-lud`, so each input vector
encodes both the game's rules and the position — a single shared model can
therefore be conditioned on the game implicitly via the embedding itself.

Degenerate games (constant side_outcome) carry no learning signal, so they
are filtered out before training and reported separately.

Usage:
    uv run --extra rocm python -m training.train_all \\
        --h5-dir embeddings/out_all_games --out-dir training/out_all
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.data_multi import (
    drop_degenerate_games,
    load_all_games,
    split_by_game,
    stratified_splits,
)
from training.model import OutcomeMLP


def pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def eval_predictions(
    model: nn.Module, emb: np.ndarray, idx: np.ndarray,
    batch_size: int, device: torch.device,
) -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            chunk = idx[i:i + batch_size]
            x = torch.from_numpy(emb[chunk]).to(device, non_blocking=True)
            outs.append(model(x).cpu().numpy())
    return np.concatenate(outs, axis=0)


def metrics_for(pred: np.ndarray, y: np.ndarray, mean_ref: float) -> dict:
    mse = float(np.mean((pred - y) ** 2))
    var = float(np.var(y))
    baseline = float(np.mean((y - mean_ref) ** 2))
    r2 = float(1.0 - mse / var) if var > 1e-12 else float("nan")
    sign_match = (np.sign(pred) == np.sign(y))
    return {
        "n": int(len(y)),
        "mse": mse,
        "baseline_mse": baseline,
        "r2": r2,
        "sign_accuracy": float(np.mean(sign_match)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5-dir", type=pathlib.Path,
                    default=pathlib.Path("embeddings/out_all_games"))
    ap.add_argument("--out-dir", type=pathlib.Path,
                    default=pathlib.Path("training/out_all"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 512],
                    help="MLP hidden widths.")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--split-by", choices=("position", "game"),
                    default="position",
                    help="'position' splits rows within each game; 'game' "
                         "holds out entire unseen games for val/test.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=None,
                    help="Default 0.1 for --split-by=position, "
                         "0.2 for --split-by=game.")
    args = ap.parse_args()
    if args.test_frac is None:
        args.test_frac = 0.2 if args.split_by == "game" else 0.1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[info] device: {device}", flush=True)

    emb, y, game_idx, game_ids = load_all_games(args.h5_dir)
    n_total, d = emb.shape
    emb, y, game_idx, kept_ids, dropped_ids = drop_degenerate_games(
        emb, y, game_idx, game_ids)
    print(f"[info] dropped {len(dropped_ids)} degenerate games "
          f"(kept {len(kept_ids)}), rows: {n_total} -> {emb.shape[0]}",
          flush=True)

    game_fold = None
    if args.split_by == "game":
        train_idx, val_idx, test_idx, game_fold = split_by_game(
            game_idx, n_games=len(kept_ids), seed=args.seed,
            val_frac=args.val_frac, test_frac=args.test_frac)
        n_train_g = int(np.sum(game_fold == "train"))
        n_val_g = int(np.sum(game_fold == "val"))
        n_test_g = int(np.sum(game_fold == "test"))
        print(f"[info] split by GAME: games train={n_train_g} "
              f"val={n_val_g} test={n_test_g}", flush=True)
    else:
        train_idx, val_idx, test_idx = stratified_splits(
            game_idx, n_games=len(kept_ids), seed=args.seed,
            val_frac=args.val_frac, test_frac=args.test_frac)
        print("[info] split by POSITION (every game appears in all folds)",
              flush=True)
    print(f"[info] splits: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}", flush=True)

    train_mean = float(np.mean(y[train_idx]))
    print(f"[info] train mean target = {train_mean:.4f}", flush=True)

    train_ds = TensorDataset(
        torch.from_numpy(emb[train_idx]), torch.from_numpy(y[train_idx]))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(emb[val_idx]),
                      torch.from_numpy(y[val_idx])),
        batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    model = OutcomeMLP(in_dim=d, hidden=tuple(args.hidden),
                       dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0
    trained_epochs = 0
    history: list[dict] = []

    t0 = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_sum, train_n = 0.0, 0
        ep_t0 = time.time()
        for x, yb in train_loader:
            x = x.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_sum += float(loss.item()) * yb.numel()
            train_n += yb.numel()
        train_mse = train_sum / max(train_n, 1)

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, yb in val_loader:
                x = x.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_sum += float(((model(x) - yb) ** 2).sum().item())
                val_n += yb.numel()
        val_mse = val_sum / max(val_n, 1)
        trained_epochs = epoch
        ep_dt = time.time() - ep_t0
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "val_mse": val_mse, "seconds": round(ep_dt, 1)})
        improved = val_mse < best_val - 1e-6
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f"[ep {epoch:3d}] train {train_mse:.4f}  val {val_mse:.4f}  "
              f"best {best_val:.4f}  ({ep_dt:.1f}s)", flush=True)
        if epochs_no_improve >= args.patience:
            print(f"[info] early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = eval_predictions(model, emb, test_idx, args.batch_size, device)
    overall = metrics_for(test_pred, y[test_idx], train_mean)

    per_game: list[dict] = []
    for g, gid in enumerate(kept_ids):
        mask = (game_idx[test_idx] == g)
        if not mask.any():
            continue
        gp = test_pred[mask]
        gy = y[test_idx][mask]
        per_game.append({"game_id": gid, **metrics_for(gp, gy, train_mean)})

    # Macro-averaged (per-game) MSE: each game weighted equally.
    valid_r2s = [p["r2"] for p in per_game if not np.isnan(p["r2"])]
    macro = {
        "n_games": len(per_game),
        "mean_mse": float(np.mean([p["mse"] for p in per_game])),
        "median_r2": float(np.median(valid_r2s)) if valid_r2s else float("nan"),
        "mean_r2": float(np.mean(valid_r2s)) if valid_r2s else float("nan"),
    }

    summary = {
        "device": str(device),
        "wall_seconds": round(time.time() - t0, 1),
        "epochs_trained": trained_epochs,
        "n_games_kept": len(kept_ids),
        "n_games_dropped": len(dropped_ids),
        "split_by": args.split_by,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "dim": int(d),
        "train_mean_label": train_mean,
        "test_overall": overall,
        "test_macro_avg": macro,
        "config": {
            "hidden": list(args.hidden), "dropout": args.dropout,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "seed": args.seed,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "per_game_metrics.json").write_text(
        json.dumps(per_game, indent=2))
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (args.out_dir / "dropped_games.json").write_text(
        json.dumps(dropped_ids, indent=2))
    if game_fold is not None:
        (args.out_dir / "game_folds.json").write_text(json.dumps(
            {gid: str(game_fold[g]) for g, gid in enumerate(kept_ids)},
            indent=2))
    torch.save({
        "model_state": model.state_dict(),
        "config": {"in_dim": d, "hidden": tuple(args.hidden),
                   "dropout": args.dropout},
        "game_ids": kept_ids,
    }, args.out_dir / "model.pt")

    print(f"[done] overall test MSE {overall['mse']:.4f} "
          f"(baseline {overall['baseline_mse']:.4f}, "
          f"R² {overall['r2']:+.3f}, sign_acc {overall['sign_accuracy']:.3f})",
          flush=True)
    print(f"[done] macro: mean_mse {macro['mean_mse']:.4f}, "
          f"median_r2 {macro['median_r2']:+.3f} across "
          f"{macro['n_games']} games", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
