"""Train one MLP per game to regress side_outcome from Qwen embeddings.

Usage:
    uv run --extra rocm python -m training.train \
        --h5-dir embeddings/out --out-dir training/out
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

from training.data import load_game, make_splits
from training.model import OutcomeMLP


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            total += float(((pred - y) ** 2).sum().item())
            n += y.numel()
    return total / max(n, 1)


def metrics_for(pred: np.ndarray, y: np.ndarray, baseline: float) -> dict:
    mse = float(np.mean((pred - y) ** 2))
    var = float(np.var(y))
    r2 = float(1.0 - mse / var) if var > 0 else float("nan")
    sign_match = (np.sign(pred) == np.sign(y))
    sign_acc = float(np.mean(sign_match))
    return {
        "mse": mse,
        "baseline_mse": baseline,
        "r2": r2,
        "sign_accuracy": sign_acc,
    }


def train_one_game(
    h5_path: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    seed: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    target: str = "side_outcome",
) -> dict:
    game_id = h5_path.stem
    game_out = out_dir / game_id
    game_out.mkdir(parents=True, exist_ok=True)

    emb, y = load_game(h5_path, target=target)
    n, d = emb.shape
    unique, counts = np.unique(y, return_counts=True)
    target_dist = {str(float(u)): int(c) for u, c in zip(unique, counts)}
    print(f"[{game_id}] N={n} D={d} targets={target_dist}", flush=True)

    if float(np.var(y)) < 1e-12:
        result = {
            "game_id": game_id,
            "status": "degenerate_target",
            "n": int(n),
            "target_distribution": target_dist,
        }
        (game_out / "metrics.json").write_text(json.dumps(result, indent=2))
        print(f"[{game_id}] skipped: target has zero variance", flush=True)
        return result

    train_idx, val_idx, test_idx = make_splits(n, seed=seed)
    train_mean = float(np.mean(y[train_idx]))
    baseline_test = float(np.mean((y[test_idx] - train_mean) ** 2))
    baseline_val = float(np.mean((y[val_idx] - train_mean) ** 2))

    def make_loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(emb[idx]), torch.from_numpy(y[idx]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=False)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    torch.manual_seed(seed)
    model = OutcomeMLP(in_dim=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0
    trained_epochs = 0

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x, yb in train_loader:
            x = x.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * yb.numel()
            train_n += yb.numel()
        train_mse = train_loss_sum / max(train_n, 1)
        val_mse = eval_loss(model, val_loader, device)
        trained_epochs = epoch
        improved = val_mse < best_val - 1e-6
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f"[{game_id}] ep {epoch:3d}  train {train_mse:.4f}  "
              f"val {val_mse:.4f}  best {best_val:.4f}"
              f"  baseline_val {baseline_val:.4f}", flush=True)
        if epochs_no_improve >= patience:
            print(f"[{game_id}] early stop at epoch {epoch} "
                  f"(no val improvement for {patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(idx: np.ndarray) -> np.ndarray:
        model.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(idx), batch_size):
                chunk = idx[i:i + batch_size]
                x = torch.from_numpy(emb[chunk]).to(device, non_blocking=True)
                outs.append(model(x).cpu().numpy())
        return np.concatenate(outs, axis=0)

    train_pred = predict(train_idx)
    val_pred = predict(val_idx)
    test_pred = predict(test_idx)

    baseline_train = float(np.mean((y[train_idx] - train_mean) ** 2))
    result = {
        "game_id": game_id,
        "status": "ok",
        "n": int(n),
        "dim": int(d),
        "target_distribution": target_dist,
        "train_mean_label": train_mean,
        "epochs_trained": trained_epochs,
        "wall_seconds": round(time.time() - t0, 2),
        "device": str(device),
        "splits": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "train": metrics_for(train_pred, y[train_idx], baseline_train),
        "val": metrics_for(val_pred, y[val_idx], baseline_val),
        "test": metrics_for(test_pred, y[test_idx], baseline_test),
    }
    (game_out / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save({
        "model_state": model.state_dict(),
        "config": {"in_dim": d, "hidden": (512, 256), "dropout": 0.2},
        "game_id": game_id,
    }, game_out / "model.pt")
    print(f"[{game_id}] done: test MSE {result['test']['mse']:.4f} "
          f"(baseline {baseline_test:.4f}), "
          f"sign_acc {result['test']['sign_accuracy']:.3f}", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5-dir", type=pathlib.Path,
                    default=pathlib.Path("embeddings/out"))
    ap.add_argument("--out-dir", type=pathlib.Path,
                    default=pathlib.Path("training/out"))
    ap.add_argument("--games", nargs="+", default=None,
                    help="Subset of h5 stems (e.g. game_001). "
                         "Default: every *.h5 in --h5-dir.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--target", choices=("side_outcome", "p1_outcome"),
                    default="side_outcome")
    args = ap.parse_args()

    h5_dir: pathlib.Path = args.h5_dir
    if args.games:
        paths = [h5_dir / f"{g}.h5" for g in args.games]
    else:
        paths = sorted(h5_dir.glob("*.h5"))
    if not paths:
        raise SystemExit(f"no h5 files found in {h5_dir}")

    device = pick_device()
    print(f"[info] device: {device}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"[warn] missing: {p}", flush=True)
            continue
        res = train_one_game(
            p, args.out_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device,
            target=args.target,
        )
        summary.append(res)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[info] wrote summary to {args.out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
