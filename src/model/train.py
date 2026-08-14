"""CLI training script for ConvLSTM-UNet.

Loads offline mask bundle, builds tensors with temporal train/test split,
trains lightweight (no W&B, no augment, no scheduler), saves checkpoint
for inference. Designed for CPU (GitHub Actions) or GPU.

Usage:
    python -m src.model.train
    python -m src.model.train --data-dir data/offline_256_adaptive --epochs 200 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch

from src.model.convlstm import ConvLSTMUNet
from src.model.evaluate import DiceBCELoss, dice_score

SEASON_ORDER = ["S1", "S2", "S3"]


def load_offline(bundle_dir: str):
    """Load mask bundle: manifest.json + *_masks.npz per AOI."""
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        manifest = json.load(f)
    masks = {}
    for fn in os.listdir(bundle_dir):
        if not fn.endswith("_masks.npz"):
            continue
        nama = fn.replace("_masks.npz", "")
        z = np.load(os.path.join(bundle_dir, fn))
        masks[nama] = {(int(k.split("_")[0]), k.split("_")[1]): z[k] for k in z.files}
    print(f"Loaded {len(masks)} AOI dari {bundle_dir}")
    return masks, manifest


def seq_index(yr: int, s: str) -> int:
    return yr * 3 + SEASON_ORDER.index(s)


def build_tensor(masks: dict, lookback: int = 3):
    """Sliding windows with seasonal-gap check → (X, y, meta)."""
    X_list, y_list, meta = [], [], []
    for nama, frames in masks.items():
        keys = sorted(frames.keys(), key=lambda k: seq_index(*k))
        for i in range(len(keys) - lookback):
            win, target = keys[i : i + lookback], keys[i + lookback]
            idx = [seq_index(*k) for k in win + [target]]
            if idx != list(range(idx[0], idx[0] + lookback + 1)):
                continue
            X_list.append(np.stack([frames[k][np.newaxis] for k in win]))
            y_list.append(frames[target][np.newaxis])
            meta.append(
                {
                    "aoi": nama,
                    "input": win,
                    "target": target,
                    "target_year": target[0],
                }
            )
    X = np.stack(X_list)
    y = np.stack(y_list)
    print(f"Tensor: X{X.shape} y{y.shape} | {len(meta)} sample dari {len(masks)} AOI")
    return X, y, meta


def train_loop(
    model: ConvLSTMUNet,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
    criterion: DiceBCELoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    model_dir: str,
    logger: logging.Logger,
) -> tuple[dict, float]:
    history = {"train_loss": [], "train_dice": [], "test_dice": []}
    n = len(X_tr)
    best_te = -1.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            tr_pred = torch.sigmoid(model(X_tr))
            te_pred = torch.sigmoid(model(X_te))
            tr_dice = dice_score(tr_pred, y_tr).mean().item()
            te_dice = dice_score(te_pred, y_te).mean().item()

        history["train_loss"].append(epoch_loss)
        history["train_dice"].append(tr_dice)
        history["test_dice"].append(te_dice)

        if te_dice > best_te:
            best_te = te_dice
            ckpt_path = os.path.join(model_dir, "checkpoint.pth")
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch:3d} | loss={epoch_loss:.4f} | "
                f"tr={tr_dice:.4f} | te={te_dice:.4f}"
            )

    return history, best_te


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ConvLSTM-UNet shoreline model")
    ap.add_argument("--data-dir", default="data/offline_256_adaptive")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # ---------- logging ----------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(args.model_dir, f"train_{run_id}.log"))
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # ---------- seed / device ----------
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Data dir: {args.data_dir}")

    # ---------- Step 2: load bundle ----------
    masks, manifest = load_offline(args.data_dir)
    masks.pop("Titik_19", None)
    logger.info(f"AOI final: {list(masks.keys())}")

    # ---------- Step 3: build tensor + split ----------
    X, y, meta = build_tensor(masks)
    TRAIN_UNTIL = 2022
    tr = [i for i, m in enumerate(meta) if m["target_year"] <= TRAIN_UNTIL]
    te = [i for i, m in enumerate(meta) if m["target_year"] > TRAIN_UNTIL]

    X_train = torch.from_numpy(X[tr]).float().to(device)
    y_train = torch.from_numpy(y[tr]).float().to(device)
    X_test = torch.from_numpy(X[te]).float().to(device)
    y_test = torch.from_numpy(y[te]).float().to(device)
    logger.info(f"Train: {len(tr)} sample | Test: {len(te)} sample")

    # ---------- Step 4: persistence baseline ----------
    persist_pred = X_test[:, -1]
    dice_baseline = dice_score(persist_pred, y_test).mean().item()
    logger.info(f"Persistence baseline Dice: {dice_baseline:.4f}")

    # ---------- Step 5: init model + train ----------
    model = ConvLSTMUNet(in_ch=1, base_ch=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history, best_te = train_loop(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        criterion,
        optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        logger=logger,
    )

    # ---------- Step 6: save outputs ----------
    meta_out = {
        "IN_CH": 1,
        "LOOKBACK": 3,
        "PATCH_SIZE": 256,
        "TIMESTAMP": run_id,
    }
    meta_path = os.path.join(args.model_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f)

    hist_path = os.path.join(args.model_dir, f"training_history_{run_id}.npz")
    np.savez_compressed(hist_path, **history)

    logger.info(f"Model meta: {meta_path}")
    logger.info(f"History: {hist_path}")
    logger.info(f"Best checkpoint: {os.path.join(args.model_dir, 'checkpoint.pth')}")

    # ---------- summary ----------
    final_gap = history["train_dice"][-1] - history["test_dice"][-1]
    sep = "=" * 50
    logger.info(f"\n{sep}")
    logger.info(f"Final train Dice: {history['train_dice'][-1]:.4f}")
    logger.info(f"Final test Dice:  {history['test_dice'][-1]:.4f}")
    logger.info(f"Best test Dice:   {best_te:.4f}")
    logger.info(f"Persistence:      {dice_baseline:.4f}")
    logger.info(f"Train-test gap:   {final_gap:.4f}")
    logger.info(f"{sep}")


if __name__ == "__main__":
    main()
