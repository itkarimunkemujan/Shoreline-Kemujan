# Lightweight Model Notebook — `lightweight_model.ipynb`

> Copy-paste setiap cell ke notebook Colab baru. Jangan edit langsung `.ipynb` biar gak conflict.

---

## Cell 1 — Mount Drive + GEE Auth (sama persis `final_model`)

```python
# ============================================================
# CELL 1 — Mount Drive + authenticate GEE + shared constants
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

!pip install geemap contextily shapely scikit-image -q

import os
import ee

GEE_PROJECT = "gen-lang-client-0412358476"
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
print("EE initialized |", GEE_PROJECT)

PATCH_SIZE = 256
SCALE_M = 10
LOOKBACK = 3

MODEL_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/models"
LOG_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
print("MODEL_DIR:", MODEL_DIR)
print("LOG_DIR  :", LOG_DIR)
```

---

## Cell 2 — Load X_train/X_test langsung dari Drive

```python
# ============================================================
# CELL 2 — Load X_train/X_test langsung dari Drive tensors.npz
# ============================================================
import json
import numpy as np
import torch

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

FINAL_DATA_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/final_data"
_tz = np.load(os.path.join(FINAL_DATA_DIR, "tensors.npz"))
with open(os.path.join(FINAL_DATA_DIR, "params.json")) as f:
    _params = json.load(f)

LOOKBACK = _params["LOOKBACK"]
ROLLOUT_STEPS = _params["ROLLOUT_STEPS"]
PATCH_SIZE = _params["PATCH_SIZE"]

# mask-only: buang 6 static channel
_X = _tz["X"][:, :, 0:1].copy()
_y = _tz["y"][:, 0].copy()  # (N, 1, 256, 256) — buang ROLLOUT_STEPS dim
_tr, _te = _tz["tr"], _tz["te"]

X_train = torch.from_numpy(_X[_tr]).float().to(device)
y_train = torch.from_numpy(_y[_tr]).float().to(device)
X_test  = torch.from_numpy(_X[_te]).float().to(device)
y_test  = torch.from_numpy(_y[_te]).float().to(device)

IN_CH = 1
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
```

---

## Cell 3 — Model + Loss + Persistence Baseline

```python
# ============================================================
# CELL 3 — Persistence baseline + ConvLSTM-UNet + loss
# ============================================================
import torch.nn as nn

def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2*inter + eps) / (pred.sum(1) + target.sum(1) + eps)

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        inter = (prob.flatten(1) * target.flatten(1)).sum(1)
        dice = 1 - (2*inter + 1e-6) / (prob.flatten(1).sum(1) + target.flatten(1).sum(1) + 1e-6)
        return self.bce_weight * self.bce(logit, target) + (1 - self.bce_weight) * dice.mean()

# persistence baseline
persist_pred = X_test[:, -1]
dice_baseline = dice_score(persist_pred, y_test).mean().item()
print(f"Persistence baseline Dice: {dice_baseline:.4f}")

# ----- ConvLSTM-UNet (lightweight, 85k params) -----
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch  # ← WAJIB: dipakai di init_hidden
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4*hidden_ch, kernel_size, padding=kernel_size//2)
    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c_next), c_next
    def init_hidden(self, batch, H, W, device):
        return (torch.zeros(batch, self.hidden_ch, H, W, device=device),
                torch.zeros(batch, self.hidden_ch, H, W, device=device))

class ConvLSTMUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.BatchNorm2d(base_ch*2), nn.ReLU())
        self.pool, self.clstm = nn.MaxPool2d(2), ConvLSTMCell(base_ch*2, base_ch*2, 3)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = nn.Sequential(nn.Conv2d(base_ch*3, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.head = nn.Conv2d(base_ch, 1, 1)
    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c, enc1_last = None, None, None
        for t in range(T):
            enc1 = self.enc1(x[:, t])
            enc2 = self.enc2(self.pool(enc1))
            if h is None:
                h, c = self.clstm.init_hidden(B, enc2.shape[2], enc2.shape[3], x.device)
            h, c = self.clstm(enc2, h, c)
            enc1_last = enc1
        return self.head(self.dec(torch.cat([self.up(h), enc1_last], dim=1)))

model = ConvLSTMUNet(in_ch=IN_CH, base_ch=16).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

---

## Cell 4 — W&B Init

```python
# ============================================================
# CELL 4 — Load W&B API key from Drive & init wandb
# ============================================================
import os

_env_candidates = [
    "/content/drive/MyDrive/Data_experiment_shoreline/.env",
    "/content/drive/MyDrive/.env",
]
for _p in _env_candidates:
    if os.path.exists(_p):
        with open(_p) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ[_k.strip()] = _v.strip()
        print(f"W&B key loaded from {_p}")
        break
else:
    print("WARNING: .env tidak ditemukan di Drive. Pastikan WANDB_API_KEY sudah ada di environment.")

import wandb
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(
    project="shoreline-kemujan",
    name=run_id,
    config={
        "model": "ConvLSTMUNet",
        "base_ch": 16,
        "in_ch": 1,
        "lookback": LOOKBACK,
        "dropout": 0.0,
        "epochs": 200,
        "batch_size": 4,
        "optimizer": "Adam",
        "lr": 1e-3,
        "norm": "BatchNorm2d",
    }
)
wandb.log({"dice/persistence_baseline": dice_baseline}, step=0)
print(f"W&B: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
```

---

## Cell 5 — Training Loop

```python
# ============================================================
# CELL 5 — Training loop (with W&B logging, no tqdm)
# ============================================================
import sys

EPOCHS = 200
BATCH_SIZE = 4

history = {"train_loss": [], "train_dice": [], "test_dice": []}
n, best_te = len(X_train), -1.0

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(n)
    epoch_loss = 0.0
    for i in range(0, n, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        optimizer.zero_grad()
        loss = criterion(model(X_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(idx)
    epoch_loss /= n

    model.eval()
    with torch.no_grad():
        tr_dice = dice_score(torch.sigmoid(model(X_train)), y_train).mean().item()
        te_dice = dice_score(torch.sigmoid(model(X_test)), y_test).mean().item()

    history["train_loss"].append(epoch_loss)
    history["train_dice"].append(tr_dice)
    history["test_dice"].append(te_dice)

    if te_dice > best_te:
        best_te = te_dice
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "checkpoint.pth"))

    wandb.log({
        "loss/train": epoch_loss,
        "dice/train": tr_dice,
        "dice/test": te_dice,
        "dice/best_test": best_te,
    }, step=epoch)

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        msg = f"Epoch {epoch:3d} | loss={epoch_loss:.4f} | tr={tr_dice:.4f} | te={te_dice:.4f} | best_te={best_te:.4f}"
        print(msg)
        sys.stdout.flush()

wandb.log({"dice/final_best": best_te})
print(f"Done. Best test Dice: {best_te:.4f}")
```

---

## Cell 6 — Save + Plot

```python
# ============================================================
# CELL 6 — Save model + history + plot (run after training)
# ============================================================
import matplotlib.pyplot as plt

model_path = os.path.join(MODEL_DIR, f"convlstm_unet_{run_id}.pth")
torch.save(model.state_dict(), model_path)
print(f"Model: {model_path}")

hist_path = os.path.join(LOG_DIR, f"history_{run_id}.npz")
np.savez_compressed(hist_path, **history)
print(f"History: {hist_path}")

# plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(history["train_loss"], color="tab:blue")
axes[0].set_title("Training Loss"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
axes[1].plot(history["train_dice"], label="train", color="tab:green")
axes[1].plot(history["test_dice"], label="test", color="tab:orange")
axes[1].axhline(dice_baseline, color="red", ls="--", label=f"persistence ({dice_baseline:.3f})")
axes[1].set_title("Dice Score"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
axes[2].plot(np.array(history["train_dice"]) - np.array(history["test_dice"]), color="tab:red")
axes[2].axhline(0.15, color="gray", ls=":", label="overfit threshold")
axes[2].set_title("Train-Test Gap"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"training_{run_id}.png"), dpi=120)
plt.show()

wandb.log({"chart/training": wandb.Image(fig)})
wandb.finish()
```

---

## Cell 7 — Summary

```python
# ============================================================
# CELL 7 — Final summary
# ============================================================
print("="*50)
print(f"Train Dice: {history['train_dice'][-1]:.4f}")
print(f"Test Dice:  {history['test_dice'][-1]:.4f}")
print(f"Best Test:  {best_te:.4f}")
print(f"Persistence: {dice_baseline:.4f}")
gap = history['train_dice'][-1] - history['test_dice'][-1]
print(f"Train-test gap: {gap:.4f} {'wajar' if gap < 0.15 else 'OVERFIT'}")
print("="*50)
```

---

## Cell 8 — Visualisasi Prediksi vs Ground Truth (test set)

```python
# ============================================================
# CELL 7 — Visualisasi: input → prediksi → ground truth
# ============================================================
import matplotlib.pyplot as plt

model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "checkpoint.pth")))
model.eval()

with torch.no_grad():
    te_prob = torch.sigmoid(model(X_test)).cpu().numpy()
    te_bin = (te_prob > 0.5).astype(np.float32)

n_show = min(6, len(X_test))
fig, axes = plt.subplots(n_show, 4, figsize=(16, 3 * n_show))
if n_show == 1:
    axes = axes[np.newaxis, :]

for i in range(n_show):
    input_mask = X_test[i, -1, 0].cpu().numpy()
    gt_mask = y_test[i, 0].cpu().numpy()
    pred_prob = te_prob[i, 0]
    pred_bin = te_bin[i, 0]

    axes[i, 0].imshow(input_mask, cmap="Blues", vmin=0, vmax=1)
    axes[i, 0].set_title(f"Input (last frame) — sample {i}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred_prob, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[i, 2].set_title(f"Prediction (prob) — Dice={dice_score(torch.from_numpy(pred_bin[np.newaxis,np.newaxis]), torch.from_numpy(gt_mask[np.newaxis,np.newaxis])).mean():.4f}")
    axes[i, 2].axis("off")

    overlay = np.zeros((*gt_mask.shape, 3))
    overlay[:, :, 0] = gt_mask        # red = GT
    overlay[:, :, 1] = pred_bin       # green = prediksi
    overlay[:, :, 2] = 0
    axes[i, 3].imshow(overlay)
    axes[i, 3].set_title("Overlay: GT(red) vs Pred(green)")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"pred_vs_gt_{run_id}.png"), dpi=120)
plt.show()
```

---

## Cell 9 — Rollout Prediction (multi-step forecast)

```python
# ============================================================
# CELL 8 — Rollout: prediksi N step ke depan
# ============================================================

def rollout(model, seed_frames, n_steps, device):
    model.eval()
    window = list(seed_frames)
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.from_numpy(np.stack([window[-1][np.newaxis]])).float().to(device)
            # rebuild full tensor with lookback
            stacked = np.stack([f[np.newaxis] for f in window[-LOOKBACK:]])  # (LOOKBACK, 1, H, W)
            x = torch.from_numpy(stacked[np.newaxis]).float().to(device)     # (1, LOOKBACK, 1, H, W)
            prob = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
            bin_pred = (prob > 0.5).astype(np.float32)
            preds.append(bin_pred)
            window.append(bin_pred)
    return preds

# ambil 3 sample test, rollout 5 step
n_rollout = 3
n_steps = 5
fig, axes = plt.subplots(n_rollout, n_steps + 1, figsize=(3 * (n_steps + 1), 3 * n_rollout))

for s in range(n_rollout):
    seed = X_test[s, :, 0].cpu().numpy()  # (LOOKBACK, H, W)
    gt = y_test[s, 0].cpu().numpy()

    # show seed last frame
    axes[s, 0].imshow(seed[-1], cmap="Blues", vmin=0, vmax=1)
    axes[s, 0].set_title(f"Seed t={-1}")
    axes[s, 0].axis("off")

    predictions = rollout(model, list(seed), n_steps, device)
    for j, pred in enumerate(predictions):
        axes[s, j + 1].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[s, j + 1].set_title(f"Rollout t+{j+1}")
        axes[s, j + 1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"rollout_{run_id}.png"), dpi=120)
plt.show()
```

---

## Cell 10 — Shoreline Contour Extraction + Error Map

```python
# ============================================================
# CELL 9 — Shoreline contour & error analysis
# ============================================================
from scipy.ndimage import binary_dilation, binary_erosion

def extract_shoreline(mask):
    """Return shoreline pixels (eroded XOR original)."""
    eroded = binary_erosion(mask, iterations=1)
    return mask & ~eroded

n_show = min(4, len(X_test))
fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))

for i in range(n_show):
    gt = y_test[i, 0].cpu().numpy() > 0.5
    pred = te_bin[i, 0] > 0.5

    gt_shore = extract_shoreline(gt)
    pred_shore = extract_shoreline(pred)
    correct = (gt == pred)
    fp = pred & ~gt   # false positive
    fn = gt & ~pred   # false negative

    # GT shoreline
    axes[i, 0].imshow(gt, cmap="Blues", alpha=0.3, vmin=0, vmax=1)
    axes[i, 0].imshow(gt_shore, cmap="Reds", alpha=1.0)
    axes[i, 0].set_title(f"GT Shoreline — sample {i}")
    axes[i, 0].axis("off")

    # Pred shoreline overlay
    axes[i, 1].imshow(gt, cmap="Blues", alpha=0.3, vmin=0, vmax=1)
    axes[i, 1].imshow(pred_shore, cmap="Greens", alpha=1.0)
    axes[i, 1].set_title("Pred Shoreline (green) overlay")
    axes[i, 1].axis("off")

    # error map
    error_map = np.zeros((*gt.shape, 3))
    error_map[..., 0] = fn  # red = missed (false neg)
    error_map[..., 1] = fp  # green = extra (false pos)
    error_map[..., 2] = correct * 0.3  # blue = correct (dim)
    axes[i, 2].imshow(error_map)
    axes[i, 2].set_title(f"Error: FN(red) FP(green) TP(blue)")
    axes[i, 2].axis("off")

    # water agreement
    axes[i, 3].imshow(gt, cmap="Blues", vmin=0, vmax=1, alpha=0.5)
    axes[i, 3].imshow(pred, cmap="Oranges", vmin=0, vmax=1, alpha=0.5)
    axes[i, 3].set_title("GT(blue) vs Pred(orange) overlay")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"shoreline_error_{run_id}.png"), dpi=120)
plt.show()

# summary metrics per sample
print(f"\n{'='*50}")
print("Per-sample metrics (test set):")
dice_vals = []
for i in range(len(X_test)):
    d = dice_score(torch.from_numpy(te_bin[i:i+1]), y_test[i:i+1]).mean().item()
    dice_vals.append(d)
print(f"  Dice: mean={np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
print(f"  Min={np.min(dice_vals):.4f}, Max={np.max(dice_vals):.4f}")
print(f"{'='*50}")
```

---

## Fix Applied

| Bug | Fix |
|-----|-----|
| `'ConvLSTMCell' object has no attribute 'hidden_ch'` | Added `self.hidden_ch = hidden_ch` di `__init__` (lupa disimpan pas di-compact) |
