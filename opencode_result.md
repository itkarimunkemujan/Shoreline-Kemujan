# Cell Baru untuk Resume Training (W&B + Augmentasi + Model Kecil)

## Cell Arsitektur — visualtorch (jalanin SETELAH Cell 1, SEBELUM Cell 3)

Copy sebagai cell baru di atas Cell 3 (setelah Cell 1 mount Drive + constants, sebelum data processing).

```python
# ============================================================
# CELL Arsitektur — Visualisasi arsitektur ConvLSTMUNet pakai visualtorch
#
# Jalanin SETELAH Cell 1 (biar IN_CH, LOOKBACK, PATCH_SIZE, LOG_DIR
# ada di scope) dan SEBELUM Cell 3 (gak depend on data apa-apa).
#
# base_ch=8, in_ch=IN_CH (dari Cell 8 / final_data/params.json),
# InstanceNorm2d, no dropout.
# ============================================================
!pip install visualtorch==1.4.1 Pillow -q

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import visualtorch

_IN_CH = IN_CH if "IN_CH" in globals() else 1
_LOOKBACK = LOOKBACK if "LOOKBACK" in globals() else 3
_PATCH_SIZE = PATCH_SIZE if "PATCH_SIZE" in globals() else 256


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch, H, W, device):
        return (torch.zeros(batch, self.hidden_ch, H, W, device=device),
                torch.zeros(batch, self.hidden_ch, H, W, device=device))


class ConvLSTMUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=8):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.InstanceNorm2d(base_ch * 2), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch * 2, base_ch * 2, 3)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch * 3, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch), nn.ReLU())
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c = None, None
        enc1_last = None
        for t in range(T):
            enc1 = self.enc1(x[:, t])
            enc2 = self.enc2(self.pool(enc1))
            if h is None:
                h, c = self.clstm.init_hidden(B, enc2.shape[2], enc2.shape[3], x.device)
            h, c = self.clstm(enc2, h, c)
            enc1_last = enc1
        up = self.up(h)
        dec = self.dec(torch.cat([up, enc1_last], dim=1))
        return self.head(dec)


# ---------- Visualtorch ----------
_viz_model = ConvLSTMUNet(in_ch=_IN_CH, base_ch=8).to("cpu").eval()
print(f"ConvLSTMUNet(base_ch=8, in_ch={_IN_CH}) — Total params: {sum(p.numel() for p in _viz_model.parameters()):,}")

LOG_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/logs"

try:
    color_map = defaultdict(dict)
    color_map[nn.Conv2d]["fill"] = "#E69F00"
    color_map[nn.ConvTranspose2d]["fill"] = "#009E73"
    color_map[nn.ReLU]["fill"] = "#56B4E9"
    color_map[nn.MaxPool2d]["fill"] = "#CC79A7"
    color_map[nn.InstanceNorm2d]["fill"] = "#D55E00"
    color_map[nn.Upsample]["fill"] = "#0072B2"

    _viz_input_shape = (1, _LOOKBACK, _IN_CH, _PATCH_SIZE, _PATCH_SIZE)
    img = visualtorch.render(_viz_model, _viz_input_shape, style="flow",
                              color_map=color_map, scale_xy=1, spacing=3)

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    MAX_WIDTH = 2200
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.Resampling.LANCZOS)

    output_path = os.path.join(LOG_DIR, "model_architecture_visualtorch.png")
    dpi = 150
    plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    print(f"✅ Diagram arsitektur: {output_path} ({img.width}×{img.height}px)")
except Exception as e:
    print(f"⚠️ visualtorch gagal: {e}")
    print(_viz_model)
```

**Ganti** Cell 9a, 9b, 9c dengan 3 cell di bawah ini.

## Cell 9a — Model Definition (BatchNorm, base_ch=16, MC Dropout 0.1)

```python
# =
# ===========================================================
# CELL 9a — Model: ConvLSTMUNet (BatchNorm2d, base_ch=16,
#   MC Dropout p=0.1), DiceBCELoss, dice_score
#
# - base_ch=16                (~86k params, sama kayak Cell 2)
# - BatchNorm2d               (lebih ringan di CPU dari InstanceNorm)
# - Dropout2d(p=0.1)          (MC Dropout — aktif pas train maupun eval)
# - Input: mask channel 0     (1 channel, gak pake static)
# ============================================================
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch, H, W, device):
        return (torch.zeros(batch, self.hidden_ch, H, W, device=device),
                torch.zeros(batch, self.hidden_ch, H, W, device=device))


class ConvLSTMUNet(nn.Module):
    """BatchNorm2d + MC Dropout 0.1 — dropout jalan pas train & eval."""

    def __init__(self, in_ch=1, base_ch=16, dropout=0.1):
        super().__init__()
        self.base_ch = base_ch
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Dropout2d(dropout))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch * 2, base_ch * 2, 3)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch * 3, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            nn.Dropout2d(dropout))
        self.head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c = None, None
        enc1_last = None
        for t in range(T):
            enc1 = self.enc1(x[:, t])
            enc2 = self.enc2(self.pool(enc1))
            if h is None:
                h, c = self.clstm.init_hidden(B, enc2.shape[2], enc2.shape[3], x.device)
            h, c = self.clstm(enc2, h, c)
            enc1_last = enc1
        up = self.up(h)
        dec = self.dec(torch.cat([up, enc1_last], dim=1))
        return self.head(dec)


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob_flat = prob.flatten(1)
        target_flat = target.flatten(1)
        inter = (prob_flat * target_flat).sum(1)
        dice = (2 * inter + 1e-6) / (prob_flat.sum(1) + target_flat.sum(1) + 1e-6)
        dice_loss = 1 - dice.mean()
        bce_loss = self.bce(logit, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)


# ================== Parameter Summary ==================
_IN_CH_preview = IN_CH if "IN_CH" in globals() else 1
_LOOKBACK_preview = LOOKBACK if "LOOKBACK" in globals() else 3
_PATCH_SIZE_preview = PATCH_SIZE if "PATCH_SIZE" in globals() else 256
_summary_model = ConvLSTMUNet(in_ch=_IN_CH_preview, base_ch=16, dropout=0.1)

total = sum(p.numel() for p in _summary_model.parameters())
print(f"ConvLSTMUNet(base_ch=16, in_ch={_IN_CH_preview}, dropout=0.1) — Total params: {total:,}")
print(f"Input shape: (B, LOOKBACK={_LOOKBACK_preview}, {_IN_CH_preview}, {_PATCH_SIZE_preview}, {_PATCH_SIZE_preview})")
print(f"Output shape: (B, 1, {_PATCH_SIZE_preview}, {_PATCH_SIZE_preview}) [mask logit, single channel]")
```

## Cell 9b — Training Loop + Augmentasi + W&B logging (definitions only)

```python
# ============================================================
# CELL 9b — Training loop (full-batch eval, noise+rotate augment)
#
# - single_forward: 1 forward pass per sample per epoch
# - augmentasi: rotate 90° + mask noise (no flip, no cutout)
# - eval: full-batch (langsung forward seluruh dataset)
# - MC Dropout: model.train() pas eval — dropout tetap aktif
# - W&B logging tiap epoch
#
# DEFINITIONS ONLY -- execution di Cell 9c.
# ============================================================
import os
import sys
import random
import numpy as np
import torch
import wandb
from tqdm import tqdm


EPOCHS = 150
BATCH_SIZE = 8
CKPT_EVERY = 10


def augment_batch(xb, yb):
    """Rotate 90° + mask noise. No flip, no cutout."""
    B, T, C, H, W = xb.shape

    k = random.randint(0, 3)
    if k > 0:
        xb = torch.rot90(xb, k, dims=[-2, -1])
        yb = torch.rot90(yb, k, dims=[-2, -1])

    if random.random() > 0.3:
        noise_lvl = random.uniform(0.02, 0.12)
        mask = torch.rand_like(xb[:, :, 0:1]) < noise_lvl
        xb[:, :, 0:1][mask] = 1.0 - xb[:, :, 0:1][mask]
        if random.random() > 0.5:
            mask_y = torch.rand_like(yb) < noise_lvl * 0.3
            yb[mask_y] = 1.0 - yb[mask_y]

    return xb, yb


def train_loop(model, optimizer, scheduler, criterion, X_tr, y_tr, X_te, y_te,
               ckpt_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, ckpt_every=CKPT_EVERY,
               logger=None):
    history = {"train_loss": [], "train_dice": [], "test_dice": []}
    n = len(X_tr)
    best_test_dice = -1.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        pbar = tqdm(range(0, n, batch_size), desc=f"Ep {epoch:{len(str(epochs-1))}d}/{epochs-1}", leave=False)
        for i in pbar:
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            xb, yb = augment_batch(xb, yb)
            optimizer.zero_grad()
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss /= n

        if scheduler is not None:
            scheduler.step(epoch_loss)

        model.train()  # MC Dropout — dropout tetap aktif
        with torch.no_grad():
            tr_pred = (torch.sigmoid(model(X_tr)) > 0.5).float()
            te_pred = (torch.sigmoid(model(X_te)) > 0.5).float()
            tr_dice = dice_score(tr_pred, y_tr).mean().item()
            te_dice = dice_score(te_pred, y_te).mean().item()

        history["train_loss"].append(epoch_loss)
        history["train_dice"].append(tr_dice)
        history["test_dice"].append(te_dice)

        wandb.log({
            "loss/train": epoch_loss,
            "dice/train": tr_dice,
            "dice/test": te_dice,
        }, step=epoch)

        if te_dice > best_test_dice:
            best_test_dice = te_dice
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
            wandb.log({"checkpoint/best": te_dice}, step=epoch)

        if epoch % ckpt_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ep{epoch:04d}.pth"))

        msg = (f"Epoch {epoch:3d}/{epochs-1} | loss={epoch_loss:.4f} | "
               f"tr={tr_dice:.4f} | te={te_dice:.4f} | best_te={best_test_dice:.4f}")
        print(msg)
        sys.stdout.flush()
        if logger is not None:
            logger.info(msg)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
    wandb.log({"checkpoint": f"epoch {epochs - 1}: FINAL -> last.pth"})
    return history, best_test_dice
```

## Cell 9c — Execution: resume from Drive + train + W&B + plot

```python
# ============================================================
# CELL 9c — EXECUTE: instantiate model/optimizer, resume from
#   Drive, run train_loop (Cell 9b), W&B logging, matplotlib plots.
#
# Run Cell 9a + 9b first (definitions).
#
# W&B: siapkan wandb login dulu (atau cell ini auto prompt).
# ============================================================
import os
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import wandb

torch.manual_seed(42)

# ---------- resume from Drive if session dropped ----------
FINAL_DATA_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/final_data"
if "X_train" not in globals():
    print("X_train tidak ada di memory -- reload dari Drive final_data/ ...")
    _tz = np.load(os.path.join(FINAL_DATA_DIR, "tensors.npz"))
    with open(os.path.join(FINAL_DATA_DIR, "params.json")) as f:
        _params = json.load(f)
    LOOKBACK = _params["LOOKBACK"]
    ROLLOUT_STEPS = _params["ROLLOUT_STEPS"]
    PATCH_SIZE = _params["PATCH_SIZE"]
    _X, _y, _tr, _te = _tz["X"], _tz["y"], _tz["tr"], _tz["te"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _X = _X[:, :, 0:1].copy()  # buang 6 static channel, pake mask aja
    IN_CH = 1
    X_train = torch.from_numpy(_X[_tr]).float().to(device)
    y_train = torch.from_numpy(_y[_tr, 0]).float().to(device)
    X_test  = torch.from_numpy(_X[_te]).float().to(device)
    y_test  = torch.from_numpy(_y[_te, 0]).float().to(device)
    print(f"✓ Reload dari Drive (mask-only): IN_CH={IN_CH}, LOOKBACK={LOOKBACK}, "
          f"PATCH_SIZE={PATCH_SIZE}, train={len(_tr)}, test={len(_te)}")
else:
    # Kalau udah di memory, yakinin constants ada (fallback dari Cell 1)
    LOOKBACK = LOOKBACK if "LOOKBACK" in globals() else 3
    PATCH_SIZE = PATCH_SIZE if "PATCH_SIZE" in globals() else 256
    print(f"Pakai X_train/X_test yang sudah ada di memory (IN_CH={IN_CH}).")

# ---------- pastiin cuma mask channel (buang static) ----------
if X_train.shape[2] > 1:
    X_train = X_train[:, :, 0:1].contiguous()
    X_test  = X_test[:, :, 0:1].contiguous()
    IN_CH = 1
    print(f"X di-slice ke mask-only: X_train {X_train.shape}, X_test {X_test.shape}")

# ---------- pastiin y cuma single-step ----------
if y_train.dim() == 5:  # (N, ROLLOUT_STEPS, 1, H, W) -> (N, 1, H, W)
    y_train = y_train[:, 0].contiguous()
    y_test  = y_test[:, 0].contiguous()
    print(f"y_train/y_test di-slice: {y_train.shape}, {y_test.shape}")

# ---------- init W&B ----------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(
    project="shoreline-kemujan",
    name=run_id,
    config={
        "model": "ConvLSTMUNet",
        "base_ch": 16,
        "in_ch": 1,
        "lookback": LOOKBACK,
        "dropout": 0.1,
        "mode": "mc_dropout_single_forward",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "lr": 1e-3,
        "augmentation": "rotate+mask_noise",
        "norm": "BatchNorm2d",
        "n_train": len(_tr) if "X_train" not in globals() else len(X_train),
        "n_test": len(_te) if "X_test" not in globals() else len(X_test),
        "n_params": 0,
    }
)
print(f"W&B run: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
print(f"  Buka link di atas buat liat chart loss/dice live selama training.")

model = ConvLSTMUNet(in_ch=IN_CH, base_ch=16, dropout=0.1).to(device)
n_params = sum(p.numel() for p in model.parameters())
wandb.config.update({"n_params": n_params})
print(f"Parameters: {n_params:,} (in_ch={IN_CH}, base_ch=16, dropout=0.1, BatchNorm)")

criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

wandb.log({"status": "training_initialized"}, step=0)
print(f"✅ W&B active — buka link di atas, liat chart loss/dice muncul tiap epoch selesai.")

# ---------- persistence baseline ----------
persist_pred = X_test[:, -1, 0:1]
dice_baseline = dice_score(persist_pred, y_test).mean().item()
print(f"Persistence baseline Dice (next-step): {dice_baseline:.4f}")
wandb.log({"dice/persistence_baseline": dice_baseline}, step=0)

# ---------- logging ----------
logger = logging.getLogger(f"train_{run_id}")
logger.setLevel(logging.INFO)
logger.handlers.clear()
LOG_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/logs"
fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{run_id}.log"))
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

CKPT_DIR = os.path.join(
    "/content/drive/MyDrive/Data_experiment_shoreline/models",
    f"checkpoints_{run_id}")
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Checkpoints: {CKPT_DIR}")

# ---------- RUN ----------
history, best_test_dice = train_loop(
    model, optimizer, scheduler, criterion,
    X_train, y_train, X_test, y_test,
    CKPT_DIR, logger=logger)

wandb.log({"best_test_dice": best_test_dice})

# ---------- save history ----------
np.savez_compressed(os.path.join(LOG_DIR, f"history_{run_id}.npz"), **history)

# ---------- matplotlib plots ----------
epochs_range = range(len(history["train_loss"]))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(epochs_range, history["train_loss"], color="tab:blue")
axes[0].set_title("Training Loss (single-step Dice+BCE)")
axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
axes[1].plot(epochs_range, history["train_dice"], label="train", color="tab:green")
axes[1].plot(epochs_range, history["test_dice"], label="test", color="tab:orange")
axes[1].axhline(dice_baseline, color="red", ls="--", label=f"persistence ({dice_baseline:.3f})")
axes[1].set_title("Dice Score (next-step)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
gap = np.array(history["train_dice"]) - np.array(history["test_dice"])
axes[2].plot(epochs_range, gap, color="tab:red")
axes[2].axhline(0.15, color="gray", ls=":", label="overfit threshold")
axes[2].set_title("Train-Test Gap")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"training_{run_id}.png"), dpi=120)
plt.show()
wandb.log({"chart/training": wandb.Image(fig)})
plt.close()

# ---------- final summary ----------
final_gap = history["train_dice"][-1] - history["test_dice"][-1]
print(f"\n{'='*50}")
print(f"Final train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final test Dice:  {history['test_dice'][-1]:.4f}")
print(f"Best test Dice:   {best_test_dice:.4f} (best.pth)")
print(f"Persistence:      {dice_baseline:.4f}")
print(f"Train-test gap:   {final_gap:.4f}")
print(f"{'='*50}")

wandb.finish()
print(f"\n✓ Training selesai. Lihat log di W&B: https://wandb.ai/runs/{run_id}")
```

## Cara Pakai

1. **Buka Colab** → Rename `final_experiment.ipynb` → buka di Colab
2. Hapus cell 9a, 9b, 9c yang lama
3. Paste 3 cell di atas (berurutan: 9a → 9b → 9c)
4. **PENTING**: Jalankan `Cell 1` dulu (Mount Drive + constants)
5. Jalankan `Cell 9a` → definisi model (BatchNorm, base_ch=16, MC Dropout 0.1)
6. Jalankan `Cell 9b` → definisi training loop (noise+rotate, full-batch eval)
7. Jalankan `Cell 9c` → training (mask-only, W&B)

Catatan: W&B bakal minta API key pas `wandb.init` — login dulu di Colab lewat `!wandb login` atau paste token pas prompt.
