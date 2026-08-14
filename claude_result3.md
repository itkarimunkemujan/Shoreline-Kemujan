# claude_result3.md — Combined Landsat+Sentinel ConvLSTM-UNet (2aug)

Cells untuk di-paste ke Colab notebook baru. Ikuti urutan cell — jangan skip.
Semua checkpoint/output pakai suffix `_2aug`. Tidak ada tqdm di mana pun.

> **Data source**: Load `tensors.npz` langsung dari Drive (sudah merged Landsat+Sentinel dari `final_experiment.ipynb`) — tidak perlu rebuild dari zip.
> **Drive paths**: `MyDrive/Data_experiment_shoreline/final_data/` (tensors), `models/` (checkpoints), `logs/` (history), `output/` (geojson).

---

## Cell 1 (2aug) — Mount Drive + All Constants

New cell (first cell in notebook).

Perubahan:
- Semua konstanta di satu tempat untuk mudah diubah
- RUN_SUFFIX = "2aug" dipakai di semua nama file

```python
# ================================================================
# CELL 1 — Mount Drive + Constants
# ================================================================
import os
import logging

from google.colab import drive
drive.mount('/content/drive')

# ── Drive paths ──────────────────────────────────────────────────
DRIVE_BASE    = "/content/drive/MyDrive/Data_experiment_shoreline"
TENSOR_PATH   = os.path.join(DRIVE_BASE, "final_data", "tensors.npz")
PARAMS_PATH   = os.path.join(DRIVE_BASE, "final_data", "params.json")
META_PATH     = os.path.join(DRIVE_BASE, "final_data", "meta.json")
MODEL_DIR     = os.path.join(DRIVE_BASE, "models")
LOG_DIR       = os.path.join(DRIVE_BASE, "logs")
OUTPUT_DIR    = os.path.join(DRIVE_BASE, "output")
ENV_PATH      = os.path.join(DRIVE_BASE, ".env")      # contains WANDB_API_KEY

for d in [MODEL_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Run identity ─────────────────────────────────────────────────
from datetime import datetime
RUN_SUFFIX = "2aug"
run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"run_id: {run_id}  suffix: {RUN_SUFFIX}")

# ── Model / training hyper-params ────────────────────────────────
LOOKBACK       = 3
TRAIN_UNTIL    = 2023        # target_year <= this → train set
BASE_CH        = 16
MC_DROPOUT_P   = 0.1
EPOCHS         = 150
BATCH_SIZE     = 4
LR             = 1e-3
SEED           = 42

# ── Inference / rollout ──────────────────────────────────────────
ROLLOUT_SEASONS   = ["S1", "S2", "S3"]          # 3 per year
SEASON_MONTHS     = {"S1": 3, "S2": 7, "S3": 11, "L1": 4, "L2": 10}
ROLLOUT_START     = 2025
ROLLOUT_END       = 2035
GEOJSON_SNAPSHOT_YEARS = [2025, 2030, 2035]     # which years to export

# ── Device + seed ────────────────────────────────────────────────
import torch
import numpy as np
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ── Root logger ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shoreline_2aug")
fh  = logging.FileHandler(os.path.join(LOG_DIR, f"train_{RUN_SUFFIX}_{run_id}.log"))
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
log.addHandler(fh)
log.info("Constants loaded. Device=%s  run_id=%s", device, run_id)
```

---

## Cell 2 (2aug) — Load Tensors from Drive

New cell after Cell 1.

Perubahan:
- Muat tensors.npz + params.json + meta.json dari Drive
- Print raw shapes sebelum split

```python
# ================================================================
# CELL 2 — Load tensors.npz from Drive
# ================================================================
import json

log.info("Loading tensors from Drive: %s", TENSOR_PATH)
_tz = np.load(TENSOR_PATH)
log.info("Keys in tensors.npz: %s", list(_tz.files))

X_raw = _tz["X"]   # shape: (N, LOOKBACK, C, H, W)  C may include static channels
y_raw = _tz["y"]   # shape: (N, ROLLOUT_STEPS, 1, H, W) or (N, 1, H, W)

log.info("X_raw: %s  dtype=%s", X_raw.shape, X_raw.dtype)
log.info("y_raw: %s  dtype=%s", y_raw.shape, y_raw.dtype)

with open(PARAMS_PATH) as f:
    params = json.load(f)
with open(META_PATH) as f:
    meta = json.load(f)

log.info("params: %s", params)
log.info("Total samples in meta: %d", len(meta))
```

---

## Cell 3 (2aug) — Slice to Mask-Only + Temporal Split

New cell after Cell 2.

Perubahan:
- Potong ke channel 0 saja (binary water mask, buang static channels)
- Split temporal: target_year <= TRAIN_UNTIL → train
- Move ke device

```python
# ================================================================
# CELL 3 — Slice to mask-only + temporal split
# ================================================================

# Slice: mask channel only (index 0 of C dimension)
X_mask = X_raw[:, :, 0:1, :, :]    # (N, LOOKBACK, 1, H, W)

# y may be (N, ROLLOUT_STEPS, 1, H, W) → take first rollout step only
if y_raw.ndim == 5:
    y_mask = y_raw[:, 0, :, :, :]  # (N, 1, H, W)
else:
    y_mask = y_raw                  # already (N, 1, H, W)

log.info("After mask-only slice — X: %s  y: %s", X_mask.shape, y_mask.shape)

# Temporal split using meta
tr_idx = [i for i, m in enumerate(meta) if m.get("target_year", 9999) <= TRAIN_UNTIL]
te_idx = [i for i, m in enumerate(meta) if m.get("target_year", 9999) >  TRAIN_UNTIL]

X_train = torch.from_numpy(X_mask[tr_idx]).float().to(device)
y_train = torch.from_numpy(y_mask[tr_idx]).float().to(device)
X_test  = torch.from_numpy(X_mask[te_idx]).float().to(device)
y_test  = torch.from_numpy(y_mask[te_idx]).float().to(device)

meta_train = [meta[i] for i in tr_idx]
meta_test  = [meta[i] for i in te_idx]

log.info("Train: %d samples (target <= %d)", len(tr_idx), TRAIN_UNTIL)
log.info("Test:  %d samples (target >  %d)", len(te_idx), TRAIN_UNTIL)
log.info("X_train: %s  X_test: %s", tuple(X_train.shape), tuple(X_test.shape))
```

---

## Cell 4 (2aug) — Persistence Baseline

New cell after Cell 3.

Perubahan:
- Hitung persistence Dice sebelum training sebagai sanity check
- Nilai ini jadi floor untuk perbandingan model

```python
# ================================================================
# CELL 4 — Persistence baseline Dice
# ================================================================

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred   = pred.flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    return (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)

with torch.no_grad():
    persist_pred    = X_test[:, -1]                          # last input frame
    persist_dice    = dice_score(persist_pred, y_test).mean().item()

log.info("Persistence baseline Dice (test): %.4f", persist_dice)
print(f"Persistence baseline Dice: {persist_dice:.4f}  ← model must beat this")
```

---

## Cell 5 (2aug) — Model Definition

New cell after Cell 4.

Perubahan:
- ConvLSTMUNet dengan MC Dropout p=0.1, BatchNorm2d
- enable_mc_dropout() dan mc_dropout_predict() helpers

```python
# ================================================================
# CELL 5 — ConvLSTM-UNet model definition
# ================================================================
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=kernel_size // 2
        )

    def forward(self, x, h, c):
        gates    = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c_next   = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next   = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, device):
        return (
            torch.zeros(B, self.hidden_ch, H, W, device=device),
            torch.zeros(B, self.hidden_ch, H, W, device=device),
        )


class ConvLSTMUNet(nn.Module):
    """
    Shallow ConvLSTM-UNet.  Skip connection from enc1 (full resolution)
    to decoder — avoids the shape-mismatch bug of connecting enc2 (half-res).
    base_ch=16 intentionally small for ~300-sample training set.
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 16, mc_dropout_p: float = 0.1):
        super().__init__()
        self.enc1  = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
            nn.Dropout2d(mc_dropout_p),
        )
        self.enc2  = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(),
            nn.Dropout2d(mc_dropout_p),
        )
        self.pool  = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch * 2, base_ch * 2, 3)
        self.up    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec   = nn.Sequential(
            nn.Conv2d(base_ch * 3, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
            nn.Dropout2d(mc_dropout_p),
        )
        self.head  = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        h = c = enc1_last = None
        for t in range(T):
            enc1 = self.enc1(x[:, t])
            enc2 = self.enc2(self.pool(enc1))
            if h is None:
                h, c = self.clstm.init_hidden(B, enc2.shape[2], enc2.shape[3], x.device)
            h, c      = self.clstm(enc2, h, c)
            enc1_last = enc1
        up  = self.up(h)
        dec = self.dec(torch.cat([up, enc1_last], dim=1))
        return self.head(dec)


def enable_mc_dropout(model: ConvLSTMUNet) -> ConvLSTMUNet:
    """Gal & Ghahramani (2016): BatchNorm uses running stats, Dropout2d keeps sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    return model


@torch.no_grad()
def mc_dropout_predict(model: ConvLSTMUNet, x: torch.Tensor, n_samples: int = 20):
    """Returns (mean_prob, std_prob); std = per-pixel epistemic uncertainty proxy."""
    enable_mc_dropout(model)
    probs = torch.stack([torch.sigmoid(model(x)) for _ in range(n_samples)])
    model.eval()
    return probs.mean(0), probs.std(0)


model = ConvLSTMUNet(in_ch=1, base_ch=BASE_CH, mc_dropout_p=MC_DROPOUT_P).to(device)
n_params = sum(p.numel() for p in model.parameters())
log.info("Model: ConvLSTMUNet(in_ch=1, base_ch=%d, mc_dropout_p=%.2f) — %s params",
         BASE_CH, MC_DROPOUT_P, f"{n_params:,}")
print(f"Parameters: {n_params:,}")
```

---

## Cell 6 (2aug) — Loss + Optimizer

New cell after Cell 5.

Perubahan:
- DiceBCELoss dengan bce_weight=0.5
- Adam dengan lr dari konstanta

```python
# ================================================================
# CELL 6 — Loss function + optimizer
# ================================================================

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob  = torch.sigmoid(logit)
        p_f   = prob.flatten(1)
        t_f   = target.flatten(1)
        inter = (p_f * t_f).sum(1)
        dice_loss = 1 - ((2 * inter + 1e-6) / (p_f.sum(1) + t_f.sum(1) + 1e-6)).mean()
        return self.bce_weight * self.bce(logit, target) + (1 - self.bce_weight) * dice_loss


criterion = DiceBCELoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
log.info("Loss: DiceBCELoss(bce_weight=0.5)  Optimizer: Adam(lr=%s)", LR)
```

---

## Cell 7 (2aug) — W&B Init

New cell after Cell 6.

Perubahan:
- Baca WANDB_API_KEY dari .env di Drive
- Init run dengan nama yang menyertakan suffix _2aug dan run_id

```python
# ================================================================
# CELL 7 — Weights & Biases init
# ================================================================
import wandb

# Read API key from Drive .env
_env_vars = {}
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                _env_vars[_k.strip()] = _v.strip()

if "WANDB_API_KEY" in _env_vars:
    os.environ["WANDB_API_KEY"] = _env_vars["WANDB_API_KEY"]
    log.info("WANDB_API_KEY loaded from Drive .env")
else:
    log.warning("WANDB_API_KEY not found in %s — W&B logging will be offline", ENV_PATH)

wandb.init(
    project="shoreline-kemujan",
    name=f"run_{RUN_SUFFIX}_{run_id}",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "base_ch": BASE_CH,
        "mc_dropout_p": MC_DROPOUT_P,
        "train_until": TRAIN_UNTIL,
        "lookback": LOOKBACK,
        "run_suffix": RUN_SUFFIX,
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
    },
    tags=[RUN_SUFFIX, "landsat+sentinel", "mask-only"],
)
wandb.log({"dice/persistence_baseline": persist_dice})
log.info("W&B run initialized: %s", wandb.run.url if wandb.run else "offline")
```

---

## Cell 8 (2aug) — Training Loop

New cell after Cell 7.

Perubahan:
- augment_batch(): rotate-90 + mask noise (tanpa flip untuk konsistensi geografis)
- train_loop(): logging tiap epoch (no tqdm), checkpoint setiap 10 epoch dengan suffix _2aug
- best_{RUN_SUFFIX}.pth tersimpan otomatis saat val Dice membaik
- final_{RUN_SUFFIX}.pth di akhir training

```python
# ================================================================
# CELL 8 — Training loop
# ================================================================

def augment_batch(xb: torch.Tensor, yb: torch.Tensor):
    """Rotate-90 augmentation + light mask noise. No horizontal flip
    (preserves E-W coastal orientation for EPR transect consistency)."""
    k     = torch.randint(0, 4, (1,)).item()
    if k > 0:
        xb = torch.rot90(xb, k, dims=[-2, -1])
        yb = torch.rot90(yb, k, dims=[-2, -1])
    # mask noise: randomly zero out ~5% of positive pixels
    if torch.rand(1).item() < 0.5:
        noise = (torch.rand_like(yb) > 0.05).float()
        yb    = yb * noise
    return xb, yb


def train_loop(model, X_tr, y_tr, X_te, y_te):
    history = {"train_loss": [], "train_dice": [], "test_dice": []}
    n       = len(X_tr)
    best_dice   = -1.0
    best_path   = os.path.join(MODEL_DIR, f"best_{RUN_SUFFIX}.pth")

    log.info("Starting training: %d train | %d test | epochs=%d | batch=%d",
             n, len(X_te), EPOCHS, BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()
        perm       = torch.randperm(n)
        epoch_loss = 0.0

        for i in range(0, n, BATCH_SIZE):
            idx        = perm[i : i + BATCH_SIZE]
            xb, yb     = X_tr[idx], y_tr[idx]
            xb, yb     = augment_batch(xb, yb)
            optimizer.zero_grad()
            loss       = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            # batched eval to avoid OOM on larger sets
            def _eval_dice(X, y, bs=16):
                dices = []
                for i in range(0, len(X), bs):
                    pred = (torch.sigmoid(model(X[i:i+bs])) > 0.5).float()
                    dices.append(dice_score(pred, y[i:i+bs]))
                return torch.cat(dices).mean().item()

            tr_dice = _eval_dice(X_tr, y_tr)
            te_dice = _eval_dice(X_te, y_te)

        history["train_loss"].append(epoch_loss)
        history["train_dice"].append(tr_dice)
        history["test_dice"].append(te_dice)

        wandb.log({
            "loss/train":      epoch_loss,
            "dice/train":      tr_dice,
            "dice/test":       te_dice,
            "dice/best_test":  best_dice,
            "epoch":           epoch,
        })

        log.info("Epoch %3d/%d | loss=%.4f | tr=%.4f | te=%.4f",
                 epoch + 1, EPOCHS, epoch_loss, tr_dice, te_dice)

        # Best checkpoint
        if te_dice > best_dice:
            best_dice = te_dice
            torch.save(model.state_dict(), best_path)
            log.info("  → New best test Dice %.4f  saved: %s", best_dice, best_path)

        # Periodic checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            ck_path = os.path.join(
                MODEL_DIR, f"convlstm_unet_{RUN_SUFFIX}_{run_id}_ep{epoch+1:03d}.pth"
            )
            torch.save(model.state_dict(), ck_path)
            log.info("  → Checkpoint saved: %s", ck_path)

    # Final checkpoint
    final_path = os.path.join(MODEL_DIR, f"final_{RUN_SUFFIX}_{run_id}.pth")
    torch.save(model.state_dict(), final_path)
    log.info("Training complete. Final saved: %s | Best val Dice: %.4f", final_path, best_dice)
    return history


history = train_loop(model, X_train, y_train, X_test, y_test)
```

---

## Cell 9 (2aug) — Save History + Training Plot

New cell after Cell 8.

Perubahan:
- Simpan history ke Drive sebagai NPZ
- 3-panel matplotlib: loss / Dice / train-test gap
- Simpan plot ke Drive dan log ke W&B

```python
# ================================================================
# CELL 9 — Save history + training plot
# ================================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

history_path = os.path.join(LOG_DIR, f"history_{RUN_SUFFIX}_{run_id}.npz")
np.savez_compressed(history_path, **history)
log.info("History saved: %s", history_path)

epochs_arr  = list(range(1, EPOCHS + 1))
tr_loss     = history["train_loss"]
tr_dice     = history["train_dice"]
te_dice     = history["test_dice"]
gap         = [tr - te for tr, te in zip(tr_dice, te_dice)]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(f"Training — ConvLSTMUNet {RUN_SUFFIX} ({run_id})", fontsize=11)

axes[0].plot(epochs_arr, tr_loss, color="steelblue")
axes[0].set_title("Train Loss (DiceBCE)")
axes[0].set_xlabel("Epoch")
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

axes[1].plot(epochs_arr, tr_dice, label="Train Dice", color="seagreen")
axes[1].plot(epochs_arr, te_dice, label="Test Dice",  color="darkorange")
axes[1].axhline(persist_dice, color="gray", ls="--", lw=1.2, label=f"Persistence {persist_dice:.4f}")
axes[1].set_title("Dice Score")
axes[1].set_xlabel("Epoch")
axes[1].legend(fontsize=8)

axes[2].plot(epochs_arr, gap, color="crimson")
axes[2].axhline(0.15, color="gray", ls=":", lw=1, label="overfit threshold 0.15")
axes[2].set_title("Train-Test Gap (Dice)")
axes[2].set_xlabel("Epoch")
axes[2].legend(fontsize=8)

plt.tight_layout()
plot_path = os.path.join(LOG_DIR, f"training_{RUN_SUFFIX}_{run_id}.png")
plt.savefig(plot_path, dpi=120)
plt.show()
log.info("Plot saved: %s", plot_path)

wandb.log({"chart/training": wandb.Image(plot_path)})
wandb.finish()
log.info("W&B run finished.")

best_te = max(te_dice)
final_gap = gap[-1]
print(f"\nBest test Dice : {best_te:.4f}")
print(f"Final train-test gap: {final_gap:.4f}  {'⚠ OVERFIT' if final_gap > 0.15 else 'OK'}")
```

---

## Cell 10 (2aug) — Load Best Checkpoint + Build Rollout Seeds

New cell after Cell 9.

Perubahan:
- Load best_{RUN_SUFFIX}.pth
- Bangun rollout seed: 3 frame GT terakhir per AOI
- Jika AOI punya < 3 frame di test set, fallback ke frame training terakhir

```python
# ================================================================
# CELL 10 — Load best checkpoint + build rollout seeds per AOI
# ================================================================

best_path = os.path.join(MODEL_DIR, f"best_{RUN_SUFFIX}.pth")
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()
log.info("Loaded best checkpoint: %s", best_path)

# Build per-AOI frame dict from ALL meta (train + test combined)
all_meta = meta_train + meta_test
all_X    = torch.cat([X_train, X_test], dim=0)   # (N, LOOKBACK, 1, H, W)
all_y    = torch.cat([y_train, y_test], dim=0)    # (N, 1, H, W)

# Group by AOI: collect (target_year, target_season, mask_tensor) tuples
from collections import defaultdict
aoi_frames = defaultdict(list)   # aoi -> [(t_float, mask)]

for i, m in enumerate(all_meta):
    aoi    = m.get("aoi", "unknown")
    t_yr   = m.get("target_year", 0)
    t_seas = m.get("target_season", "S1")
    t_mon  = SEASON_MONTHS.get(t_seas, 6)
    t_float = t_yr + t_mon / 12.0
    mask   = all_y[i]    # (1, H, W)
    aoi_frames[aoi].append((t_float, mask))

# Sort each AOI chronologically
for aoi in aoi_frames:
    aoi_frames[aoi].sort(key=lambda x: x[0])

# Build seed: last 3 frames per AOI
rollout_seeds = {}
for aoi, frames in aoi_frames.items():
    if len(frames) < LOOKBACK:
        log.warning("AOI %s has only %d frames — padding with last frame", aoi, len(frames))
        while len(frames) < LOOKBACK:
            frames.insert(0, frames[0])
    seed_masks = torch.stack([f[1] for f in frames[-LOOKBACK:]], dim=0)  # (LOOKBACK, 1, H, W)
    rollout_seeds[aoi] = seed_masks.to(device)
    log.info("AOI %s seed shape: %s  (last GT: t=%.2f)", aoi, tuple(seed_masks.shape), frames[-1][0])

print(f"Rollout seeds ready for {len(rollout_seeds)} AOIs: {list(rollout_seeds.keys())}")
```

---

## Cell 11 (2aug) — Rollout Inference 2025–2035

New cell after Cell 10.

Perubahan:
- Autoregressive rollout 3 season/tahun dari 2025–2035 (~33 langkah per AOI)
- Simpan masks dan uncertainty (MC Dropout 20 samples)
- Simpan contour (koordinat pixel) per AOI per step

```python
# ================================================================
# CELL 11 — Autoregressive rollout 2025–2035 (3 seasons/year)
# ================================================================
from skimage import measure
from scipy.interpolate import splprep, splev

def extract_contour(mask_np: np.ndarray, smooth: bool = True):
    """Extract largest shoreline contour from binary mask. Returns (row, col) arrays."""
    contours = measure.find_contours(mask_np, level=0.5)
    if not contours:
        return None, None
    contour = max(contours, key=len)      # largest contour
    if smooth and len(contour) > 5:
        try:
            tck, _ = splprep([contour[:, 1], contour[:, 0]], s=len(contour), k=3)
            xi, yi = splev(np.linspace(0, 1, 200), tck)
            return yi, xi                 # row (y), col (x)
        except Exception:
            pass
    return contour[:, 0], contour[:, 1]


# Generate rollout timesteps
rollout_steps = []
for yr in range(ROLLOUT_START, ROLLOUT_END + 1):
    for seas in ROLLOUT_SEASONS:
        mon = SEASON_MONTHS[seas]
        rollout_steps.append({"year": yr, "season": seas, "t": yr + mon / 12.0})

log.info("Total rollout steps: %d  (%d–%d, %d seasons/yr)",
         len(rollout_steps), ROLLOUT_START, ROLLOUT_END, len(ROLLOUT_SEASONS))

# Run rollout per AOI
rollout_results = {}   # aoi -> list of dicts per step

for aoi, seed in rollout_seeds.items():
    window = seed.clone()                      # (LOOKBACK, 1, H, W)
    aoi_results = []

    for step_info in rollout_steps:
        x_in = window.unsqueeze(0)             # (1, LOOKBACK, 1, H, W)
        with torch.no_grad():
            logit = model(x_in)                # (1, 1, H, W)
        prob_mean, prob_std = mc_dropout_predict(model, x_in, n_samples=20)
        pred_mask = (prob_mean > 0.5).float()  # (1, 1, H, W)

        mask_np  = pred_mask[0, 0].cpu().numpy()
        prob_np  = prob_mean[0, 0].cpu().numpy()
        unc_np   = prob_std[0, 0].cpu().numpy()

        rows, cols = extract_contour(mask_np)

        aoi_results.append({
            **step_info,
            "mask_np":       mask_np,
            "prob_np":       prob_np,
            "unc_np":        unc_np,
            "contour_row":   rows,
            "contour_col":   cols,
            "unc_mean":      float(unc_np.mean()),
        })

        # Advance window: drop oldest, append new prediction
        window = torch.cat([window[1:], pred_mask[0]], dim=0)

    rollout_results[aoi] = aoi_results
    log.info("AOI %s rollout done — %d steps, avg unc=%.4f",
             aoi, len(aoi_results), np.mean([r["unc_mean"] for r in aoi_results]))

print(f"Rollout complete for {len(rollout_results)} AOIs.")
```

---

## Cell 12 (2aug) — EPR + LRR + Transect Analysis

New cell after Cell 11.

Perubahan:
- Bangun transect DSAS dari contour 2025 sebagai baseline
- Hitung EPR per transect (model rollout) dan LRR (regresi linier GT historis)
- Klasifikasi DSAS: Erosi Parah / Erosi / Stabil / Akresi / Akresi Kuat
- Simpan sebagai DataFrame dan dict

```python
# ================================================================
# CELL 12 — EPR / LRR / Transect analysis
# ================================================================
import pandas as pd
from shapely.geometry import LineString, Point

# ── helpers ──────────────────────────────────────────────────────

H_PIX, W_PIX = X_train.shape[-2], X_train.shape[-1]   # 256, 256
SCALE_M       = params.get("scale_m", 10)              # metres per pixel (10 m Sentinel grid)

def px_to_lonlat(col, row, bbox):
    """Convert pixel (col, row) to (lon, lat) using bbox [west, south, east, north]."""
    west, south, east, north = bbox
    lon = west + (col / W_PIX) * (east  - west)
    lat = north - (row / H_PIX) * (north - south)
    return lon, lat

def build_transects(rows, cols, bbox, n_transects=40, length_m=200):
    """Build perpendicular DSAS-style transects from contour."""
    if rows is None or len(rows) < 4:
        return []
    length_deg = length_m / 111_000   # approx degrees
    pts = list(zip(rows, cols))
    # Evenly spaced baseline points
    total   = len(pts)
    indices = [int(i * (total - 1) / (n_transects - 1)) for i in range(n_transects)]
    transects = []
    for idx in indices:
        r, c = pts[idx]
        # tangent via finite differences
        r0, c0 = pts[max(idx - 2, 0)]
        r1, c1 = pts[min(idx + 2, total - 1)]
        dr, dc  = r1 - r0, c1 - c0
        norm    = max((dr**2 + dc**2) ** 0.5, 1e-9)
        # perpendicular in pixel space
        perp_r, perp_c = -dc / norm, dr / norm
        # convert baseline origin to lon/lat
        lon, lat = px_to_lonlat(c, r, bbox)
        # perpendicular offsets in degrees
        perp_lon = perp_c / W_PIX * (bbox[2] - bbox[0])
        perp_lat = -perp_r / H_PIX * (bbox[3] - bbox[1])
        p1 = (lon - perp_lon * length_deg * 10,
              lat - perp_lat * length_deg * 10)
        p2 = (lon + perp_lon * length_deg * 10,
              lat + perp_lat * length_deg * 10)
        transects.append({"id": idx, "origin_lon": lon, "origin_lat": lat,
                           "line": LineString([p1, p2])})
    return transects


def contour_intersection_distance(transect_line, rows, cols, bbox, origin_lon, origin_lat):
    """Signed distance (m) from baseline origin to shoreline intersection along transect."""
    if rows is None or len(rows) < 2:
        return None
    lons = [px_to_lonlat(c, r, bbox)[0] for c, r in zip(cols, rows)]
    lats = [px_to_lonlat(c, r, bbox)[1] for c, r in zip(cols, rows)]
    shoreline_line = LineString(list(zip(lons, lats)))
    try:
        inter = transect_line.intersection(shoreline_line)
        if inter.is_empty:
            return None
        if hasattr(inter, "geoms"):
            inter = list(inter.geoms)[0]
        dx = (inter.x - origin_lon) * 111_000 * np.cos(np.radians(origin_lat))
        dy = (inter.y - origin_lat) * 111_000
        return float((dx**2 + dy**2) ** 0.5 * np.sign(dx))
    except Exception:
        return None


DSAS_CLASSES = [
    (-np.inf, -2.0, "Erosi Parah"),
    (-2.0,    -0.5, "Erosi"),
    (-0.5,     0.5, "Stabil"),
    ( 0.5,     2.0, "Akresi"),
    ( 2.0,  np.inf, "Akresi Kuat"),
]

def classify_epr(epr_m_yr):
    for lo, hi, label in DSAS_CLASSES:
        if lo <= epr_m_yr < hi:
            return label
    return "Stabil"


# ── main analysis ─────────────────────────────────────────────────

transect_records = []    # flat list of dicts for DataFrame

for aoi, results in rollout_results.items():
    # Retrieve bbox from params/manifest (fallback to None-safe)
    bbox = None
    if "aoi" in params and isinstance(params["aoi"], dict):
        aoi_info = params["aoi"].get(aoi, {})
        bbox = aoi_info.get("bbox")
    if bbox is None:
        log.warning("AOI %s: no bbox in params — skipping transect analysis", aoi)
        continue

    # Build baseline transects from first rollout year (2025 S1)
    first = next((r for r in results if r["year"] == ROLLOUT_START and r["season"] == "S1"), None)
    if first is None or first["contour_row"] is None:
        log.warning("AOI %s: no 2025 S1 contour — skipping", aoi)
        continue

    transects = build_transects(first["contour_row"], first["contour_col"], bbox)
    if not transects:
        log.warning("AOI %s: could not build transects", aoi)
        continue

    # Collect model shoreline position per transect per step
    # Store: transect_id -> {t: distance_m}
    from collections import defaultdict
    t_pos = {t["id"]: {} for t in transects}

    for r in results:
        if r["contour_row"] is None:
            continue
        for t in transects:
            d = contour_intersection_distance(
                t["line"], r["contour_row"], r["contour_col"], bbox,
                t["origin_lon"], t["origin_lat"]
            )
            t_pos[t["id"]][r["t"]] = d

    # GT historical positions (from aoi_frames)
    gt_frames = aoi_frames.get(aoi, [])
    gt_t_pos = {t["id"]: {} for t in transects}
    for t_float, gt_mask in gt_frames:
        rows_gt, cols_gt = extract_contour(gt_mask[0].cpu().numpy())
        for t in transects:
            d = contour_intersection_distance(
                t["line"], rows_gt, cols_gt, bbox,
                t["origin_lon"], t["origin_lat"]
            )
            gt_t_pos[t["id"]][t_float] = d

    # Compute EPR (model) and LRR (GT historical linear regression)
    for t in transects:
        tid = t["id"]
        model_pos = {tt: dd for tt, dd in t_pos[tid].items() if dd is not None}
        gt_pos    = {tt: dd for tt, dd in gt_t_pos[tid].items() if dd is not None}

        # EPR: (last - first) / Δt
        if len(model_pos) >= 2:
            t_sorted = sorted(model_pos)
            epr = (model_pos[t_sorted[-1]] - model_pos[t_sorted[0]]) / (t_sorted[-1] - t_sorted[0])
            nsm = model_pos[t_sorted[-1]] - model_pos[t_sorted[0]]
            sce = max(model_pos.values()) - min(model_pos.values())
        else:
            epr = nsm = sce = float("nan")

        # LRR: linear regression on GT positions vs time
        if len(gt_pos) >= 2:
            gt_t_arr = np.array(sorted(gt_pos))
            gt_d_arr = np.array([gt_pos[tt] for tt in gt_t_arr])
            coeffs   = np.polyfit(gt_t_arr, gt_d_arr, 1)
            lrr      = float(coeffs[0])   # m/yr
        else:
            lrr = float("nan")

        # Uncertainty from MC Dropout (mean over rollout steps)
        mc_uncs = [r["unc_mean"] for r in results]
        mc_unc_mean = float(np.nanmean(mc_uncs)) if mc_uncs else float("nan")

        # Per-year model EPR positions for Sanity
        model_positions_by_year = {}
        for r in results:
            yr_key = f"{r['year']}_{r['season']}"
            if t_pos[tid].get(r["t"]) is not None:
                model_positions_by_year[yr_key] = round(t_pos[tid][r["t"]], 3)

        transect_records.append({
            "aoi":             aoi,
            "transect_id":     tid,
            "origin_lon":      round(t["origin_lon"], 6),
            "origin_lat":      round(t["origin_lat"], 6),
            "epr_m_per_yr":    round(epr, 4) if not np.isnan(epr) else None,
            "nsm_m":           round(nsm, 3) if not np.isnan(nsm) else None,
            "sce_m":           round(sce, 3) if not np.isnan(sce) else None,
            "lrr_m_per_yr":    round(lrr, 4) if not np.isnan(lrr) else None,
            "classification":  classify_epr(epr) if not np.isnan(epr) else "Unknown",
            "mc_uncertainty":  round(mc_unc_mean, 5),
            "model_positions": model_positions_by_year,
        })

transect_df = pd.DataFrame(transect_records)
log.info("Transect analysis done: %d records across %d AOIs",
         len(transect_df), transect_df["aoi"].nunique() if len(transect_df) > 0 else 0)

if len(transect_df) > 0:
    print(transect_df.groupby("aoi")["epr_m_per_yr"].describe().round(3))
else:
    print("No transect records — check bbox in params.json")
```

---

## Cell 13 (2aug) — GeoJSON Export (2025 / 2030 / 2035)

New cell after Cell 12.

Perubahan:
- Export 3 GeoJSON snapshots: 2025, 2030, 2035
- Setiap file punya model shoreline + LRR shoreline projected
- Rich properties per feature
- Disimpan ke Drive output/ dan juga /content/

```python
# ================================================================
# CELL 13 — GeoJSON export at 2025 / 2030 / 2035
# ================================================================
import json as _json
from datetime import timezone

def lrr_project_contour(rows_base, cols_base, bbox, lrr_m_yr, delta_t_yr, direction=None):
    """
    Project shoreline outward/inward based on LRR rate and delta time.
    Shifts contour points in the cross-shore direction.
    direction = perpendicular unit vector per pixel (approximate: move in col direction).
    """
    if rows_base is None:
        return None, None
    shift_m   = lrr_m_yr * delta_t_yr
    shift_px  = shift_m / SCALE_M
    # Simple: shift columns by shift_px (cross-shore approximation)
    new_cols  = np.array(cols_base, dtype=float) + shift_px
    new_cols  = np.clip(new_cols, 0, W_PIX - 1)
    return np.array(rows_base, dtype=float), new_cols


def make_geojson_feature(aoi, year, season, source, rows, cols, bbox, properties_extra):
    if rows is None or len(rows) < 2:
        return None
    west, south, east, north = bbox
    coords = [
        [round(west + (c / W_PIX) * (east - west), 6),
         round(north - (r / H_PIX) * (north - south), 6)]
        for r, c in zip(rows, cols)
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {
            "aoi":    aoi,
            "year":   year,
            "season": season,
            "source": source,
            **properties_extra,
        }
    }


# Build per-AOI LRR average (for projection)
aoi_lrr = {}
if len(transect_df) > 0:
    for aoi, grp in transect_df.groupby("aoi"):
        aoi_lrr[aoi] = float(grp["lrr_m_per_yr"].mean(skipna=True))

run_utc = datetime.now(tz=timezone.utc).isoformat()

for snap_year in GEOJSON_SNAPSHOT_YEARS:
    features = []
    for aoi, results in rollout_results.items():
        bbox = None
        if "aoi" in params and isinstance(params["aoi"], dict):
            bbox = params["aoi"].get(aoi, {}).get("bbox")
        if bbox is None:
            continue

        # Get per-AOI summary metrics from transect_df
        aoi_tdf = transect_df[transect_df["aoi"] == aoi] if len(transect_df) > 0 else pd.DataFrame()
        mean_epr = float(aoi_tdf["epr_m_per_yr"].mean(skipna=True)) if len(aoi_tdf) > 0 else None
        mean_nsm = float(aoi_tdf["nsm_m"].mean(skipna=True)) if len(aoi_tdf) > 0 else None
        mean_sce = float(aoi_tdf["sce_m"].mean(skipna=True)) if len(aoi_tdf) > 0 else None
        mean_lrr = float(aoi_tdf["lrr_m_per_yr"].mean(skipna=True)) if len(aoi_tdf) > 0 else None
        mc_unc   = float(aoi_tdf["mc_uncertainty"].mean(skipna=True)) if len(aoi_tdf) > 0 else None

        clf_counts = {}
        if len(aoi_tdf) > 0:
            for k, v in aoi_tdf["classification"].value_counts().items():
                clf_counts[k] = int(v)

        # Model rollout contour at snap_year (use S2 = mid-year)
        snap_result = next(
            (r for r in results if r["year"] == snap_year and r["season"] == "S2"), None
        )
        if snap_result and snap_result["contour_row"] is not None:
            model_feat = make_geojson_feature(
                aoi, snap_year, "S2", "model_rollout",
                snap_result["contour_row"], snap_result["contour_col"], bbox,
                {
                    "mean_epr_m_per_yr":    round(mean_epr, 4) if mean_epr else None,
                    "mean_nsm_m":           round(mean_nsm, 3) if mean_nsm else None,
                    "mean_sce_m":           round(mean_sce, 3) if mean_sce else None,
                    "mean_lrr_m_per_yr":    round(mean_lrr, 4) if mean_lrr else None,
                    "mc_uncertainty_mean":  round(mc_unc, 5) if mc_unc else None,
                    "classification_counts": clf_counts,
                    "run_utc":              run_utc,
                    "model_source":         f"convlstm_unet_{RUN_SUFFIX}",
                }
            )
            if model_feat:
                features.append(model_feat)

        # LRR-projected contour at snap_year (extrapolate from last GT)
        if aoi in aoi_frames and aoi_frames[aoi]:
            last_t, last_mask = aoi_frames[aoi][-1]
            rows_gt, cols_gt  = extract_contour(last_mask[0].cpu().numpy())
            delta_t           = snap_year + SEASON_MONTHS["S2"] / 12.0 - last_t
            lrr_rate          = aoi_lrr.get(aoi, 0.0)
            rows_lrr, cols_lrr = lrr_project_contour(rows_gt, cols_gt, bbox, lrr_rate, delta_t)
            lrr_feat = make_geojson_feature(
                aoi, snap_year, "S2", "lrr_projection",
                rows_lrr, cols_lrr, bbox,
                {
                    "lrr_m_per_yr":  round(lrr_rate, 4) if not np.isnan(lrr_rate) else None,
                    "delta_t_yr":    round(delta_t, 2),
                    "run_utc":       run_utc,
                }
            )
            if lrr_feat:
                features.append(lrr_feat)

    geojson = {
        "type":     "FeatureCollection",
        "metadata": {
            "generated_utc": run_utc,
            "model_source":  f"convlstm_unet_{RUN_SUFFIX}_{run_id}",
            "snapshot_year": snap_year,
        },
        "features": [f for f in features if f is not None],
    }

    fname = f"shoreline_{snap_year}_{RUN_SUFFIX}.geojson"
    local_path  = f"/content/{fname}"
    drive_path  = os.path.join(OUTPUT_DIR, fname)

    for path in [local_path, drive_path]:
        with open(path, "w") as fp:
            _json.dump(geojson, fp, indent=2)

    n_feat = len(geojson["features"])
    log.info("GeoJSON %d: %d features → %s", snap_year, n_feat, drive_path)
    print(f"  {snap_year}: {n_feat} features saved")

print("\nGeoJSON snapshots saved:")
for yr in GEOJSON_SNAPSHOT_YEARS:
    print(f"  shoreline_{yr}_{RUN_SUFFIX}.geojson")
```

---

## Cell 14 (2aug) — Sanity Bootstrap Push *(one-time from notebook; CI takes over later)*

New cell after Cell 13.

Perubahan:
- Push `shorelineForecast` documents (rich schema) ke Sanity via HTTP mutate API
- Satu doc per AOI per tahun-season (deterministic _id → UPSERT aman)
- Semua data tersedia untuk frontend: timestamp, EPR, NSM, SCE, LRR, per-transect, uncertainty, GeoJSON inline
- **Catatan**: cell ini hanya untuk bootstrap pertama kali. CI workflow (`.github/workflows/recalculate.yml`) akan menggantikan ini setelah model dipromote ke checkpoint resmi.

> Sebelum jalankan cell ini, tambahkan ke environment (atau ke Drive .env):
> - `SANITY_PROJECT_ID`
> - `SANITY_DATASET`
> - `SANITY_API_TOKEN`

```python
# ================================================================
# CELL 14 — Sanity bootstrap push (shorelineForecast document type)
# NOTE: untuk CI, lihat src/output/sanity_push.py — cell ini hanya
#       untuk first-time bootstrap dari notebook.
# ================================================================
import requests

# Load Sanity credentials from environment or Drive .env
for key in ["SANITY_PROJECT_ID", "SANITY_DATASET", "SANITY_API_TOKEN"]:
    if key not in os.environ and key in _env_vars:
        os.environ[key] = _env_vars[key]

missing = [k for k in ["SANITY_PROJECT_ID", "SANITY_DATASET", "SANITY_API_TOKEN"]
           if not os.environ.get(k)]
if missing:
    print(f"SKIP: missing env vars {missing}")
    print("Add to Drive .env or set via: os.environ['SANITY_PROJECT_ID'] = '...'")
else:
    SANITY_API_VERSION = os.environ.get("SANITY_API_VERSION", "2024-01-01")
    project_id = os.environ["SANITY_PROJECT_ID"]
    dataset    = os.environ["SANITY_DATASET"]
    token      = os.environ["SANITY_API_TOKEN"]

    url = f"https://{project_id}.api.sanity.io/v{SANITY_API_VERSION}/data/mutate/{dataset}"

    def push_batch(mutations, label=""):
        resp = requests.post(
            url,
            json={"mutations": mutations},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        resp.raise_for_status()
        log.info("Sanity push OK: %d docs  %s", len(mutations), label)
        return resp.json()

    # Group transect records by AOI for fast lookup
    transects_by_aoi = {}
    for _, row in transect_df.iterrows():
        transects_by_aoi.setdefault(row["aoi"], []).append(row.to_dict())

    # Build all documents
    all_mutations = []
    for aoi, results in rollout_results.items():
        bbox = None
        if "aoi" in params and isinstance(params["aoi"], dict):
            bbox = params["aoi"].get(aoi, {}).get("bbox")

        aoi_transects = transects_by_aoi.get(aoi, [])
        aoi_tdf       = transect_df[transect_df["aoi"] == aoi] if len(transect_df) > 0 else pd.DataFrame()

        mean_epr = float(aoi_tdf["epr_m_per_yr"].mean(skipna=True)) if len(aoi_tdf) > 0 else None
        mean_nsm = float(aoi_tdf["nsm_m"].mean(skipna=True))       if len(aoi_tdf) > 0 else None
        mean_sce = float(aoi_tdf["sce_m"].mean(skipna=True))       if len(aoi_tdf) > 0 else None
        mean_lrr = float(aoi_tdf["lrr_m_per_yr"].mean(skipna=True)) if len(aoi_tdf) > 0 else None
        mc_unc   = float(aoi_tdf["mc_uncertainty"].mean(skipna=True)) if len(aoi_tdf) > 0 else None

        clf_counts = {}
        if len(aoi_tdf) > 0:
            for k, v in aoi_tdf["classification"].value_counts().items():
                clf_counts[str(k)] = int(v)

        # Serializable transect list (no numpy types)
        clean_transects = []
        for t in aoi_transects:
            clean_transects.append({
                "id":           int(t["transect_id"]),
                "originLon":    float(t["origin_lon"]),
                "originLat":    float(t["origin_lat"]),
                "eprMPerYr":    float(t["epr_m_per_yr"]) if t["epr_m_per_yr"] is not None else None,
                "nsmM":         float(t["nsm_m"])        if t["nsm_m"] is not None else None,
                "sceM":         float(t["sce_m"])        if t["sce_m"] is not None else None,
                "lrrMPerYr":    float(t["lrr_m_per_yr"]) if t["lrr_m_per_yr"] is not None else None,
                "classification": str(t["classification"]),
                "mcUncertainty": float(t["mc_uncertainty"]),
                "modelPositions": {str(k): float(v) for k, v in (t.get("model_positions") or {}).items()},
            })

        # One document per step
        for r in results:
            yr, seas = r["year"], r["season"]
            doc_id   = f"shorelineForecast-{aoi}-{yr}-{seas}"

            # Inline GeoJSON for this step's contour
            geojson_inline = None
            if bbox and r["contour_row"] is not None:
                west, south, east, north = bbox
                coords = [
                    [round(west + (c / W_PIX) * (east - west), 6),
                     round(north - (rr / H_PIX) * (north - south), 6)]
                    for rr, c in zip(r["contour_row"], r["contour_col"])
                ]
                geojson_inline = {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"aoi": aoi, "year": yr, "season": seas, "source": "model_rollout"},
                }

            doc = {
                "_type":                 "shorelineForecast",
                "_id":                   doc_id,
                "aoi":                   aoi,
                "year":                  yr,
                "season":                seas,
                "periodT":               round(r["t"], 4),
                "runUtc":                run_utc,
                "modelSource":           f"convlstm_unet_{RUN_SUFFIX}_{run_id}",
                "mcUncertaintyMean":     round(float(r["unc_mean"]), 6),
                "geojson":               geojson_inline,
                # AOI-level summaries (same across all seasons for this AOI)
                "meanEprMPerYr":         round(mean_epr, 4) if mean_epr and not np.isnan(mean_epr) else None,
                "meanNsmM":              round(mean_nsm, 3) if mean_nsm and not np.isnan(mean_nsm) else None,
                "meanSceM":              round(mean_sce, 3) if mean_sce and not np.isnan(mean_sce) else None,
                "meanLrrMPerYr":         round(mean_lrr, 4) if mean_lrr and not np.isnan(mean_lrr) else None,
                "classificationCounts":  clf_counts,
                "nTransects":            len(clean_transects),
                "transects":             clean_transects,
            }
            all_mutations.append({"createOrReplace": doc})

    log.info("Prepared %d shorelineForecast documents across %d AOIs",
             len(all_mutations), len(rollout_results))

    # Push in batches of 100
    BATCH = 100
    pushed = 0
    for start in range(0, len(all_mutations), BATCH):
        batch    = all_mutations[start : start + BATCH]
        push_batch(batch, label=f"batch {start//BATCH + 1}")
        pushed  += len(batch)
        log.info("Pushed %d / %d", pushed, len(all_mutations))

    print(f"\nDone. Pushed {pushed} shorelineForecast documents to Sanity ({project_id}/{dataset})")
    print(f"Document IDs pattern: shorelineForecast-{{aoi}}-{{year}}-{{season}}")
```

---

## Sanity Schema Definition (tambahkan ke CMS)

Tambahkan file baru di `tourism-kemujan/studio/schemaTypes/shorelineForecast.js`:

```javascript
// studio/schemaTypes/shorelineForecast.js
import { defineField, defineType } from 'sanity'

export default defineType({
  name: 'shorelineForecast',
  title: 'Shoreline Forecast',
  type: 'document',
  fields: [
    defineField({ name: 'aoi',          title: 'AOI ID',         type: 'string' }),
    defineField({ name: 'year',         title: 'Year',           type: 'number' }),
    defineField({ name: 'season',       title: 'Season',         type: 'string' }),
    defineField({ name: 'periodT',      title: 'Period (t)',     type: 'number' }),
    defineField({ name: 'runUtc',       title: 'Run UTC',        type: 'string' }),
    defineField({ name: 'modelSource',  title: 'Model Source',   type: 'string' }),
    defineField({ name: 'mcUncertaintyMean', title: 'MC Uncertainty Mean', type: 'number' }),
    defineField({
      name: 'geojson', title: 'GeoJSON Shoreline', type: 'object',
      fields: [
        defineField({ name: 'type',     type: 'string' }),
        defineField({ name: 'geometry', type: 'object',
          fields: [
            defineField({ name: 'type',        type: 'string' }),
            defineField({ name: 'coordinates', type: 'array', of: [{ type: 'array', of: [{ type: 'number' }] }] }),
          ]
        }),
        defineField({ name: 'properties', type: 'object',
          fields: [
            defineField({ name: 'aoi',    type: 'string' }),
            defineField({ name: 'year',   type: 'number' }),
            defineField({ name: 'season', type: 'string' }),
            defineField({ name: 'source', type: 'string' }),
          ]
        }),
      ]
    }),
    defineField({ name: 'meanEprMPerYr',    title: 'Mean EPR (m/yr)',    type: 'number' }),
    defineField({ name: 'meanNsmM',         title: 'Mean NSM (m)',       type: 'number' }),
    defineField({ name: 'meanSceM',         title: 'Mean SCE (m)',       type: 'number' }),
    defineField({ name: 'meanLrrMPerYr',    title: 'Mean LRR (m/yr)',    type: 'number' }),
    defineField({ name: 'nTransects',       title: 'N Transects',        type: 'number' }),
    defineField({
      name: 'classificationCounts', title: 'Classification Counts', type: 'object',
      fields: ['Erosi Parah','Erosi','Stabil','Akresi','Akresi Kuat'].map(c =>
        defineField({ name: c.replace(/ /g, '_'), title: c, type: 'number' })
      )
    }),
    defineField({
      name: 'transects', title: 'Transects', type: 'array',
      of: [{
        type: 'object',
        fields: [
          defineField({ name: 'id',             type: 'number' }),
          defineField({ name: 'originLon',      type: 'number' }),
          defineField({ name: 'originLat',      type: 'number' }),
          defineField({ name: 'eprMPerYr',      type: 'number' }),
          defineField({ name: 'nsmM',           type: 'number' }),
          defineField({ name: 'sceM',           type: 'number' }),
          defineField({ name: 'lrrMPerYr',      type: 'number' }),
          defineField({ name: 'classification', type: 'string' }),
          defineField({ name: 'mcUncertainty',  type: 'number' }),
        ]
      }]
    }),
  ],
  preview: {
    select: { title: 'aoi', subtitle: 'year' },
    prepare: ({ title, subtitle }) => ({ title: `${title} — ${subtitle}` }),
  },
})
```

Daftarkan di `studio/schemaTypes/index.js`:

```javascript
// tambahkan import dan masukkan ke array schemas
import shorelineForecast from './shorelineForecast'
// ...
export const schemaTypes = [...existingTypes, shorelineForecast]
```

---

## CI Workflow Update (setelah model di-promote)

Update `.github/workflows/recalculate.yml` — tambahkan step Sanity push setelah inference:

```yaml
# tambahkan ke jobs.run.steps setelah step inference
- name: Push to Sanity
  env:
    SANITY_PROJECT_ID: ${{ secrets.SANITY_PROJECT_ID }}
    SANITY_DATASET:    ${{ secrets.SANITY_DATASET }}
    SANITY_API_TOKEN:  ${{ secrets.SANITY_API_TOKEN }}
  run: python -m src.output.sanity_push --run-dir data/interim/run_${{ steps.run_date.outputs.date }}
```

Tambahkan GitHub Secrets di repo settings:
- `SANITY_PROJECT_ID`
- `SANITY_DATASET`
- `SANITY_API_TOKEN`

---

## Commands to Run (manual — jangan dijalankan di sandbox)

```bash
# 1. Setelah training selesai — verifikasi checkpoint di Drive
#    (jalankan di Colab cell baru)
import os
MODEL_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/models"
print([f for f in os.listdir(MODEL_DIR) if "2aug" in f])

# 2. Verifikasi GeoJSON
OUTPUT_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/output"
print([f for f in os.listdir(OUTPUT_DIR) if "2aug" in f])

# 3. Setelah Sanity push — cek di Sanity Studio (GROQ):
# *[_type == "shorelineForecast"] | order(year asc) {aoi, year, season, meanEprMPerYr}
```
