# claude_result10.md — Balik ke `train_256.ipynb` (stabil, in_ch=1) + W&B + Sanity push

## Kenapa balik ke sini

Pipeline "2aug" (multichannel + Sentinel-only, `claude_result9.md`) ternyata gak stabil pas
rollout — divergence parah (garis kontur ngelantur motong daratan), kemungkinan besar
karena channel statis (elevation/slope/dll) belum dinormalisasi dan datanya cuma n=130.
Waktu udah mepet buat debug lebih jauh, jadi **keputusan: balik ke `notebooks/train_256.ipynb`
apa adanya** — pipeline lama ini `in_ch=1` (mask-only, gak ada channel statis), pakai
seluruh data Landsat+Sentinel gabungan dari `offline_256_adaptive.zip` (bukan `tensors.npz`
yang skemanya bermasalah), `TRAIN_UNTIL=2022`, `epochs=200`. Ini yang dulu udah pernah
jalan dan dianalisis (catatan diagnosis "rollout collapse" ada di cell markdown terakhir
notebook ini) — collapse itu masalah kualitas prediksi jangka panjang, **bukan** crash/
divergence kayak yang baru terjadi. Untuk deadline sekarang, stabil > canggih.

**Yang ditambahkan** di atas notebook asli: W&B logging pas training, dan push ke Sanity
di akhir (DSAS/EPR/LRR per AOI + ringkasan training) — schema Sanity-nya **reuse yang
sudah didefinisikan** di `claude_result3.md`/`claude_result9.md`, tidak ada schema baru.

**Catatan jujur soal keterbatasan**: model di notebook ini **tidak punya MC Dropout**
(`ConvLSTMUNet` di sini gak ada `nn.Dropout2d` sama sekali) — jadi **tidak ada
uncertainty quantification** di versi ini. Field uncertainty di skema Sanity akan
dikosongin (null), bukan diisi asal-asalan. Ini trade-off yang sadar diambil demi
kestabilan & waktu, bukan kelupaan.

---

## Cell 1 — SAMA PERSIS

Copy **cell index 1** dari `notebooks/train_256.ipynb` apa adanya (mount Drive + extract
`offline_256_adaptive.zip` ke `/content/bundle_256`).

## Cell 2 — SAMA PERSIS

Copy **cell index 3** apa adanya (set `BUNDLE_DIR`, `MODEL_DIR`, `LOG_DIR`).

## Cell 3 — SAMA PERSIS

Copy **cell index 4** apa adanya:
```python
masks, manifest = load_offline(BUNDLE_DIR)
masks.pop('Titik_19', None)   # exclude — hapus baris ini kalau mau include
print(f"AOI final: {list(masks.keys())}")
```

## Cell 4 — DIUBAH — Training + W&B (dari cell index 5)

Copy **cell index 5** ("CELL 2 — ConvLSTM-UNet Training") sebagai basis, dengan 3 perubahan:

1. **Hapus** baris `masks, manifest = load_offline(BUNDLE_DIR)` di dalam cell ini —
   cell ini aslinya reload ulang `masks` dari nol, yang secara diam-diam **membatalkan**
   exclude `Titik_19` yang udah dilakukan di Cell 3. Pakai `masks`/`manifest` yang udah
   ada di memory dari Cell 3.
2. **Tambah W&B**: init sebelum training, log per-epoch di dalam `train_loop`, upload
   plot & `wandb.finish()` di akhir.
3. Semua yang lain — definisi model, loss, hyperparameter (`TRAIN_UNTIL=2022`,
   `epochs=200`, `batch_size=4`, `in_ch=1`, `base_ch=16`) — **tidak berubah sama sekali**
   dari notebook asli.

```python
# ============================================================
# CELL 2 — ConvLSTM-UNet Training (256×256 input) + W&B
# ============================================================
import os, json, logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

PATCH_SIZE = 256
LOOKBACK   = 3
TRAIN_UNTIL = 2022
SEASON_ORDER = ['S1', 'S2', 'S3']

# NOTE: masks/manifest sudah di-load di Cell 3 (dengan Titik_19 di-exclude) —
# TIDAK di-reload di sini supaya exclude-nya tetap berlaku.

# ================== BUILD TENSOR ==================
def seq_index(yr, s):
    return yr * 3 + SEASON_ORDER.index(s)

def build_tensor(masks, lookback=LOOKBACK):
    X_list, y_list, meta = [], [], []
    for nama, frames in masks.items():
        keys = sorted(frames.keys(), key=lambda k: seq_index(*k))
        for i in range(len(keys) - lookback):
            win, target = keys[i:i+lookback], keys[i+lookback]
            idx = [seq_index(*k) for k in win + [target]]
            if idx != list(range(idx[0], idx[0] + lookback + 1)):
                continue
            X_list.append(np.stack([frames[k][np.newaxis] for k in win]))
            y_list.append(frames[target][np.newaxis])
            meta.append({'aoi': nama, 'input': win, 'target': target,
                         'target_year': target[0]})
    X = np.stack(X_list)
    y = np.stack(y_list)
    print(f"Tensor: X{X.shape} y{y.shape} | {len(meta)} sample dari {len(masks)} AOI")
    return X, y, meta

X, y, meta = build_tensor(masks)

# Split temporal
tr = [i for i, m in enumerate(meta) if m['target_year'] <= TRAIN_UNTIL]
te = [i for i, m in enumerate(meta) if m['target_year'] > TRAIN_UNTIL]
X_train = torch.from_numpy(X[tr]).float().to(device)
y_train = torch.from_numpy(y[tr]).float().to(device)
X_test  = torch.from_numpy(X[te]).float().to(device)
y_test  = torch.from_numpy(y[te]).float().to(device)
meta_train = [meta[i] for i in tr]
meta_test  = [meta[i] for i in te]
print(f"Train: {len(tr)} sample (target ≤ {TRAIN_UNTIL}) | Test: {len(te)} sample")

# ================== PERSISTENCE BASELINE ==================
def dice_score(pred, target, eps=1e-6):
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2*inter + eps) / (pred.sum(1) + target.sum(1) + eps)

persist_pred = X_test[:, -1]
dice_baseline = dice_score(persist_pred, y_test).mean().item()
print(f"Persistence baseline Dice: {dice_baseline:.4f}")

# ================== MODEL: ConvLSTM-UNet ==================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4*hidden_ch, kernel_size, padding=kernel_size//2)
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
    """FIXED: skip connection pakai enc1 (resolusi H,W - MATCH sama up),
    bukan enc2_last (resolusi H/2 - itu penyebab shape mismatch)."""
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1), nn.BatchNorm2d(base_ch*2), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.clstm = ConvLSTMCell(base_ch*2, base_ch*2, 3)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch*3, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU())
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

# ================== LOSS ==================
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
        dice = (2*inter + 1e-6) / (prob_flat.sum(1) + target_flat.sum(1) + 1e-6)
        dice_loss = 1 - dice.mean()
        bce_loss = self.bce(logit, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

# ================== W&B INIT ==================
import wandb

DRIVE_BASE = "/content/drive/MyDrive/Data_experiment_shoreline"
ENV_PATH   = os.path.join(DRIVE_BASE, ".env")   # berisi WANDB_API_KEY, opsional

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
else:
    print(f"WANDB_API_KEY tidak ada di {ENV_PATH} — akan prompt login manual kalau perlu")

# ================== TRAINING LOOP ==================
model = ConvLSTMUNet(in_ch=1, base_ch=16).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{run_id}.log"))
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler())

wandb.init(
    project="shoreline-kemujan",
    name=f"run_train256_{run_id}",
    config={
        "epochs": 200, "batch_size": 4, "lr": 1e-3, "base_ch": 16, "in_ch": 1,
        "train_until": TRAIN_UNTIL, "lookback": LOOKBACK,
        "n_train": len(tr), "n_test": len(te), "aoi_list": list(masks.keys()),
        "pipeline": "train_256_stable_fallback",
    },
    tags=["train256", "mask-only", "stable-fallback"],
)
wandb.log({"dice/persistence_baseline": dice_baseline})

def train_loop(model, X_tr, y_tr, X_te, y_te, epochs=150, batch_size=4):
    history = {'train_loss': [], 'train_dice': [], 'test_dice': []}
    n = len(X_tr)
    logger.info(f"Training: {n} train | {len(X_te)} test | epochs={epochs} | batch={batch_size}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            tr_pred = (torch.sigmoid(model(X_tr)) > 0.5).float()
            te_pred = (torch.sigmoid(model(X_te)) > 0.5).float()
            tr_dice = dice_score(tr_pred, y_tr).mean().item()
            te_dice = dice_score(te_pred, y_te).mean().item()
        model.train()

        history['train_loss'].append(epoch_loss)
        history['train_dice'].append(tr_dice)
        history['test_dice'].append(te_dice)

        wandb.log({
            "loss/train": epoch_loss, "dice/train": tr_dice, "dice/test": te_dice,
            "epoch": epoch,
        })

        if epoch % 20 == 0 or epoch == epochs-1:
            logger.info(f"Epoch {epoch:3d} | loss={epoch_loss:.4f} | tr={tr_dice:.4f} | te={te_dice:.4f}")

    return history


history = train_loop(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=4)

# ================== SAVE ==================
model_path = os.path.join(MODEL_DIR, f"convlstm_unet_256_{run_id}.pth")
torch.save(model.state_dict(), model_path)
np.savez_compressed(os.path.join(LOG_DIR, f"history_{run_id}.npz"), **history)
print(f"\n✓ Model: {model_path}")
print(f"✓ History: {os.path.join(LOG_DIR, f'history_{run_id}.npz')}")

# ================== PLOT ==================
epochs_range = range(len(history['train_loss']))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(epochs_range, history['train_loss'], color='tab:blue')
axes[0].set_title('Training Loss (Dice+BCE)'); axes[0].set_xlabel('epoch'); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history['train_dice'], label='train', color='tab:green')
axes[1].plot(epochs_range, history['test_dice'], label='test', color='tab:orange')
axes[1].axhline(dice_baseline, color='red', ls='--', label=f'persistence ({dice_baseline:.3f})')
axes[1].set_title('Dice Score'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

gap = np.array(history['train_dice']) - np.array(history['test_dice'])
axes[2].plot(epochs_range, gap, color='tab:red')
axes[2].axhline(0.15, color='gray', ls=':', label='overfit threshold')
axes[2].set_title('Train-Test Gap'); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(LOG_DIR, f"training_{run_id}.png")
plt.savefig(plot_path, dpi=120)
plt.show()

final_gap = history['train_dice'][-1] - history['test_dice'][-1]
print(f"\n{'='*50}")
print(f"Final train Dice: {history['train_dice'][-1]:.4f}")
print(f"Final test Dice:  {history['test_dice'][-1]:.4f}")
print(f"Persistence:      {dice_baseline:.4f}")
print(f"Train-test gap:   {final_gap:.4f} {'⚠ overfit' if final_gap > 0.15 else '✓ wajar'}")
print(f"Model {'✓ MENGALAHKAN' if history['test_dice'][-1] > dice_baseline else '✗ KALAH DARI'} persistence")
print(f"{'='*50}")

wandb.log({"chart/training": wandb.Image(plot_path),
           "dice/best_test": max(history['test_dice'])})
wandb.finish()
print("\n✓ Cell 4 selesai (training + W&B).")
```

## Cell 5 — SAMA PERSIS

Copy **cell index 6** apa adanya (`from scipy.interpolate import splprep, splev`).

## Cell 6 — SAMA PERSIS (opsional, redundant)

Copy **cell index 8** apa adanya kalau mau — ini cuma re-save + verifikasi checkpoint yang
**sudah** disimpan Cell 4 (path & isinya identik, `run_id` global sama). Aman dilewatin.

## Cell 7 — SAMA PERSIS

Copy **cell index 10** apa adanya (`!pip install contextily -q`).

## Cell 8 — SAMA PERSIS + 1 baris fix (dari cell index 11)

Copy **cell index 11** ("CELL 3 — Forecast Rollout + Transect EPR + Basemap") apa adanya,
**kecuali** tambahkan 1 baris exclude `Titik_19` supaya konsisten dengan training —
cell aslinya reload `masks` dari nol tanpa exclude, jadi AOI yang gak pernah dilatih
diam-diam ikut dianalisis:

Cari baris ini:
```python
masks, manifest = load_offline(BUNDLE_DIR)
AOI_CONFIG = {
```
Tambahkan tepat di bawahnya (sebelum `AOI_CONFIG = {`):
```python
masks.pop('Titik_19', None)   # konsisten dengan exclude di Cell 3 (training)
```

Selebihnya (definisi `ConvLSTMUNet` ulang, `rollout_forecast`, `full_analysis_plot`, loop
`for nama in AOI_CONFIG`) **copy persis apa adanya**.

## Cell 9 — SAMA PERSIS

Copy **cell index 12** ("CELL (REVISI) — Grid Per Tahun — EXTENDED") apa adanya. Ini
menghasilkan `all_metrics` (dict `{aoi: df_metrics}`, EPR per transect per tahun) yang
dipakai Sanity push di bawah — **tidak perlu diubah**.

## Cell 10 — SAMA PERSIS + loop semua AOI (dari cell index 16)

Copy **cell index 16** ("CELL — Explainability & Validasi") apa adanya **kecuali** bagian
paling akhir (`nama_target = list(AOI_CONFIG.keys())[0]` dst) — versi asli cuma jalan buat
1 AOI. Ganti blok akhir itu jadi loop semua AOI, biar `lrr_vs_model` punya data lengkap
buat push ke Sanity:

Ganti:
```python
# ---------- jalankan ----------
nama_target = list(AOI_CONFIG.keys())[0]   # <-- ganti manual kalau perlu
TARGET_YEAR_CHECK = 2035                    # <-- coba tahun yg lebih dekat dulu drpd 2043

changes = diagnose_rollout_collapse(nama_target, masks, model, target_year=TARGET_YEAR_CHECK)
transects, lrr_results, model_dist = compute_lrr_and_compare(
    nama_target, masks, model, target_year=TARGET_YEAR_CHECK)
```

Jadi:
```python
# ---------- jalankan untuk semua AOI ----------
TARGET_YEAR_CHECK = 2035

collapse_diag = {}
lrr_vs_model  = {}
for nama_target in AOI_CONFIG:
    if nama_target not in masks:
        continue
    print(f"\n### {nama_target} ###")
    collapse_diag[nama_target] = diagnose_rollout_collapse(
        nama_target, masks, model, target_year=TARGET_YEAR_CHECK)
    lrr_vs_model[nama_target] = compute_lrr_and_compare(
        nama_target, masks, model, target_year=TARGET_YEAR_CHECK)
```

**Sebelum lanjut ke push Sanity di bawah**, cek dulu print output `diagnose_rollout_collapse`
tiap AOI — kalau `avg_rollout` jauh lebih GEDE dari `avg_seed` (bukan lebih kecil), berarti
divergence masih ada juga di pipeline lama ini, dan angka yang mau di-push ke Sanity gak
reliable. Kalau `avg_rollout` lebih kecil/sebanding (collapse ringan seperti yang
didiagnosis sebelumnya di notebook ini) — itu ekspektasi yang sudah diketahui, aman
dilanjut dengan disclaimer horizon pendek.

---

## Cell 11 — BARU — Push DSAS/EPR/LRR **+ geojson per tahun** ke Sanity

Reuse skema `shorelineForecast` (sekarang sudah beneran ada filenya di
`tourism-kemujan/studio/schemaTypes/documents/shorelineForecast.ts`) — tidak ada schema
baru. **Revisi dari draft sebelumnya**: versi awal cuma push 1 dokumen snapshot tanpa
geometri sama sekali — itu gak cukup buat nampilin garis pantai per tahun di frontend.
Versi ini push **1 dokumen per AOI per TAHUN** (dari `baseline_year` sampai
`TARGET_YEAR_CHECK`), masing-masing punya:
- **geometri asli** (GeoJSON LineString, lon/lat) — real kalau tahun itu ≤ tahun terakhir
  data GT, rollout kalau proyeksi. Ini yang dulu bolong.
- EPR per-transect **di tahun itu** (bukan cuma tahun terakhir) — diambil dari baris
  `all_metrics[aoi]` yang sesuai, jadi konsisten sama Cell 9.
- `classificationCounts`/`modelPositions` dalam bentuk **array-of-pairs**
  (`[{label, count}]` / `[{key, value}]`), bukan dict — field Sanity `object` butuh key
  yang di-declare di schema, gak bisa nerima key dinamis.

```python
# ================================================================
# CELL 11 — Push DSAS/EPR/LRR + geojson PER TAHUN ke Sanity (shorelineForecast)
# 1 dokumen per AOI per tahun (baseline_year..TARGET_YEAR_CHECK), real+rollout.
# ================================================================
import requests
from datetime import datetime, timezone

for key in ["SANITY_PROJECT_ID", "SANITY_DATASET", "SANITY_API_TOKEN"]:
    if key not in os.environ and key in _env_vars:
        os.environ[key] = _env_vars[key]

missing = [k for k in ["SANITY_PROJECT_ID", "SANITY_DATASET", "SANITY_API_TOKEN"]
           if not os.environ.get(k)]
if missing:
    print(f"SKIP: missing env vars {missing}")
else:
    SANITY_API_VERSION = os.environ.get("SANITY_API_VERSION", "2024-01-01")
    project_id = os.environ["SANITY_PROJECT_ID"]
    dataset    = os.environ["SANITY_DATASET"]
    token      = os.environ["SANITY_API_TOKEN"]
    url = f"https://{project_id}.api.sanity.io/v{SANITY_API_VERSION}/data/mutate/{dataset}"

    def push_batch(mutations, label=""):
        resp = requests.post(url, json={"mutations": mutations},
                              headers={"Authorization": f"Bearer {token}"}, timeout=60)
        resp.raise_for_status()
        print(f"Sanity push OK: {len(mutations)} docs  {label}")
        return resp.json()

    run_utc = datetime.now(tz=timezone.utc).isoformat()

    CHANGE_BINS = [
        (-float("inf"), -2.0, "Erosi Parah"), (-2.0, -0.5, "Erosi"),
        (-0.5, 0.5, "Stabil"), (0.5, 2.0, "Akresi"), (2.0, float("inf"), "Akresi Kuat"),
    ]
    def classify(epr):
        if epr is None or np.isnan(epr):
            return "Tidak valid"
        for lo, hi, label in CHANGE_BINS:
            if lo <= epr < hi:
                return label
        return "Tidak valid"

    def pick_repr(season_dict, priority=('S2', 'S3', 'S1')):
        for s in priority:
            if s in season_dict and season_dict[s] is not None:
                return season_dict[s], s
        for s, v in season_dict.items():
            if v is not None:
                return v, s
        return None, None

    mutations = []
    for aoi, df_metrics in all_metrics.items():
        if df_metrics is None or df_metrics.empty:
            continue
        if aoi not in masks or aoi not in AOI_CONFIG:
            continue

        lon_c, lat_c = AOI_CONFIG[aoi]['coord']
        baseline_year = df_metrics.index[0]
        last_year     = df_metrics.index[-1]

        # ---------- bangun ulang mask per tahun (real + rollout), sama
        # persis pola yang dipakai yearly_grid_plot_extended (Cell 9) ----------
        frames = masks[aoi]
        keys_sorted = sorted(frames.keys(), key=lambda k: seq_index(*k))
        last_key = keys_sorted[-1]
        last_real_year = last_key[0]

        gt_by_year = {}
        for (yr, s), m in frames.items():
            gt_by_year.setdefault(yr, {})[s] = m

        n_steps = max((TARGET_YEAR_CHECK - last_real_year) * 3, 0)
        future_labels = next_n_labels(last_key, n_steps) if n_steps > 0 else []
        seed = [frames[k] for k in keys_sorted[-LOOKBACK:]]
        future_preds = rollout_forecast(model, seed, n_steps) if n_steps > 0 else []
        future_by_year = {}
        for (yr, s), pred in zip(future_labels, future_preds):
            future_by_year.setdefault(yr, {})[s] = pred

        lrr_data = lrr_vs_model.get(aoi)
        lrr_results = lrr_data[1] if lrr_data else {}   # {tid: {"rate_m_per_year", "pred_dist", "n_obs"}}
        lrr_rates = [v["rate_m_per_year"] for v in lrr_results.values() if v]
        mean_lrr  = float(np.mean(lrr_rates)) if lrr_rates else None

        # sce (Shoreline Change Envelope) = properti seluruh timeline, sama
        # nilainya di tiap dokumen tahun (bukan properti per-tahun)
        sce_full = (df_metrics.max() - df_metrics.min())

        for year in df_metrics.index:
            is_real = year in gt_by_year
            season_dict = gt_by_year.get(year) or future_by_year.get(year) or {}
            mask_np, s_used = pick_repr(season_dict)
            if mask_np is None:
                continue

            contour = extract_main_contour(mask_np, use_spline=True, spline_smoothing=2.0)
            if contour is None:
                continue
            coords = [
                [round(lo, 6), round(la, 6)]
                for lo, la in (pixel_to_lonlat(r, c, lon_c, lat_c) for r, c in contour)
            ]
            geojson_feat = {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "aoi": aoi, "year": int(year), "season": s_used,
                    "source": "real" if is_real else "rollout",
                },
            }

            row  = df_metrics.loc[year]                       # EPR per transect DI TAHUN INI
            dt   = max(year - baseline_year, 0)

            transects_out = []
            for tid_str in df_metrics.columns:
                tid = int(tid_str.replace("T", ""))
                epr_val = float(row[tid_str]) if not np.isnan(row[tid_str]) else None
                nsm_val = epr_val * dt if epr_val is not None else None
                lrr_entry = lrr_results.get(tid - 1)   # basis index beda (0-based) di compute_lrr_and_compare
                transects_out.append({
                    "id": tid,
                    "eprMPerYr": round(epr_val, 4) if epr_val is not None else None,
                    "nsmM": round(nsm_val, 3) if nsm_val is not None else None,
                    "sceM": round(float(sce_full[tid_str]), 3) if not np.isnan(sce_full[tid_str]) else None,
                    "lrrMPerYr": round(lrr_entry["rate_m_per_year"], 4) if lrr_entry else None,
                    "classification": classify(epr_val),
                    "mcUncertainty": None,   # model ini gak punya MC Dropout
                    "modelPositions": [],    # tidak dilacak per-tahun di pipeline ini
                })

            clf_pairs = {}
            for t in transects_out:
                clf_pairs[t["classification"]] = clf_pairs.get(t["classification"], 0) + 1
            classification_counts = [{"label": k, "count": v} for k, v in clf_pairs.items()]

            mean_epr = float(np.nanmean(row))
            mean_nsm = float(np.nanmean([t["nsmM"] for t in transects_out if t["nsmM"] is not None])) \
                       if any(t["nsmM"] is not None for t in transects_out) else None
            mean_sce = float(np.nanmean(sce_full))

            doc = {
                "_type": "shorelineForecast",
                "_id":   f"shorelineForecast-{aoi}-{int(year)}-train256",
                "aoi": aoi, "year": int(year), "season": s_used or "unknown",
                "runUtc": run_utc,
                "modelSource": f"convlstm_unet_256_{run_id}",
                "mcUncertaintyMean": None,
                "geojson": geojson_feat,
                "meanEprMPerYr": round(mean_epr, 4) if not np.isnan(mean_epr) else None,
                "meanNsmM": round(mean_nsm, 3) if mean_nsm is not None else None,
                "meanSceM": round(mean_sce, 3) if not np.isnan(mean_sce) else None,
                "meanLrrMPerYr": round(mean_lrr, 4) if mean_lrr is not None else None,
                "classificationCounts": classification_counts,
                "nTransects": len(transects_out),
                "transects": transects_out,
            }
            mutations.append({"createOrReplace": doc})

    if mutations:
        # push per batch 100 biar gak kena limit payload
        BATCH = 100
        for start in range(0, len(mutations), BATCH):
            push_batch(mutations[start:start + BATCH], label=f"batch {start // BATCH + 1}")
        print(f"\nDone. Pushed {len(mutations)} shorelineForecast documents "
              f"(1 per AOI per tahun, {baseline_year}-{TARGET_YEAR_CHECK}).")
    else:
        print("Tidak ada dokumen untuk di-push — cek all_metrics/lrr_vs_model/masks.")
```

---

## Cell 12 — BARU — Push ringkasan training ke Sanity (`shorelineModelRun`)

Reuse schema `shorelineModelRun` **yang sudah didefinisikan di `claude_result9.md`** —
tidak perlu bikin schema baru, cukup pastikan file `studio/schemaTypes/shorelineModelRun.js`
dari `claude_result9.md` sudah ditambahkan ke Sanity Studio.

```python
# ================================================================
# CELL 12 — Push ringkasan training (shorelineModelRun) ke Sanity
# Reuse schema dari claude_result9.md — field uncertainty-related
# tidak ada karena skema training ini gak punya kolom itu (bukan masalah).
# ================================================================

missing = [k for k in ["SANITY_PROJECT_ID", "SANITY_DATASET", "SANITY_API_TOKEN"]
           if not os.environ.get(k)]
if missing:
    print(f"SKIP: missing env vars {missing}")
else:
    run_doc = {
        "_type": "shorelineModelRun",
        "_id":   f"shorelineModelRun-train256-{run_id}",
        "runId": run_id,
        "runSuffix": "train256",
        "runUtc": datetime.now(tz=timezone.utc).isoformat(),
        "sentinelOnly": False,   # pipeline ini pakai Landsat+Sentinel gabungan
        "nChannels": 1,          # mask-only, gak ada channel statis
        "nSamplesTotal": len(meta),
        "nSamplesKept": len(meta),
        "nSamplesDroppedLandsat": 0,
        "trainUntil": TRAIN_UNTIL,
        "lookback": LOOKBACK,
        "epochs": 200,
        "baseCh": 16,
        "mcDropoutP": None,      # model ini gak punya MC Dropout
        "persistDiceTrain": None,   # persistence baseline cuma dihitung utk test di pipeline ini
        "persistDiceTest": round(dice_baseline, 4),
        "finalTrainDice": round(history["train_dice"][-1], 4),
        "finalTestDice": round(history["test_dice"][-1], 4),
        "bestTestDice": round(max(history["test_dice"]), 4),
        "epochTrainLoss": [round(float(v), 6) for v in history["train_loss"]],
        "epochTrainDice": [round(float(v), 6) for v in history["train_dice"]],
        "epochTestDice":  [round(float(v), 6) for v in history["test_dice"]],
        "wandbUrl": wandb.run.url if wandb.run else None,
    }
    resp = requests.post(url, json={"mutations": [{"createOrReplace": run_doc}]},
                          headers={"Authorization": f"Bearer {token}"}, timeout=60)
    resp.raise_for_status()
    print(f"Pushed shorelineModelRun -> {run_doc['_id']}")
```

---

## Ringkasan urutan final

`1 → 2 → 3 → 4 (training+W&B) → 5 → 6(opsional) → 7 → 8 → 9 → 10 (cek collapse dulu!) → 11 (Sanity DSAS) → 12 (Sanity training summary)`
