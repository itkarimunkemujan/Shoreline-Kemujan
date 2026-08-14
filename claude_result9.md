# claude_result9.md — Rebuild 2aug: Sentinel-only + static multiband channels + transparency ke Sanity

Konteks keputusan (biar jelas kenapa cell-cell ini bentuknya begini):
- Diagnostic mask Landsat vs Sentinel (dari percakapan sebelumnya) mengonfirmasi mask
  Landsat memang berantakan secara visual, bukan cuma dugaan. Karena waktu mepet, **tidak
  reprocessing NDWI dari nol** — cukup **filter keluar sample yang tersentuh Landsat**
  dari `tensors.npz` yang sudah ada di Drive.
- `X_raw` di `tensors.npz` ternyata bukan cuma 1 channel (mask). Sesuai docstring
  `src/model/convlstm.py`: *"in_ch is dynamic: 1 (water mask) + N static per-pixel
  channels (elevation, slope, mangrove, landcover, SDB index)"*. Versi `claude_result3.md`
  Cell 3 SALAH karena motong ke `X_raw[:, :, 0:1, :, :]` — buang semua channel statis itu.
  Sekarang semua channel dipakai, jumlahnya dibaca otomatis dari data (`X_train.shape[2]`),
  tidak di-hardcode.
- Train loss sudah otomatis kelihatan di Cell 9 (plot 3-panel) — tidak berubah.
  Uncertainty (MC Dropout) & DSAS (EPR/LRR/klasifikasi) sudah ada di skema
  `shorelineForecast` yang di-push ke Sanity di Cell 14 lama — tidak berubah, cuma
  di-patch bbox-nya (dari `claude_result7.md`). **Tambahan baru**: Cell 14b — push 1
  dokumen ringkasan training (`shorelineModelRun`) ke Sanity biar riwayat training
  (loss/dice per epoch, berapa sample yang dibuang karena Landsat, dst) juga transparan
  di frontend, bukan cuma di plot notebook.

**Total blast radius** dari 2 keputusan ini (filter Sentinel-only + multichannel):
Cell 3, 4, 4b, 5 (instantiation line saja), 10, 11 berubah. Cell 1, 2, 6, 7, 8, 9, 11a,
11b, 11c, 11d, 12, 13, 14 **TIDAK BERUBAH SECARA LOGIKA** — instruksi copy di bawah.

---

## Cell 1 (2aug-v2) — SAMA PERSIS + 1 baris tambahan

Copy **Cell 1** utuh dari `claude_result3.md`. Tambahkan **satu baris** di blok
"Inference / rollout" (dekat `ROLLOUT_START`, dst):

```python
SENTINEL_START_YEAR = 2019   # Sentinel-2 mulai tersedia; < tahun ini pasti Landsat
```

---

## Cell 2 (2aug-v2) — SAMA PERSIS

Copy **Cell 2** utuh dari `claude_result3.md`, tidak ada perubahan.

---

## Cell 3 (2aug-v2) — BARU — Filter Sentinel-only + pertahankan semua channel

Perubahan dari `claude_result3.md` Cell 3:
- **Tidak lagi** slice ke `X_raw[:, :, 0:1, :, :]` — semua channel (mask + statis)
  dipertahankan apa adanya.
- **Filter sample**: sebuah sample dibuang kalau ADA frame (input manapun atau target)
  yang tersentuh Landsat — dicek dari 2 kemungkinan skema meta (biar tahan terhadap
  perbedaan skema antar builder):
  - skema lama (`{'input': [(yr,s),...], 'target': (yr,s)}`) — cek semua pasangan.
  - skema baru (`{'target_year', 'target_season'}`) — cek target saja (fallback kalau
    field `input`/`target` tidak ada).
  - Kriteria "Landsat": `season in ("L1","L2")` ATAU `year < SENTINEL_START_YEAR`.
- Filtering dilakukan **sebelum** split train/test, supaya train & test dua-duanya bersih.
- Simpan statistik filter (`N_TOTAL_SAMPLES`, `N_KEPT_SAMPLES`, `N_DROPPED_LANDSAT`) ke
  variabel global — dipakai lagi di Cell 14b buat transparansi ke Sanity.

```python
# ================================================================
# CELL 3 — Filter Sentinel-only + pertahankan semua channel (mask + statis)
# ================================================================

def _sample_seasons(m: dict):
    """Semua pasangan (year, season) yang disentuh 1 sample, tahan-skema."""
    pairs = []
    if "input" in m and "target" in m:
        pairs.extend(m["input"])
        pairs.append(m["target"])
    else:
        yr, seas = m.get("target_year"), m.get("target_season")
        if yr is not None and seas is not None:
            pairs.append((yr, seas))
    return pairs


def _is_sentinel_only(m: dict) -> bool:
    pairs = _sample_seasons(m)
    if not pairs:
        log.warning("Sample tanpa info season yang bisa dibaca — dianggap TIDAK aman, dibuang")
        return False
    return all(
        (isinstance(s, str) and s in ("S1", "S2", "S3")) and (yr >= SENTINEL_START_YEAR)
        for yr, s in pairs
    )


N_TOTAL_SAMPLES = len(meta)
keep_idx        = [i for i, m in enumerate(meta) if _is_sentinel_only(m)]
N_KEPT_SAMPLES  = len(keep_idx)
N_DROPPED_LANDSAT = N_TOTAL_SAMPLES - N_KEPT_SAMPLES

log.info("Filter Sentinel-only: keep %d / %d sample (buang %d sample tersentuh Landsat)",
         N_KEPT_SAMPLES, N_TOTAL_SAMPLES, N_DROPPED_LANDSAT)
print(f"Sample total   : {N_TOTAL_SAMPLES}")
print(f"Sample dipakai : {N_KEPT_SAMPLES} (Sentinel-only)")
print(f"Sample dibuang : {N_DROPPED_LANDSAT} (tersentuh Landsat)")

X_full = X_raw[keep_idx]     # (N_kept, LOOKBACK, C, H, W) — C = 1 mask + N channel statis
meta   = [meta[i] for i in keep_idx]

if y_raw.ndim == 5:
    y_mask = y_raw[keep_idx, 0, :, :, :]
else:
    y_mask = y_raw[keep_idx]

log.info("Setelah filter — X_full: %s  y_mask: %s", X_full.shape, y_mask.shape)
N_CHANNELS = X_full.shape[2]
print(f"Jumlah channel input (mask + statis): {N_CHANNELS}")

# Temporal split
tr_idx = [i for i, m in enumerate(meta) if m.get("target_year", 9999) <= TRAIN_UNTIL]
te_idx = [i for i, m in enumerate(meta) if m.get("target_year", 9999) >  TRAIN_UNTIL]

X_train = torch.from_numpy(X_full[tr_idx]).float().to(device)
y_train = torch.from_numpy(y_mask[tr_idx]).float().to(device)
X_test  = torch.from_numpy(X_full[te_idx]).float().to(device)
y_test  = torch.from_numpy(y_mask[te_idx]).float().to(device)

meta_train = [meta[i] for i in tr_idx]
meta_test  = [meta[i] for i in te_idx]

log.info("Train: %d samples (target <= %d)", len(tr_idx), TRAIN_UNTIL)
log.info("Test:  %d samples (target >  %d)", len(te_idx), TRAIN_UNTIL)
log.info("X_train: %s  X_test: %s", tuple(X_train.shape), tuple(X_test.shape))
```

---

## Cell 4 (2aug-v2) — DIUBAH SEDIKIT — Persistence baseline (ambil channel mask saja)

Beda dari `claude_result3.md` Cell 4: `X_test[:, -1]` sekarang berdimensi
`(N, N_CHANNELS, H, W)` (bukan `(N, 1, H, W)` lagi), jadi harus diambil channel 0
(mask) saja supaya bisa dibandingkan ke `y_test` yang selalu 1 channel.

```python
# ================================================================
# CELL 4 — Persistence baseline Dice (channel 0 = mask saja)
# ================================================================

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred   = pred.flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    return (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)

with torch.no_grad():
    persist_pred = X_test[:, -1, 0:1]                        # last input frame, channel mask saja
    persist_dice = dice_score(persist_pred, y_test).mean().item()

log.info("Persistence baseline Dice (test): %.4f", persist_dice)
print(f"Persistence baseline Dice: {persist_dice:.4f}  ← model must beat this")
```

---

## Cell 4b (2aug-v2) — DIUBAH SEDIKIT — Persistence baseline train (dari `claude_result4.md`)

Sama seperti Cell 4b di `claude_result4.md`, cuma tambah `0:1` di index channel:

```python
# ================================================================
# CELL 4b — Persistence baseline Dice (TRAIN set, channel 0 = mask saja)
# ================================================================

with torch.no_grad():
    persist_pred_train = X_train[:, -1, 0:1]
    persist_dice_train  = dice_score(persist_pred_train, y_train).mean().item()

log.info("Persistence baseline Dice (train): %.4f", persist_dice_train)
print(f"Persistence baseline Dice — train: {persist_dice_train:.4f}")
print(f"Persistence baseline Dice — test:  {persist_dice:.4f}")
print(f"Gap (test - train): {persist_dice - persist_dice_train:.4f}")
```

---

## Cell 5 (2aug-v2) — SAMA PERSIS kecuali baris instantiation terakhir

Copy definisi class `ConvLSTMCell`, `ConvLSTMUNet`, `enable_mc_dropout`,
`mc_dropout_predict` **apa adanya** dari `claude_result3.md` Cell 5 (class-nya sudah
channel-agnostic, `in_ch` cuma parameter). Ganti **3 baris terakhir** saja:

```python
model = ConvLSTMUNet(in_ch=N_CHANNELS, base_ch=BASE_CH, mc_dropout_p=MC_DROPOUT_P).to(device)
n_params = sum(p.numel() for p in model.parameters())
log.info("Model: ConvLSTMUNet(in_ch=%d, base_ch=%d, mc_dropout_p=%.2f) — %s params",
         N_CHANNELS, BASE_CH, MC_DROPOUT_P, f"{n_params:,}")
print(f"in_ch (mask + statis): {N_CHANNELS} | Parameters: {n_params:,}")
```

---

## Cell 6 (2aug-v2) — SAMA PERSIS

Copy **Cell 6** utuh dari `claude_result3.md` — `DiceBCELoss`, optimizer, tidak ada
ketergantungan channel.

---

## Cell 7 (2aug-v2) — SAMA PERSIS + beberapa key tambahan di `config=` (opsional, transparansi)

Copy **Cell 7** utuh dari `claude_result3.md`. Kalau mau W&B run-nya juga transparan soal
keputusan filtering, tambahkan key ini ke dict `config={...}` di `wandb.init(...)`:

```python
        "sentinel_only": True,
        "n_channels": N_CHANNELS,
        "n_samples_total": N_TOTAL_SAMPLES,
        "n_samples_dropped_landsat": N_DROPPED_LANDSAT,
```

---

## Cell 8 (2aug-v2) — SAMA PERSIS

Copy **Cell 8** utuh dari `claude_result3.md` — `augment_batch` (rotate-90 + mask noise)
dan `train_loop` sudah channel-agnostic (`torch.rot90` beroperasi di dim H/W, tidak peduli
jumlah channel). **Catatan**: kalau salah satu channel statis itu punya makna arah
(mis. "aspect"/arah hadap lereng), rotasi 90° bisa bikin channel itu jadi tidak konsisten
secara fisik dengan mask yang ikut dirotasi bersamanya — tapi berdasarkan daftar di
docstring `convlstm.py` (elevation, slope, mangrove, landcover, SDB index) semuanya
skalar/kelas, bukan arah, jadi aman untuk sekarang.

---

## Cell 9 (2aug-v2) — SAMA PERSIS

Copy **Cell 9** utuh dari `claude_result3.md`. Ini yang menghasilkan **plot train loss +
train/test Dice + train-test gap** yang lu minta transparan — sudah otomatis kebentuk di
sini, tidak perlu cell tambahan buat itu.

---

## Cell 10 (2aug-v2) — BARU — Load checkpoint + rollout seed dengan channel statis ikut terbawa

Beda dari `claude_result3.md` Cell 10: dulu `aoi_frames`/`rollout_seeds` dibangun cuma
dari `all_y` (mask 1-channel) — itu benar waktu `in_ch=1`. Sekarang model butuh
`in_ch=N_CHANNELS`, jadi tiap frame historis yang dipakai sebagai "seed" rollout harus
punya channel statis juga. Karena channel statis itu konstan per-AOI (di-broadcast sama
di semua timestep & semua sample AOI yang sama), cukup ambil sekali dari `all_X` per AOI
(`aoi_static[aoi]`), lalu digabung ke tiap frame mask historis.

```python
# ================================================================
# CELL 10 — Load best checkpoint + build rollout seeds per AOI
# (channel statis diikutkan — lihat Cell 3: in_ch sekarang > 1)
# ================================================================

best_path = os.path.join(MODEL_DIR, f"best_{RUN_SUFFIX}.pth")
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()
log.info("Loaded best checkpoint: %s", best_path)

all_meta = meta_train + meta_test
all_X    = torch.cat([X_train, X_test], dim=0)   # (N, LOOKBACK, N_CHANNELS, H, W)
all_y    = torch.cat([y_train, y_test], dim=0)   # (N, 1, H, W)

# Channel statis konstan per-AOI: ambil sekali dari sample pertama tiap AOI
# (timestep manapun sama saja, sudah di-broadcast waktu tensors.npz dibuat)
aoi_static = {}
for i, m in enumerate(all_meta):
    aoi = m.get("aoi", "unknown")
    if aoi not in aoi_static:
        aoi_static[aoi] = all_X[i, 0, 1:, :, :].clone()   # (N_CHANNELS-1, H, W)

from collections import defaultdict
aoi_frames = defaultdict(list)   # aoi -> [(t_float, mask_full)]  mask_full: (N_CHANNELS,H,W)

for i, m in enumerate(all_meta):
    aoi    = m.get("aoi", "unknown")
    t_yr   = m.get("target_year", 0)
    t_seas = m.get("target_season", "S1")
    t_mon  = SEASON_MONTHS.get(t_seas, 6)
    t_float = t_yr + t_mon / 12.0
    mask_only = all_y[i]                              # (1, H, W)
    mask_full = torch.cat([mask_only, aoi_static[aoi]], dim=0)  # (N_CHANNELS, H, W)
    aoi_frames[aoi].append((t_float, mask_full))

for aoi in aoi_frames:
    aoi_frames[aoi].sort(key=lambda x: x[0])

rollout_seeds = {}
for aoi, frames in aoi_frames.items():
    if len(frames) < LOOKBACK:
        log.warning("AOI %s has only %d frames — padding with last frame", aoi, len(frames))
        while len(frames) < LOOKBACK:
            frames.insert(0, frames[0])
    seed_masks = torch.stack([f[1] for f in frames[-LOOKBACK:]], dim=0)  # (LOOKBACK, N_CHANNELS, H, W)
    rollout_seeds[aoi] = seed_masks.to(device)
    log.info("AOI %s seed shape: %s  (last GT: t=%.2f)", aoi, tuple(seed_masks.shape), frames[-1][0])

print(f"Rollout seeds ready for {len(rollout_seeds)} AOIs: {list(rollout_seeds.keys())}")
```

---

## Cell 11 (2aug-v2) — BARU — Rollout, window advance ikut bawa channel statis

Gabungan dari 2 hal: fix `torch.cat` dari `claude_result4.md` (jangan index `[0]` di
`pred_mask`) **plus** rekonstruksi frame full-channel tiap langkah (model cuma
mengeluarkan prediksi mask 1-channel, jadi channel statis yang konstan per-AOI harus
ditempel lagi sebelum jadi "frame" baru di window).

```python
# ================================================================
# CELL 11 — Autoregressive rollout 2025–2035 (3 seasons/year)
# window sekarang (LOOKBACK, N_CHANNELS, H, W); model output tetap 1-channel
# mask, direkonstruksi jadi full-channel via aoi_static[aoi] sebelum
# dipakai lagi sebagai input langkah berikutnya.
# ================================================================
from skimage import measure
from scipy.interpolate import splprep, splev

def extract_contour(mask_np: np.ndarray, smooth: bool = True):
    contours = measure.find_contours(mask_np, level=0.5)
    if not contours:
        return None, None
    contour = max(contours, key=len)
    if smooth and len(contour) > 5:
        try:
            tck, _ = splprep([contour[:, 1], contour[:, 0]], s=len(contour), k=3)
            xi, yi = splev(np.linspace(0, 1, 200), tck)
            return yi, xi
        except Exception:
            pass
    return contour[:, 0], contour[:, 1]


rollout_steps = []
for yr in range(ROLLOUT_START, ROLLOUT_END + 1):
    for seas in ROLLOUT_SEASONS:
        mon = SEASON_MONTHS[seas]
        rollout_steps.append({"year": yr, "season": seas, "t": yr + mon / 12.0})

log.info("Total rollout steps: %d  (%d–%d, %d seasons/yr)",
         len(rollout_steps), ROLLOUT_START, ROLLOUT_END, len(ROLLOUT_SEASONS))

rollout_results = {}

for aoi, seed in rollout_seeds.items():
    window       = seed.clone()               # (LOOKBACK, N_CHANNELS, H, W)
    static_stack = aoi_static[aoi]             # (N_CHANNELS-1, H, W), konstan
    aoi_results  = []

    for step_info in rollout_steps:
        x_in = window.unsqueeze(0)             # (1, LOOKBACK, N_CHANNELS, H, W)
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

        # Rekonstruksi frame full-channel: mask baru (channel 0) + statis (konstan)
        new_frame = torch.cat([pred_mask[0], static_stack], dim=0).unsqueeze(0)  # (1, N_CHANNELS, H, W)
        window = torch.cat([window[1:], new_frame], dim=0)

    rollout_results[aoi] = aoi_results
    log.info("AOI %s rollout done — %d steps, avg unc=%.4f",
             aoi, len(aoi_results), np.mean([r["unc_mean"] for r in aoi_results]))

print(f"Rollout complete for {len(rollout_results)} AOIs.")
```

---

## Cell 11a (2aug-v2) — SAMA PERSIS

Copy **Cell 11a** utuh dari `claude_result6.md` (sudah termasuk fix `get_bbox`). Tidak
ada ketergantungan channel — semua di sini beroperasi di `all_y` (mask, selalu 1 channel)
dan `rollout_results` (mask/contour output model, juga selalu 1 channel).

Sambil di situ, hapus dead code `def patch_bounds(lon, lat): ...` yang masih nyangkut
(pakai `PATCH_SIZE` yang tidak terdefinisi) — tidak bikin error tapi membingungkan kalau
dibaca lagi nanti.

## Cell 11b, 11c, 11d (2aug-v2) — SAMA PERSIS

Copy **Cell 11b**, **Cell 11c**, **Cell 11d** utuh dari `claude_result5.md`. Tidak ada
ketergantungan channel sama sekali di sini.

---

## Cell 12, 13, 14 (2aug-v2) — SAMA PERSIS + patch bbox

Copy **Cell 12** (EPR/LRR/Transect), **Cell 13** (GeoJSON export), **Cell 14** (Sanity
bootstrap push) utuh dari `claude_result3.md`, lalu terapkan **ketiga patch** di
`claude_result7.md` (ganti bbox lookup manual jadi `get_bbox(aoi)` di ketiga cell).

Ini sudah otomatis memenuhi "uncertainty & DSAS keliatan di Sanity" — skema
`shorelineForecast` di Cell 14 sudah punya `mcUncertaintyMean`, `meanEprMPerYr`,
`meanNsmM`, `meanSceM`, `meanLrrMPerYr`, `classificationCounts`, dan array `transects[]`
lengkap per-transect (`eprMPerYr`, `lrrMPerYr`, `classification`, `mcUncertainty`) —
tidak ada yang perlu ditambah di skema Sanity-nya.

---

## Cell 14b (2aug-v2) — BARU — Push ringkasan training (`shorelineModelRun`) ke Sanity

Supaya transparansi training (loss/dice per epoch, berapa sample dibuang karena Landsat,
berapa channel input dipakai) juga bisa dilihat dari frontend/Sanity Studio, bukan cuma
dari plot di notebook. Satu dokumen per run (pakai `run_id` sebagai bagian dari `_id`,
jadi tidak akan ke-overwrite run lain).

```python
# ================================================================
# CELL 14b — Push ringkasan training (shorelineModelRun) ke Sanity
# NOTE: pakai kredensial Sanity yang sama seperti Cell 14 (SANITY_PROJECT_ID,
#       SANITY_DATASET, SANITY_API_TOKEN) — jalankan Cell 14 dulu minimal
#       sekali di sesi ini supaya `push_batch`/`url`/`token` sudah terdefinisi,
#       atau paste ulang blok load-kredensial di awal Cell 14 kalau cell ini
#       dijalankan terpisah.
# ================================================================

missing = [k for k in ["SANITY_PROJECT_ID", "SANITY_DATASET", "SANITY_API_TOKEN"]
           if not os.environ.get(k)]
if missing:
    print(f"SKIP: missing env vars {missing}")
else:
    run_doc = {
        "_type":  "shorelineModelRun",
        "_id":    f"shorelineModelRun-{RUN_SUFFIX}-{run_id}",
        "runId":  run_id,
        "runSuffix": RUN_SUFFIX,
        "runUtc": datetime.now(tz=timezone.utc).isoformat(),
        "sentinelOnly": True,
        "nChannels": N_CHANNELS,
        "nSamplesTotal": N_TOTAL_SAMPLES,
        "nSamplesKept": N_KEPT_SAMPLES,
        "nSamplesDroppedLandsat": N_DROPPED_LANDSAT,
        "trainUntil": TRAIN_UNTIL,
        "lookback": LOOKBACK,
        "epochs": EPOCHS,
        "baseCh": BASE_CH,
        "mcDropoutP": MC_DROPOUT_P,
        "persistDiceTrain": round(persist_dice_train, 4),
        "persistDiceTest": round(persist_dice, 4),
        "finalTrainDice": round(history["train_dice"][-1], 4),
        "finalTestDice": round(history["test_dice"][-1], 4),
        "bestTestDice": round(max(history["test_dice"]), 4),
        # array penuh per-epoch supaya frontend bisa render chart training
        # yang sama persis dengan plot di notebook (transparansi penuh)
        "epochTrainLoss": [round(float(v), 6) for v in history["train_loss"]],
        "epochTrainDice": [round(float(v), 6) for v in history["train_dice"]],
        "epochTestDice":  [round(float(v), 6) for v in history["test_dice"]],
        "wandbUrl": wandb.run.url if wandb.run else None,
    }

    resp = requests.post(
        url,
        json={"mutations": [{"createOrReplace": run_doc}]},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    log.info("shorelineModelRun pushed: %s", run_doc["_id"])
    print(f"Pushed shorelineModelRun -> {run_doc['_id']}")
```

### Tambahan skema Sanity untuk `shorelineModelRun`

File baru `studio/schemaTypes/shorelineModelRun.js`:

```javascript
import { defineField, defineType } from 'sanity'

export default defineType({
  name: 'shorelineModelRun',
  title: 'Shoreline Model Run',
  type: 'document',
  fields: [
    defineField({ name: 'runId', type: 'string' }),
    defineField({ name: 'runSuffix', type: 'string' }),
    defineField({ name: 'runUtc', type: 'string' }),
    defineField({ name: 'sentinelOnly', type: 'boolean' }),
    defineField({ name: 'nChannels', type: 'number' }),
    defineField({ name: 'nSamplesTotal', type: 'number' }),
    defineField({ name: 'nSamplesKept', type: 'number' }),
    defineField({ name: 'nSamplesDroppedLandsat', type: 'number' }),
    defineField({ name: 'trainUntil', type: 'number' }),
    defineField({ name: 'lookback', type: 'number' }),
    defineField({ name: 'epochs', type: 'number' }),
    defineField({ name: 'baseCh', type: 'number' }),
    defineField({ name: 'mcDropoutP', type: 'number' }),
    defineField({ name: 'persistDiceTrain', type: 'number' }),
    defineField({ name: 'persistDiceTest', type: 'number' }),
    defineField({ name: 'finalTrainDice', type: 'number' }),
    defineField({ name: 'finalTestDice', type: 'number' }),
    defineField({ name: 'bestTestDice', type: 'number' }),
    defineField({ name: 'epochTrainLoss', type: 'array', of: [{ type: 'number' }] }),
    defineField({ name: 'epochTrainDice', type: 'array', of: [{ type: 'number' }] }),
    defineField({ name: 'epochTestDice',  type: 'array', of: [{ type: 'number' }] }),
    defineField({ name: 'wandbUrl', type: 'string' }),
  ],
  preview: {
    select: { title: 'runId', subtitle: 'runUtc' },
  },
})
```

Daftarkan di `studio/schemaTypes/index.js` (sama pola seperti `shorelineForecast` di
`claude_result3.md`):

```javascript
import shorelineModelRun from './shorelineModelRun'
// ...
export const schemaTypes = [...existingTypes, shorelineForecast, shorelineModelRun]
```

---

## Ringkasan urutan final notebook

`1 → 2 → 3 → 4 → 4b → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 11a → 11b → 11c → 11d → 12 → 13 → 14 → 14b`
