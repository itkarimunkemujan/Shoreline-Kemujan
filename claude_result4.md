# claude_result4.md — Fix rollout bug + diagnostic cell (2aug)

Dua perubahan untuk notebook 2aug:

1. **Cell 11 (REPLACE)** — fix `torch.cat` dimension mismatch yang bikin error saat rollout.
2. **Cell 4b (INSERT setelah Cell 4)** — hitung persistence Dice di train set juga, untuk
   konfirmasi apakah gap train/test Dice disebabkan oleh campuran Landsat 30m + Sentinel 10m
   di training set.

---

## Cell 11 (2aug) — REPLACE — Autoregressive rollout 2025–2035

Perubahan:
- Baris `window = torch.cat([window[1:], pred_mask[0]], dim=0)` diganti jadi
  `torch.cat([window[1:], pred_mask], dim=0)`.
- Root cause: `window` shape `(LOOKBACK, 1, H, W)` — 4 dims. `pred_mask` shape `(1, 1, H, W)`
  sudah pas jadi "1 timestep baru" di axis waktu (4 dims). Indexing `pred_mask[0]` membuang
  batch-dim sehingga jadi `(1, H, W)` — 3 dims — dan `torch.cat` gagal karena jumlah dimensi
  tensor yang digabung harus sama.

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
        # NOTE: pred_mask is already (1, 1, H, W) == (new_T=1, C=1, H, W),
        # matching window[1:] which is (LOOKBACK-1, 1, H, W). Do NOT index [0] here —
        # that drops the batch dim and leaves a 3D tensor that torch.cat can't
        # concatenate against the 4D window.
        window = torch.cat([window[1:], pred_mask], dim=0)

    rollout_results[aoi] = aoi_results
    log.info("AOI %s rollout done — %d steps, avg unc=%.4f",
             aoi, len(aoi_results), np.mean([r["unc_mean"] for r in aoi_results]))

print(f"Rollout complete for {len(rollout_results)} AOIs.")
```

Catatan tambahan: baris `logit = model(x_in)` di versi lama dihapus karena hasilnya tidak
pernah dipakai — `mc_dropout_predict` sudah menjalankan forward pass sendiri 20x secara
independen, jadi baris itu cuma pemborosan satu forward pass per step tanpa manfaat.

---

## Cell 4b (2aug) — INSERT setelah Cell 4 — Persistence baseline di TRAIN set juga

Perubahan:
- Cell 4 yang lama cuma hitung persistence Dice untuk test set. Untuk konfirmasi apakah
  gap besar antara train Dice (~0.6) dan test Dice (~0.96) itu disebabkan oleh data
  training yang secara struktural lebih "berisik" (campuran Landsat 30m lama + Sentinel
  10m baru — lihat `notebooks/all_code2.py:414`, field `"sensor": "Landsat (union
  composite, 30m)"`), cell ini menghitung metrik yang sama persis tapi di train set.
- Kalau `persist_dice_train` juga rendah (mendekati 0.6, jauh di bawah `persist_dice`
  test 0.9472), itu bukti kuat bahwa train set memang punya perubahan/noise antar-frame
  yang jauh lebih besar secara natural — bukan model yang gagal fit.

```python
# ================================================================
# CELL 4b — Persistence baseline Dice (TRAIN set, untuk banding cek)
# ================================================================

with torch.no_grad():
    persist_pred_train = X_train[:, -1]                       # last input frame
    persist_dice_train  = dice_score(persist_pred_train, y_train).mean().item()

log.info("Persistence baseline Dice (train): %.4f", persist_dice_train)
print(f"Persistence baseline Dice — train: {persist_dice_train:.4f}")
print(f"Persistence baseline Dice — test:  {persist_dice:.4f}")
print(f"Gap (test - train): {persist_dice - persist_dice_train:.4f}")

if persist_dice - persist_dice_train > 0.15:
    print("⚠ Train set punya perubahan antar-frame jauh lebih besar dari test set — "
          "kemungkinan besar karena campuran sensor (Landsat 30m vs Sentinel 10m) atau "
          "label noise di tahun-tahun lama, BUKAN karena model gagal belajar.")
```
