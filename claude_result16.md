# claude_result16.md — Guide: promote model/data abrasi ke production

Panduan lengkap dari kondisi **notebook kosong** (kernel Colab kemungkinan besar udah
disconnect lagi, semua variabel ilang) sampai data baru live di frontend WebGIS, plus
setup Prefect Cloud buat dashboard monitoring pipeline Actions.

Kalau notebook masih hidup (variabel `masks`/`model`/`run_id` dkk masih ada), skip
langsung ke **Bagian 2**.

---

## Bagian 1 — Resume notebook dari nol

Cek dulu isi `MODEL_DIR` di Drive buat nemuin `run_id` checkpoint yang mau dipromosikan:

```python
import os
MODEL_DIR = "/content/drive/MyDrive/Data_experiment_shoreline/models"
print(sorted(os.listdir(MODEL_DIR)))
```

Cari file `convlstm_unet_256_<run_id>.pth` yang paling relevan (biasanya paling baru).
Copy bagian `<run_id>`-nya.

Paste **Cell RESUME** dari `claude_result13.md` ke Colab, isi `RESUME_RUN_ID` dengan nilai
yang barusan ditemukan, lalu jalankan. Ini mount Drive, load `masks`/`AOI_CONFIG`, load
checkpoint model, dan rebuild semua fungsi helper — **tanpa training ulang**.

Setelah itu, jalankan berurutan (bebas urutan, dua-duanya independen, sama-sama harus
kelar sebelum Bagian 3):
1. **Cell 9** (`yearly_grid_plot_extended`, dari `claude_result10.md`) — ngisi `all_metrics`.
2. **Cell explainability** (`diagnose_rollout_collapse` + `compute_lrr_and_compare`, patched
   fix `TypeError: None - float` yang sempat dibahas) — ngisi `lrr_vs_model` dan
   `TARGET_YEAR_CHECK`.

---

## Bagian 2 — Export `model_meta.json`

Paste cell dari `claude_result15.md` (section "Cell baru — Export model_meta.json"),
jalankan. Hasilnya: `model_meta_<run_id>.json` di `MODEL_DIR`, isinya `IN_CH`/`LOOKBACK`/
`BASE_CH` model yang lagi dipakai — ini yang bakal dibaca `src/inference/predict.py` di
pipeline production (dia gak lagi hardcode bentuk model, baca dinamis dari file ini).

Download 2 file dari Drive ke laptop:
- `MODEL_DIR/convlstm_unet_256_<run_id>.pth` → rename jadi `checkpoint.pth`
- `MODEL_DIR/model_meta_<run_id>.json` → rename jadi `model_meta.json`

---

## Bagian 3 — Push data forecast ke Sanity (opsional)

Skip bagian ini kalau cuma mau promosikan MODEL-nya doang (checkpoint baru), tanpa
nge-update data abrasi yang udah tampil di WebGIS sekarang.

Kalau mau data-nya ikut ter-update:
1. **Cell 11** dari `claude_result14.md` (push `abrasionDataset` + `shorelineForecast`,
   transect per-tahun, geojson-nya udah di-fix).
2. **Cell 12** dari `claude_result10.md` (push `shorelineModelRun`, ringkasan training).

Pastiin `SANITY_API_TOKEN` yang dipakai itu token yang **udah di-rotate** (bukan yang
sempat ke-hardcode di notebook lama) — cek `.env` di Drive.

---

## Bagian 4 — Bikin GitHub Release (promosi model)

Di GitHub, buka repo `shoreline-kemujan` → tab **Releases** → **Draft a new release**.

- **Tag**: naikin dari yang terakhir dipakai di `.github/workflows/recalculate.yml`
  (`MODEL_RELEASE_TAG`). Kalau sekarang `model-v1`, tag baru `model-v2`, dst.
- **Title/notes**: bebas, tapi disaranin catet run_id asal + tanggal + ringkasan kenapa
  di-promote (mis. "checkpoint train_256 20260802_064800, ganti dari mask-only v1 gara2
  ...").
- **Attach files**: upload `checkpoint.pth` dan `model_meta.json` yang tadi di-download di
  Bagian 2.
- Klik **Publish release**.

Alternatif via `gh` CLI (jalanin sendiri di terminal lu, bukan gw yang jalanin):

```bash
gh release create model-v2 checkpoint.pth model_meta.json \
  --title "model-v2 (train_256, 2026-08-02)" \
  --notes "Checkpoint dari run_id 20260802_064800, mask-only in_ch=1."
```

---

## Bagian 5 — Update `MODEL_RELEASE_TAG`

Edit `.github/workflows/recalculate.yml`:

```yaml
env:
  ...
  MODEL_RELEASE_TAG: model-v2   # <-- ganti dari model-v1
```

Commit + push perubahan ini. Ini yang bikin pipeline Actions selanjutnya narik checkpoint
baru, bukan yang lama.

---

## Bagian 6 — Setup Prefect Cloud (buat dashboard monitoring pipeline)

1. Daftar akun gratis di **app.prefect.cloud**. Cek dulu di halaman pricing/signup apakah
   free tier sekarang minta kartu kredit atau tidak — cek langsung, gw gak punya akses
   internet buat mastiin kondisi terkininya.
2. Bikin **Workspace** baru (atau pakai default yang otomatis dibuatkan).
3. Ambil **API key**: Workspace settings → **API Keys** → **Create API Key** → copy
   nilainya (cuma keliatan sekali saat dibuat, langsung disimpen di tempat aman).
4. Ambil **API URL**: masih di Workspace settings, formatnya biasanya
   `https://api.prefect.cloud/api/accounts/<account_id>/workspaces/<workspace_id>`.
5. Simpen keduanya sebagai **GitHub Secrets** di repo `shoreline-kemujan`: **Settings** →
   **Secrets and variables** → **Actions** → **New repository secret**:
   - `PREFECT_API_KEY`
   - `PREFECT_API_URL`

   (`recalculate.yml` udah baca env ini otomatis — gak perlu edit workflow lagi.)

6. **Test lokal dulu** sebelum ngandelin Actions (opsional tapi disaranin banget, biar
   ketauan konfigurasinya bener sebelum nunggu cron/manual-dispatch):
   ```bash
   pip install prefect
   prefect cloud login -k <API_KEY_DARI_LANGKAH_3>
   python -m src.pipeline --run-dir data/interim/test_run
   ```
   Buka dashboard Prefect Cloud, cek flow run **"shoreline_pipeline"** beneran muncul di
   situ, dengan status tiap task (`fetch_composite`, `build_mask`, `track_a_predict`,
   `build_payload`, `push_to_sanity`) kelihatan jelas.

Kalau langkah ini di-skip sepenuhnya, pipeline **tetap jalan normal** — Prefect otomatis
jalan lokal/ephemeral tanpa lapor kemana-mana kalau `PREFECT_API_KEY`/`PREFECT_API_URL`
kosong. Ini bukan blocker, cuma hilang dashboard-nya.

---

## Bagian 7 — Trigger & verifikasi pipeline Actions

1. Buka repo `shoreline-kemujan` → tab **Actions** → workflow **"Recalculate shoreline
   predictions"** → **Run workflow** (manual dispatch, gak perlu nunggu cron bulanan).
2. Perhatiin log tiap step, terutama:
   - **"Download model checkpoint + meta from GitHub Release"** — harus berhasil nemuin
     Release dengan tag yang di-set di Bagian 5.
   - **"Run pipeline"** — ini yang jalanin `python -m src.pipeline`, semua 5 stage.
3. Kalau Bagian 6 udah disetup, cek run yang sama juga muncul di dashboard Prefect Cloud
   — dua tempat ini (Actions log dan Prefect dashboard) harus nunjukin run yang sama,
   cuma beda cara liatnya.
4. Cek dokumen `abrasionDataset` dan `shorelineForecast` di Sanity — via Vision tool
   (query `*[_id == "singleton-abrasionDataset"][0]` dan
   `*[_type == "shorelineForecast"] | order(year desc)[0...5]`) — pastiin
   `dataUpdatedAt`/`pipelineVersion` nunjukin timestamp run barusan, bukan yang lama.
5. Buka WebGIS frontend (`/webgis-kemujan` atau `/v2/webgis-kemujan`), cek overlay
   shoreline + marker AOI nampilin data baru.

---

## Bagian 8 — Checklist prasyarat yang masih outstanding

Sebelum nyoba Bagian 7 (trigger Actions), pastiin ini semua udah beres — kalau belum,
pipeline bakal gagal di tengah jalan:

- [ ] **`config/thresholds.json`** masih semua `null` — pipeline bakal gagal keras di
      step mask (`load_thresholds` sengaja hard-raise). Harus diisi dari
      `manifest['aoi'][name]['threshold_val']` hasil training/notebook run.
- [ ] **Token Sanity** yang sempat ke-hardcode di `train_256_final.ipynb` cell 14 —
      pastiin udah di-rotate (revoke token lama di Sanity Manage, bikin token write baru),
      dan `SANITY_API_TOKEN` di GitHub Secrets pakai yang baru.
- [ ] **`data/state/lrr_baseline.json`** — kalau belum pernah dibuat, field
      `lrrMPerYr`/`lrrProjectedM` di `shorelineForecast` bakal kosong (null, bukan salah,
      tapi juga gak keisi). Cara bikinnya ada di docstring `load_lrr_baseline()` di
      `src/output/geojson.py`.
- [ ] **GitHub Release** (Bagian 4) — belum ada release `model-v1` pun berarti pipeline
      Actions belum pernah bisa jalan sukses sama sekali sampai sekarang (step download
      checkpoint bakal gagal). Ini prasyarat paling dasar sebelum Bagian 7.
