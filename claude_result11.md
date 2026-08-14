# claude_result11.md — Cell 11 REVISI FINAL: push ke `abrasionDataset` (overlay) + `shorelineForecast` (drawer per-AOI)

## Kenapa direvisi lagi

Ternyata frontend `tourism-kemujan` udah punya kontrak integrasi sendiri yang **sudah
ter-wired ke overlay peta** (`studio/schemaTypes/singletons/abrasionDataset.ts` — singleton,
3 file GeoJSON ter-upload + metrics agregat), beda total dari `shorelineForecast` yang
dipush di `claude_result10.md`. Supaya overlay-nya beneran muncul di WebGIS **tanpa nulis
kode frontend baru**, push utama garis pantai harus ke `abrasionDataset`.

`shorelineForecast` **tetap dipakai** — bukan mubazir — buat fitur baru yang diminta:
marker per-AOI di peta + drawer yang nampilin heatmap EPR (transect × tahun) dan rata-rata
per region saat marker-nya diklik. Itu butuh data granular per-AOI-per-tahun yang gak ada
di `abrasionDataset` (yang cuma nyimpen 1 angka rata-rata gabungan semua AOI).

**Field baru** ditambahkan ke schema `shorelineForecast.ts`: `aoiLon`/`aoiLat` (titik pusat
AOI, buat naro marker — beda dari `originLon`/`originLat` per-transect yang udah ada).
Sudah di-patch langsung di `tourism-kemujan/studio/schemaTypes/documents/shorelineForecast.ts`.

**Ganti Cell 11 di `claude_result10.md` dengan cell ini.** Cell 12 (`shorelineModelRun`,
ringkasan training) **tidak berubah**, tetap pakai yang di `claude_result10.md`.

---

## Cell 11 (REVISI FINAL) — Push `abrasionDataset` (overlay) + `shorelineForecast` (drawer per-AOI)

```python
# ================================================================
# CELL 11 — Push ke Sanity: abrasionDataset (singleton, overlay utama)
# + shorelineForecast (per-AOI-per-tahun, buat marker+drawer heatmap)
# ================================================================
import requests
import json as _json
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
    mutate_url = f"https://{project_id}.api.sanity.io/v{SANITY_API_VERSION}/data/mutate/{dataset}"
    upload_url = f"https://{project_id}.api.sanity.io/v{SANITY_API_VERSION}/assets/files/{dataset}"

    # Harus SAMA PERSIS dengan studio/schemaTypes/_singletonIds.ts
    ABRASION_SINGLETON_ID = "singleton-abrasionDataset"

    def push_batch(mutations, label=""):
        resp = requests.post(mutate_url, json={"mutations": mutations},
                              headers={"Authorization": f"Bearer {token}"}, timeout=60)
        resp.raise_for_status()
        print(f"Sanity push OK: {len(mutations)} docs  {label}")
        return resp.json()

    def upload_geojson_asset(feature_collection, filename):
        """Upload 1 FeatureCollection sebagai Sanity file asset. Return asset _id."""
        content = _json.dumps(feature_collection).encode("utf-8")
        resp = requests.post(
            upload_url, params={"filename": filename},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            data=content, timeout=120,
        )
        resp.raise_for_status()
        asset_id = resp.json()["document"]["_id"]
        print(f"Uploaded {filename} ({len(content)/1024:.1f} KB) -> {asset_id}")
        return asset_id

    def classify(epr):
        if epr is None or np.isnan(epr):
            return "Tidak valid"
        for lo, hi, label in CHANGE_BINS:
            if lo <= epr < hi:
                return label
        return "Tidak valid"

    CHANGE_BINS = [
        (-float("inf"), -2.0, "Erosi Parah"), (-2.0, -0.5, "Erosi"),
        (-0.5, 0.5, "Stabil"), (0.5, 2.0, "Akresi"), (2.0, float("inf"), "Akresi Kuat"),
    ]

    def pick_repr(season_dict, priority=('S2', 'S3', 'S1')):
        for s in priority:
            if s in season_dict and season_dict[s] is not None:
                return season_dict[s], s
        for s, v in season_dict.items():
            if v is not None:
                return v, s
        return None, None

    run_utc = datetime.now(tz=timezone.utc).isoformat()

    current_features   = []   # shorelineCurrent.geojson  — semua AOI, tahun real
    predicted_features  = []   # shorelinePredicted.geojson — semua AOI, tahun rollout
    transect_features   = []   # transects.geojson — garis transect + EPR/NSM/LRR final
    forecast_mutations  = []   # shorelineForecast per-AOI-per-tahun (buat drawer)

    per_aoi_mean_epr, per_aoi_mean_nsm, per_aoi_mean_lrr = [], [], []

    for aoi, df_metrics in all_metrics.items():
        if df_metrics is None or df_metrics.empty:
            continue
        if aoi not in masks or aoi not in AOI_CONFIG:
            continue

        lon_c, lat_c = AOI_CONFIG[aoi]['coord']
        baseline_year = df_metrics.index[0]
        last_year     = df_metrics.index[-1]

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
        lrr_results = lrr_data[1] if lrr_data else {}
        lrr_rates = [v["rate_m_per_year"] for v in lrr_results.values() if v]
        mean_lrr_aoi = float(np.mean(lrr_rates)) if lrr_rates else None

        sce_full = (df_metrics.max() - df_metrics.min())

        # ---------- (A) kontur per tahun -> current/predicted GeoJSON + shorelineForecast ----------
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
            feat = {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "aoi": aoi, "year": int(year), "season": s_used,
                    "source": "real" if is_real else "rollout",
                },
            }
            (current_features if is_real else predicted_features).append(feat)

            row = df_metrics.loc[year]
            dt  = max(year - baseline_year, 0)
            transects_out = []
            for tid_str in df_metrics.columns:
                tid = int(tid_str.replace("T", ""))
                epr_val = float(row[tid_str]) if not np.isnan(row[tid_str]) else None
                nsm_val = epr_val * dt if epr_val is not None else None
                lrr_entry = lrr_results.get(tid - 1)
                transects_out.append({
                    "id": tid,
                    "eprMPerYr": round(epr_val, 4) if epr_val is not None else None,
                    "nsmM": round(nsm_val, 3) if nsm_val is not None else None,
                    "sceM": round(float(sce_full[tid_str]), 3) if not np.isnan(sce_full[tid_str]) else None,
                    "lrrMPerYr": round(lrr_entry["rate_m_per_year"], 4) if lrr_entry else None,
                    "classification": classify(epr_val),
                    "mcUncertainty": None,
                    "modelPositions": [],
                })

            clf_pairs = {}
            for t in transects_out:
                clf_pairs[t["classification"]] = clf_pairs.get(t["classification"], 0) + 1
            classification_counts = [{"label": k, "count": v} for k, v in clf_pairs.items()]

            mean_epr_yr = float(np.nanmean(row))
            nsm_vals = [t["nsmM"] for t in transects_out if t["nsmM"] is not None]
            mean_nsm_yr = float(np.nanmean(nsm_vals)) if nsm_vals else None
            mean_sce_yr = float(np.nanmean(sce_full))

            forecast_mutations.append({"createOrReplace": {
                "_type": "shorelineForecast",
                "_id":   f"shorelineForecast-{aoi}-{int(year)}-train256",
                "aoi": aoi, "aoiLon": lon_c, "aoiLat": lat_c,
                "year": int(year), "season": s_used or "unknown",
                "runUtc": run_utc,
                "modelSource": f"convlstm_unet_256_{run_id}",
                "mcUncertaintyMean": None,
                "geojson": feat,
                "meanEprMPerYr": round(mean_epr_yr, 4) if not np.isnan(mean_epr_yr) else None,
                "meanNsmM": round(mean_nsm_yr, 3) if mean_nsm_yr is not None else None,
                "meanSceM": round(mean_sce_yr, 3) if not np.isnan(mean_sce_yr) else None,
                "meanLrrMPerYr": round(mean_lrr_aoi, 4) if mean_lrr_aoi is not None else None,
                "classificationCounts": classification_counts,
                "nTransects": len(transects_out),
                "transects": transects_out,
            }})

            # simpan metrik tahun terakhir buat rata-rata agregat abrasionDataset
            if year == last_year:
                per_aoi_mean_epr.append(mean_epr_yr)
                if mean_nsm_yr is not None:
                    per_aoi_mean_nsm.append(mean_nsm_yr)
                if mean_lrr_aoi is not None:
                    per_aoi_mean_lrr.append(mean_lrr_aoi)

        # ---------- (B) garis transect statis (posisi tetap) + EPR/klasifikasi tahun terakhir ----------
        baseline_mask, baseline_season = pick_repr(gt_by_year.get(baseline_year, {}))
        baseline_contour = extract_main_contour(baseline_mask, use_spline=True, spline_smoothing=2.0) \
                            if baseline_mask is not None else None
        if baseline_contour is not None:
            transects_px = make_transects(baseline_contour, n_transects=len(df_metrics.columns),
                                           length_px=8)
            last_row = df_metrics.iloc[-1]
            for (p1, p2), tid_str in zip(transects_px, df_metrics.columns):
                tid = int(tid_str.replace("T", ""))
                epr_val = float(last_row[tid_str]) if not np.isnan(last_row[tid_str]) else None
                lon1, lat1 = pixel_to_lonlat(p1[1], p1[0], lon_c, lat_c)
                lon2, lat2 = pixel_to_lonlat(p2[1], p2[0], lon_c, lat_c)
                transect_features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
                    "properties": {
                        "aoi": aoi, "transectId": tid,
                        "eprMPerYr": epr_val,
                        "classification": classify(epr_val),
                    },
                })

    if not forecast_mutations:
        print("Tidak ada data untuk di-push — cek all_metrics/masks/lrr_vs_model.")
    else:
        # ---------- upload 3 file GeoJSON, push abrasionDataset singleton ----------
        current_fc   = {"type": "FeatureCollection", "features": current_features}
        predicted_fc = {"type": "FeatureCollection", "features": predicted_features}
        transects_fc = {"type": "FeatureCollection", "features": transect_features}

        asset_current   = upload_geojson_asset(current_fc, f"shoreline_current_{run_id}.geojson")
        asset_predicted = upload_geojson_asset(predicted_fc, f"shoreline_predicted_{run_id}.geojson") \
                          if predicted_features else None
        asset_transects = upload_geojson_asset(transects_fc, f"transects_{run_id}.geojson") \
                          if transect_features else None

        abrasion_doc = {
            "_type": "abrasionDataset",
            "_id":   ABRASION_SINGLETON_ID,
            "shorelineCurrent": {"_type": "file", "asset": {"_type": "reference", "_ref": asset_current}},
            "metrics": {
                "aoiCount": len(all_metrics),
                "meanNsm": round(float(np.mean(per_aoi_mean_nsm)), 3) if per_aoi_mean_nsm else 0,
                "meanEpr": round(float(np.mean(per_aoi_mean_epr)), 4) if per_aoi_mean_epr else 0,
                "meanLrr": round(float(np.mean(per_aoi_mean_lrr)), 4) if per_aoi_mean_lrr else 0,
            },
            "dataUpdatedAt": run_utc,
            "pipelineVersion": f"train256-{run_id}",
            "attribution": "Citra Sentinel-2 & Landsat · Tim KKN UNDIP",
            "predictionHorizonYear": TARGET_YEAR_CHECK,
            "isPublished": True,
        }
        if asset_predicted:
            abrasion_doc["shorelinePredicted"] = {
                "_type": "file", "asset": {"_type": "reference", "_ref": asset_predicted}}
        if asset_transects:
            abrasion_doc["transects"] = {
                "_type": "file", "asset": {"_type": "reference", "_ref": asset_transects}}

        push_batch([{"createOrReplace": abrasion_doc}], label="abrasionDataset (singleton)")

        # ---------- push shorelineForecast per AOI per tahun (buat marker+drawer) ----------
        BATCH = 100
        for start in range(0, len(forecast_mutations), BATCH):
            push_batch(forecast_mutations[start:start + BATCH], label=f"shorelineForecast batch {start // BATCH + 1}")

        print(f"\nDone.")
        print(f"  abrasionDataset: 1 doc, {len(current_features)} garis current, "
              f"{len(predicted_features)} garis predicted, {len(transect_features)} transect")
        print(f"  shorelineForecast: {len(forecast_mutations)} docs "
              f"({len(all_metrics)} AOI x rata-rata {len(forecast_mutations)//max(len(all_metrics),1)} tahun)")
```

**Catatan field yang sengaja dikosongin/disederhanakan** dibanding schema `abrasionDataset`
lengkap: `description`/`disclaimer` (teks i18n, editorial — biar diisi manual oleh
Pokdarwis via Studio, bukan dari pipeline, sesuai komentar di schema-nya sendiri: *"Every
label/heading... Pokdarwis... can edit this dataset's description"*). `iou`/`f1` juga
dikosongin — model `train256` ini gak dievaluasi pakai metrik itu.

---

## Yang masih perlu dikerjain di frontend (belum bagian notebook ini)

Sesuai yang diminta — marker per-AOI + drawer heatmap — ini kerja frontend React, bukan
notebook. Pattern yang udah ketemu buat direuse:
- **Marker**: `src/components/features/webgisv2/layers/poi-layer.tsx` (pattern marker yang
  sudah ada) — AOI marker query `shorelineForecast` grouped by `aoi`, ambil `aoiLon`/`aoiLat`.
- **Drawer**: `src/components/features/webgisv2/panels/webgis-place-detail.tsx` (pattern
  detail panel yang sudah ada saat marker diklik).
- **Chart heatmap**: `recharts` sudah ada di `package.json`, tinggal pakai `<AreaChart>`/
  custom heatmap grid buat transect × tahun.

Ini belum ditulis — mau lanjut ke situ setelah cell push ini fix?
