# claude_result12.md — Cell 11 fix: drop invalid `geojson` field + add LRR projection

## Kenapa direvisi lagi

`sanity deploy` di `tourism-kemujan/studio` gagal:

```
Forecast › geojson.geometry.coordinates ›
  ✖ Found array member declaration of type "array" - multidimensional arrays are not
    currently supported by Sanity
```

`shorelineForecast.geojson.geometry.coordinates` didefinisikan sebagai array-of-array
(`type:'array', of:[{type:'array', of:[{type:'number'}]}]`) — persis bentuk koordinat
GeoJSON LineString (`[[lon,lat], [lon,lat], ...]`). Sanity **gak support array
multidimensi sama sekali** di schema-nya.

**Fix**: field `geojson` di `shorelineForecast` **dihapus total**, bukan di-workaround.
Garis shoreline-nya sendiri gak hilang — dia tetap ada di 3 file yang di-upload ke
`abrasionDataset` (`shorelineCurrent`/`shorelinePredicted`, masing-masing FeatureCollection
gabungan semua AOI semua tahun, tiap Feature udah ditandain `properties.aoi`/
`properties.year`). Buat ambil garis 1 AOI tertentu nanti di frontend: fetch file itu
(sekali, udah kejadian buat overlay utama), terus filter client-side
`features.filter(f => f.properties.aoi === "Titik_02")`. Gak perlu query/field terpisah.

**Sekalian ditambahin**: field `lrrProjectedM` per transect (proyeksi posisi LRR di tahun
dokumen itu, diekstrapolasi dari `rate_m_per_year` + `pred_dist` yang udah dihitung
`compute_lrr_and_compare` di Cell 10 — **Cell 10 gak berubah sama sekali**). Ini yang
bakal jadi kurva biru "Model vs LRR" kalau nanti digambar di frontend, sejajar sama
`nsmM` (posisi hasil model).

**File yang berubah cuma 2**: `tourism-kemujan/studio/schemaTypes/documents/shorelineForecast.ts`
(sudah di-patch: `geojson` dihapus, `lrrProjectedM` ditambah) dan Cell 11 di bawah ini
(pengganti Cell 11 di `claude_result11.md`). Cell 10 dan Cell 12 (`shorelineModelRun`)
**tidak berubah**.

---

## Cell 11 (REVISI) — Push `abrasionDataset` + `shorelineForecast` (tanpa geojson, + LRR projection)

```python
# ================================================================
# CELL 11 — Push ke Sanity: abrasionDataset (singleton, overlay utama)
# + shorelineForecast (per-AOI-per-tahun, buat marker+drawer heatmap)
# FIX: geojson di-drop dari shorelineForecast (Sanity gak support
# array-of-array, lihat studio deploy error) — geometri tetap ada,
# cuma lewat file upload abrasionDataset (current_features/predicted_features).
# BARU: lrrProjectedM per transect per tahun (proyeksi LRR, diekstrapolasi
# dari rate_m_per_year + pred_dist yang udah dihitung compute_lrr_and_compare
# di Cell 10 — gak perlu ubah Cell 10 sama sekali).
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

    ABRASION_SINGLETON_ID = "singleton-abrasionDataset"

    def push_batch(mutations, label=""):
        resp = requests.post(mutate_url, json={"mutations": mutations},
                              headers={"Authorization": f"Bearer {token}"}, timeout=60)
        resp.raise_for_status()
        print(f"Sanity push OK: {len(mutations)} docs  {label}")
        return resp.json()

    def upload_geojson_asset(feature_collection, filename):
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

    run_utc = datetime.now(tz=timezone.utc).isoformat()

    current_features   = []
    predicted_features = []
    transect_features  = []
    forecast_mutations = []

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

                # proyeksi posisi LRR di tahun ini (ekstrapolasi linear dari
                # pred_dist@TARGET_YEAR_CHECK + rate_m_per_year)
                lrr_pred_m = None
                if lrr_entry:
                    lrr_pred_m = lrr_entry["pred_dist"] + lrr_entry["rate_m_per_year"] * (year - TARGET_YEAR_CHECK)

                transects_out.append({
                    "id": tid,
                    "eprMPerYr": round(epr_val, 4) if epr_val is not None else None,
                    "nsmM": round(nsm_val, 3) if nsm_val is not None else None,
                    "sceM": round(float(sce_full[tid_str]), 3) if not np.isnan(sce_full[tid_str]) else None,
                    "lrrMPerYr": round(lrr_entry["rate_m_per_year"], 4) if lrr_entry else None,
                    "lrrProjectedM": round(lrr_pred_m, 3) if lrr_pred_m is not None else None,
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
                # "geojson" SENGAJA gak diikutin — Sanity gak support
                # array-of-array. Garis shoreline tetap ada lewat file upload
                # abrasionDataset di bawah (current_features/predicted_features).
                "meanEprMPerYr": round(mean_epr_yr, 4) if not np.isnan(mean_epr_yr) else None,
                "meanNsmM": round(mean_nsm_yr, 3) if mean_nsm_yr is not None else None,
                "meanSceM": round(mean_sce_yr, 3) if not np.isnan(mean_sce_yr) else None,
                "meanLrrMPerYr": round(mean_lrr_aoi, 4) if mean_lrr_aoi is not None else None,
                "classificationCounts": classification_counts,
                "nTransects": len(transects_out),
                "transects": transects_out,
            }})

            if year == last_year:
                per_aoi_mean_epr.append(mean_epr_yr)
                if mean_nsm_yr is not None:
                    per_aoi_mean_nsm.append(mean_nsm_yr)
                if mean_lrr_aoi is not None:
                    per_aoi_mean_lrr.append(mean_lrr_aoi)

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

        BATCH = 100
        for start in range(0, len(forecast_mutations), BATCH):
            push_batch(forecast_mutations[start:start + BATCH], label=f"shorelineForecast batch {start // BATCH + 1}")

        print(f"\nDone.")
        print(f"  abrasionDataset: 1 doc, {len(current_features)} garis current, "
              f"{len(predicted_features)} garis predicted, {len(transect_features)} transect")
        print(f"  shorelineForecast: {len(forecast_mutations)} docs")
```

## Cell 12 — SAMA PERSIS

Tidak berubah dari `claude_result10.md` (`shorelineModelRun`).

---

## Sebelum push `shorelineForecast` bisa lolos

`sanity deploy` di `tourism-kemujan/studio` harus di-rerun dulu supaya schema
`shorelineForecast.ts` yang sudah dibenerin (drop `geojson`, tambah `lrrProjectedM`)
kepublish. Push lewat HTTP mutate API technically gak divalidasi ketat oleh schema Studio
(beda dari `sanity deploy`), tapi dokumen yang punya field gak dikenal Studio bakal susah
di-browse/query lewat Vision/Studio UI — mending deploy dulu biar bersih.
