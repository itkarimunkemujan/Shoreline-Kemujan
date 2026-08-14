#!/usr/bin/env python3
"""Postprocess entry point: combine Track A (model inference, fresh each run)
and the last full historical LRR analysis (NOT recomputed each run -- see
below) into the exact payload shape `src.output.sanity_push` needs to fill
`abrasionDataset` (singleton, 3 GeoJSON files + aggregate metrics) and
`shorelineForecast` (per-AOI-per-year documents), matching
`tourism-kemujan/studio/schemaTypes/{singletons/abrasionDataset.ts,
documents/shorelineForecast.ts}`.

Why LRR isn't recomputed here: `data/state/*_history.npz` (see
`src.preprocessing.mask.update_history`) only retains a rolling LOOKBACK-sized
window (currently 3 frames) -- enough for Track A's next-step inference, but
statistically too thin to refit a meaningful linear-regression trend (LRR is
supposed to be a slow-moving baseline fit over years of observations, like
the full historical run in `notebooks/train_256_final.ipynb` did). Refitting
it from 3 points every month would produce noisy, low-confidence numbers
masquerading as a stable trend. Instead, this script:

  - Computes EPR/NSM/SCE fresh every run from whatever rolling-history
    frames are available (`src.analysis.transect.epr_per_year`) -- this is a
    legitimate "recent observed change" signal even with few points, and IS
    meant to update every run.
  - Reads `lrrMPerYr`/`pred_dist`/`target_year` per transect from a
    `--lrr-baseline` JSON file (produced ONCE from a full historical
    analysis, not by this pipeline -- see `export_lrr_baseline()` docstring
    below for how to generate it) and re-extrapolates `lrrProjectedM` for
    THIS run's year using that fixed rate. If the file is missing,
    `lrrMPerYr`/`lrrProjectedM` are simply omitted (null) rather than
    fabricated -- an honest "no baseline yet" state, not a silent zero.

Usage: python -m src.output.geojson --state-dir data/state --pred-dir data/interim/run_<date> \\
           --aoi-config config/aoi_points.geojson --out-dir data/interim/run_<date> \\
           [--lrr-baseline data/state/lrr_baseline.json]
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from src.analysis.metrics import classify_epr, nsm as compute_nsm, sce as compute_sce
from src.analysis.spatial import extract_main_contour, pixel_to_lonlat
from src.analysis.transect import build_baseline_transects, epr_per_year
from src.preprocessing.mask import load_history

PATCH_SIZE = 256
SCALE_M = 10
N_TRANSECTS = 40
TRANSECT_LENGTH_PX = 4  # matches the frontend-facing fix in claude_result14.md


def load_aoi_lonlat(config_path: str) -> dict[str, tuple[float, float]]:
    with open(config_path) as f:
        features = json.load(f)["features"]
    return {f["properties"]["name"]: tuple(f["geometry"]["coordinates"]) for f in features}


def load_lrr_baseline(path: str | None) -> dict[str, dict[int, dict]]:
    """{aoi: {transect_id: {"rate_m_per_year": float, "pred_dist": float,
    "target_year": int}}}. Returns {} (not an error) if `path` is None or the
    file doesn't exist yet -- see module docstring: no baseline is a valid,
    expected state before the first full historical analysis has run.

    To PRODUCE this file (one-time, from a full historical analysis session
    like `notebooks/train_256_final.ipynb`'s `compute_lrr_and_compare`),
    export `lrr_results` per AOI as:
        {aoi: {tid: {"rate_m_per_year": v["rate_m_per_year"],
                      "pred_dist": v["pred_dist"], "target_year": TARGET_YEAR_CHECK}
               for tid, v in lrr_results.items() if v}}
    and json.dump the combined dict to this path.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {aoi: {int(tid): entry for tid, entry in transects.items()}
            for aoi, transects in raw.items()}


def run_geojson(state_dir: str, out_dir: str, pred_dir: str | None = None,
                 aoi_config: str = "config/aoi_points.geojson",
                 lrr_baseline_path: str = "data/state/lrr_baseline.json") -> dict[str, str]:
    """Core postprocess logic, callable directly (Prefect task wraps this) as
    well as via the CLI `main()` below. Returns {name: path} for the 5
    output files written, so callers (the Sanity push stage) don't need to
    hardcode/guess the filenames."""
    aoi_lonlat = load_aoi_lonlat(aoi_config)
    lrr_baseline = load_lrr_baseline(lrr_baseline_path)
    os.makedirs(out_dir, exist_ok=True)

    run_utc = datetime.now(timezone.utc).isoformat()
    run_year = datetime.now(timezone.utc).year

    current_features: list[dict] = []
    predicted_features: list[dict] = []
    transect_features: list[dict] = []
    forecast_docs: list[dict] = []
    per_aoi_mean_epr, per_aoi_mean_nsm, per_aoi_mean_lrr = [], [], []

    for aoi_file in sorted(f for f in os.listdir(state_dir) if f.endswith("_history.npz")):
        aoi = aoi_file.replace("_history.npz", "")
        if aoi not in aoi_lonlat:
            print(f"[{aoi}] SKIP -- not in {aoi_config}")
            continue
        try:
            lon_c, lat_c = aoi_lonlat[aoi]
            history = load_history(state_dir, aoi)
            if len(history) < 2:
                print(f"[{aoi}] SKIP -- need >=2 historical frames for transect analysis, have {len(history)}")
                continue

            mask_series = [(float(i), frame) for i, frame in enumerate(history)]
            baseline_t, baseline_contour, transects = build_baseline_transects(
                mask_series, n_transects=N_TRANSECTS, length_px=TRANSECT_LENGTH_PX)
            epr_by_t = epr_per_year(mask_series, transects, baseline_t, baseline_contour, SCALE_M)

            latest_t = mask_series[-1][0]
            latest_epr = epr_by_t.get(latest_t, [None] * len(transects))
            epr_matrix = np.array([
                [v if v is not None else np.nan for v in epr_by_t[t]] for t, _ in mask_series
            ])
            total_dt = max(latest_t - baseline_t, 1)
            sce_per_transect = compute_sce(epr_matrix, total_dt)

            aoi_lrr = lrr_baseline.get(aoi, {})

            # ---------- current shoreline (latest rolling-history frame) ----------
            _, latest_mask = mask_series[-1]
            latest_contour = extract_main_contour(latest_mask, use_spline=True, spline_smoothing=2.0)
            if latest_contour is not None:
                coords = [[round(lo, 6), round(la, 6)] for lo, la in (
                    pixel_to_lonlat(r, c, lon_c, lat_c, PATCH_SIZE, SCALE_M) for r, c in latest_contour
                )]
                current_features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"aoi": aoi, "year": run_year, "season": "current", "source": "real"},
                })

            # ---------- predicted shoreline (Track A, if available) ----------
            if pred_dir:
                pred_mask_path = os.path.join(pred_dir, f"{aoi}_pred_mask.npy")
                if os.path.exists(pred_mask_path):
                    pred_contour = extract_main_contour(np.load(pred_mask_path), use_spline=True, spline_smoothing=2.0)
                    if pred_contour is not None:
                        coords = [[round(lo, 6), round(la, 6)] for lo, la in (
                            pixel_to_lonlat(r, c, lon_c, lat_c, PATCH_SIZE, SCALE_M) for r, c in pred_contour
                        )]
                        predicted_features.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": {"aoi": aoi, "year": run_year + 1, "season": "predicted", "source": "rollout"},
                        })

            # ---------- per-transect: fresh EPR/NSM/SCE + carried-forward LRR ----------
            transects_out = []
            clf_counts: dict[str, int] = {}
            for tid, (p1, p2) in enumerate(transects):
                epr_val = latest_epr[tid] if tid < len(latest_epr) else None
                # nsm() only ever reads the series' last element * total_dt_years,
                # so a 1-element list carrying the current EPR is sufficient here.
                nsm_val = compute_nsm([epr_val], total_dt) if epr_val is not None else None
                sce_val = float(sce_per_transect[tid]) if tid < len(sce_per_transect) and not np.isnan(sce_per_transect[tid]) else None
                classification, _ = classify_epr(epr_val)
                clf_counts[classification] = clf_counts.get(classification, 0) + 1

                lrr_entry = aoi_lrr.get(tid)
                lrr_rate = lrr_entry["rate_m_per_year"] if lrr_entry else None
                lrr_projected = None
                if lrr_entry:
                    lrr_projected = lrr_entry["pred_dist"] + lrr_entry["rate_m_per_year"] * (
                        run_year - lrr_entry["target_year"])

                lon1, lat1 = pixel_to_lonlat(p1[1], p1[0], lon_c, lat_c, PATCH_SIZE, SCALE_M)
                lon2, lat2 = pixel_to_lonlat(p2[1], p2[0], lon_c, lat_c, PATCH_SIZE, SCALE_M)
                transect_features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
                    "properties": {
                        "aoi": aoi, "transectId": tid, "year": run_year,
                        "eprMPerYr": epr_val, "classification": classification,
                    },
                })
                transects_out.append({
                    "id": tid,
                    "eprMPerYr": round(epr_val, 4) if epr_val is not None else None,
                    "nsmM": round(nsm_val, 3) if nsm_val is not None else None,
                    "sceM": round(sce_val, 3) if sce_val is not None else None,
                    "lrrMPerYr": round(lrr_rate, 4) if lrr_rate is not None else None,
                    "lrrProjectedM": round(lrr_projected, 3) if lrr_projected is not None else None,
                    "classification": classification,
                    "mcUncertainty": None,
                    "modelPositions": [],
                })

            mean_epr = float(np.nanmean([v for v in latest_epr if v is not None])) if any(v is not None for v in latest_epr) else None
            mean_nsm = float(np.nanmean([t["nsmM"] for t in transects_out if t["nsmM"] is not None])) \
                if any(t["nsmM"] is not None for t in transects_out) else None
            lrr_rates = [t["lrrMPerYr"] for t in transects_out if t["lrrMPerYr"] is not None]
            mean_lrr = float(np.mean(lrr_rates)) if lrr_rates else None

            forecast_docs.append({
                "aoi": aoi, "aoiLon": lon_c, "aoiLat": lat_c,
                "year": run_year, "season": "current", "runUtc": run_utc,
                "meanEprMPerYr": round(mean_epr, 4) if mean_epr is not None else None,
                "meanNsmM": round(mean_nsm, 3) if mean_nsm is not None else None,
                "meanSceM": round(float(np.nanmean(sce_per_transect)), 3) if sce_per_transect.size else None,
                "meanLrrMPerYr": round(mean_lrr, 4) if mean_lrr is not None else None,
                "classificationCounts": [{"label": k, "count": v} for k, v in clf_counts.items()],
                "nTransects": len(transects_out),
                "transects": transects_out,
            })

            if mean_epr is not None:
                per_aoi_mean_epr.append(mean_epr)
            if mean_nsm is not None:
                per_aoi_mean_nsm.append(mean_nsm)
            if mean_lrr is not None:
                per_aoi_mean_lrr.append(mean_lrr)

            print(f"[{aoi}] OK -- {len(transects_out)} transects, mean EPR="
                  f"{mean_epr if mean_epr is not None else 'n/a'}, "
                  f"LRR baseline={'yes' if aoi_lrr else 'no'}")
        except Exception as exc:  # noqa: BLE001 -- one bad AOI shouldn't sink the whole run
            print(f"[{aoi}] SKIP -- unexpected error: {exc}")
            continue

    current_fc = {"type": "FeatureCollection", "features": current_features}
    predicted_fc = {"type": "FeatureCollection", "features": predicted_features}
    transects_fc = {"type": "FeatureCollection", "features": transect_features}

    metrics = {
        "aoiCount": len(forecast_docs),
        "meanNsm": round(float(np.mean(per_aoi_mean_nsm)), 3) if per_aoi_mean_nsm else 0,
        "meanEpr": round(float(np.mean(per_aoi_mean_epr)), 4) if per_aoi_mean_epr else 0,
        "meanLrr": round(float(np.mean(per_aoi_mean_lrr)), 4) if per_aoi_mean_lrr else 0,
    }

    output_paths = {
        "current_geojson": os.path.join(out_dir, "shoreline_current.geojson"),
        "predicted_geojson": os.path.join(out_dir, "shoreline_predicted.geojson"),
        "transects_geojson": os.path.join(out_dir, "transects.geojson"),
        "metrics_json": os.path.join(out_dir, "metrics.json"),
        "forecast_docs_json": os.path.join(out_dir, "forecast_docs.json"),
    }
    with open(output_paths["current_geojson"], "w") as f:
        json.dump(current_fc, f)
    with open(output_paths["predicted_geojson"], "w") as f:
        json.dump(predicted_fc, f)
    with open(output_paths["transects_geojson"], "w") as f:
        json.dump(transects_fc, f)
    with open(output_paths["metrics_json"], "w") as f:
        json.dump({"run_utc": run_utc, **metrics}, f, indent=2)
    with open(output_paths["forecast_docs_json"], "w") as f:
        json.dump(forecast_docs, f, indent=2)

    print(f"Done -> {out_dir}/ "
          f"({len(current_features)} current, {len(predicted_features)} predicted, "
          f"{len(transect_features)} transect, {len(forecast_docs)} forecast docs)")
    return output_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="data/state")
    ap.add_argument("--pred-dir", help="Track A output dir from src.inference.predict (optional)")
    ap.add_argument("--aoi-config", default="config/aoi_points.geojson")
    ap.add_argument("--lrr-baseline", default="data/state/lrr_baseline.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run_geojson(args.state_dir, args.out_dir, args.pred_dir, args.aoi_config, args.lrr_baseline)


if __name__ == "__main__":
    main()
