#!/usr/bin/env python3
"""Postprocess entry point: combine Track A (predicted mask) and Track B
(transect NSM/EPR/LRR) into one GeoJSON + metrics.json per run.

Output schema (first version -- docs/skema_output.md was empty, documenting
here instead of leaving it implicit):

GeoJSON FeatureCollection, one Feature per transect per AOI:
  geometry: LineString [[lon,lat],[lon,lat]] -- the transect's actual two
            endpoints (converted from track_b_analysis's line_px), not a
            placeholder point
  properties: {aoi, transect_id, rate_m_per_year, classification, color,
               track_a_prob (model's predicted water-probability at the
               transect midpoint, if Track A ran for this AOI, else null)}

metrics.json: {run_utc, aoi: {aoi_name: {mean_epr_m_per_yr, n_transects,
               classification_counts, track_a_available: bool}}}

Usage: python -m src.output.geojson --state-dir data/state --pred-dir data/interim/run_<date> \
           --aoi-config config/aoi_points.geojson --out-dir data/interim/run_<date>
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from src.analysis.spatial import pixel_to_lonlat
from src.baseline.lrr_kalman import track_b_analysis
from src.preprocessing.mask import load_history

PATCH_SIZE = 256
SCALE_M = 10


def load_aoi_lonlat(config_path: str) -> dict[str, tuple[float, float]]:
    with open(config_path) as f:
        features = json.load(f)["features"]
    return {f["properties"]["name"]: tuple(f["geometry"]["coordinates"]) for f in features}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="data/state")
    ap.add_argument("--pred-dir", help="Track A output dir from src.inference.predict (optional)")
    ap.add_argument("--aoi-config", default="config/aoi_points.geojson")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    aoi_lonlat = load_aoi_lonlat(args.aoi_config)
    os.makedirs(args.out_dir, exist_ok=True)

    features = []
    metrics = {"run_utc": datetime.now(timezone.utc).isoformat(), "aoi": {}}

    for aoi_file in sorted(f for f in os.listdir(args.state_dir) if f.endswith("_history.npz")):
        aoi = aoi_file.replace("_history.npz", "")
        if aoi not in aoi_lonlat:
            continue
        lon_c, lat_c = aoi_lonlat[aoi]
        history = load_history(args.state_dir, aoi)
        if len(history) < 2:
            print(f"[{aoi}] SKIP -- need >=2 historical frames for transect analysis, have {len(history)}")
            continue

        mask_series = [(float(i), frame) for i, frame in enumerate(history)]
        result = track_b_analysis(aoi, mask_series, SCALE_M)

        track_a_prob = None
        if args.pred_dir:
            prob_path = os.path.join(args.pred_dir, f"{aoi}_pred_prob.npy")
            if os.path.exists(prob_path):
                track_a_prob = np.load(prob_path)

        epr_values, class_counts = [], Counter()
        for t in result["transects"]:
            if t["rate_m_per_year"] is None:
                continue
            epr_values.append(t["rate_m_per_year"])
            class_counts[t["classification"]] += 1

            (row1, col1), (row2, col2) = t["line_px"]
            lon1, lat1 = pixel_to_lonlat(row1, col1, lon_c, lat_c, PATCH_SIZE, SCALE_M)
            lon2, lat2 = pixel_to_lonlat(row2, col2, lon_c, lat_c, PATCH_SIZE, SCALE_M)

            mid_row, mid_col = int((row1 + row2) / 2), int((col1 + col2) / 2)
            prob_at_mid = None
            if track_a_prob is not None:
                mid_row = min(max(mid_row, 0), PATCH_SIZE - 1)
                mid_col = min(max(mid_col, 0), PATCH_SIZE - 1)
                prob_at_mid = float(track_a_prob[mid_row, mid_col])

            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
                "properties": {
                    "aoi": aoi, "transect_id": t["transect_id"],
                    "rate_m_per_year": t["rate_m_per_year"], "classification": t["classification"],
                    "color": t["color"], "track_a_prob": prob_at_mid,
                },
            })

        metrics["aoi"][aoi] = {
            "mean_epr_m_per_yr": float(np.mean(epr_values)) if epr_values else None,
            "n_transects": result["n_transects"],
            "classification_counts": dict(class_counts),
            "track_a_available": track_a_prob is not None,
        }
        print(f"[{aoi}] {len(epr_values)} valid transect(s), track_a={'yes' if track_a_prob is not None else 'no'}")

    geojson = {"type": "FeatureCollection", "features": features}
    with open(os.path.join(args.out_dir, "shoreline.geojson"), "w") as f:
        json.dump(geojson, f)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Done -> {args.out_dir}/shoreline.geojson + metrics.json")


if __name__ == "__main__":
    main()
