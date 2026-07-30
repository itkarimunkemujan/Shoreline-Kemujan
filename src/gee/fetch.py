#!/usr/bin/env python3
"""Entry point: fetch the latest Sentinel-2 composite, compute MNDWI per AOI,
save raw arrays for preprocessing/mask.py. The only script in this pipeline
that talks to GEE.

Usage: python -m src.gee.fetch --out-dir data/interim/run_<date>
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta

import ee
import numpy as np

from src.gee.auth import initialize_ee
from src.gee.composite import fetch_sentinel_composite, get_union_bbox
from src.gee.export import download_patch

GEE_PROJECT = os.environ.get("GEE_PROJECT", "gen-lang-client-0412358476")
PATCH_SIZE = 256
SCALE_M = 10
LOOKBACK_DAYS = 120  # composite window ending today; matches ~1 season


def load_aoi(config_path: str) -> list[dict]:
    with open(config_path) as f:
        return json.load(f)["features"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi-config", default="config/aoi_points.geojson")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cloud-max", type=float, default=30)
    args = ap.parse_args()

    initialize_ee(GEE_PROJECT)
    aoi_features = load_aoi(args.aoi_config)
    union_bbox = get_union_bbox(aoi_features)

    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)
    composite, n_scene = fetch_sentinel_composite(
        union_bbox, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), args.cloud_max)
    if composite is None:
        raise RuntimeError(f"No Sentinel-2 scenes found in the last {LOOKBACK_DAYS} days (cloud_max={args.cloud_max})")

    os.makedirs(args.out_dir, exist_ok=True)
    run_meta = {"run_utc": datetime.utcnow().isoformat(), "n_scene": n_scene,
                "window_start": start.strftime("%Y-%m-%d"), "window_end": end.strftime("%Y-%m-%d"),
                "aoi": []}

    for feature in aoi_features:
        name = feature["properties"]["name"]
        lon, lat = feature["geometry"]["coordinates"]
        region = ee.Geometry.Point([lon, lat]).buffer(PATCH_SIZE * SCALE_M / 2).bounds()
        mndwi = download_patch(composite, "MNDWI", region, SCALE_M)
        np.save(os.path.join(args.out_dir, f"{name}_mndwi.npy"), mndwi)
        run_meta["aoi"].append({"name": name, "lon": lon, "lat": lat})
        print(f"[{name}] MNDWI patch OK -- shape {mndwi.shape}")

    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Done: {len(aoi_features)} AOI, {n_scene} scene(s), -> {args.out_dir}")


if __name__ == "__main__":
    main()
