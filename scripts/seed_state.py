#!/usr/bin/env python3
"""Seed the production rolling history (`data/state/*_history.npz`) from the
archival offline masks (`data/offline_256_adaptive_cleaned/*_masks.npz`), and
generate `data/state/lrr_baseline.json`.

Why this exists: the production pipeline appends exactly ONE Sentinel-2 frame
per run (`src/gee/fetch.py` fetches a single ~120-day composite), so a fresh
pipeline has 1 frame of history -- too few for Track A inference
(`src/inference/predict.py` needs LOOKBACK=3) and for transect/LRR analysis
(`src/output/geojson.py` needs >=2). The archival masks already contain
22-23 seasonal frames per AOI (2018-2025), which is exactly the full
historical record the training notebook built its tensors from. Seeding the
rolling state from that archive lets the very next scheduled run produce real
predictions instead of waiting ~8 months of cold-start runs.

The LRR baseline is generated with the SAME geometry used at runtime
(`src/output/geojson.py`: build_baseline_transects over the rolling history
with n_transects=40 / length_px=4), so transect ids in the baseline file line
up with the transect ids geojson.py emits each run. The rate is fit in
calendar years (season_frac), and the per-transect position series is the
signed pixel distance from the baseline contour scaled to meters.

Usage:
  python -m scripts.seed_state
  python -m scripts.seed_state --offline-dir data/offline_256_adaptive_cleaned \
      --state-dir data/state --target-year 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from src.analysis.spatial import extract_main_contour, transect_intersect
from src.analysis.transect import build_baseline_transects, linear_regression_rate
from src.output.geojson import N_TRANSECTS, PATCH_SIZE, SCALE_M, TRANSECT_LENGTH_PX
from src.preprocessing.mask import history_path

SEASON_RANK = {"S1": 0, "S2": 1, "S3": 2}
SEASON_FRAC = {"S1": 0.15, "S2": 0.5, "S3": 0.85}


def _sort_key(key: str) -> tuple[int, int]:
    year_s, season = key.split("_")
    return int(year_s), SEASON_RANK.get(season, 99)


def seed_history(offline_dir: str, state_dir: str) -> dict[str, list[str]]:
    """Copy each AOI's chronological mask series into
    `data/state/{aoi}_history.npz` with frame_N keys, oldest first. Returns
    {aoi: [sorted key list]} for the LRR baseline step."""
    os.makedirs(state_dir, exist_ok=True)
    seeded: dict[str, list[str]] = {}
    for fname in sorted(os.listdir(offline_dir)):
        if not fname.endswith("_masks.npz"):
            continue
        aoi = fname.replace("_masks.npz", "")
        z = np.load(os.path.join(offline_dir, fname))
        keys = sorted(z.files, key=_sort_key)
        if not keys:
            print(f"[{aoi}] empty archive -- skipping")
            continue
        frames = [z[k] for k in keys]
        np.savez_compressed(
            history_path(state_dir, aoi),
            **{f"frame_{i}": f for i, f in enumerate(frames)})
        seeded[aoi] = keys
        print(f"[{aoi}] seeded {len(frames)} frame(s) -> {history_path(state_dir, aoi)}")
    return seeded


def generate_lrr_baseline(offline_dir: str, state_dir: str,
                          target_year: int) -> dict[str, dict[int, dict]]:
    """Fit per-transect LRR (m/yr) over the archival series using the exact
    transect geometry geojson.py uses at runtime, so `aoi_lrr.get(tid)` in
    src/output/geojson.py matches the transects it builds each run."""
    baseline: dict[str, dict[int, dict]] = {}
    for fname in sorted(os.listdir(offline_dir)):
        if not fname.endswith("_masks.npz"):
            continue
        aoi = fname.replace("_masks.npz", "")
        z = np.load(os.path.join(offline_dir, fname))
        keys = sorted(z.files, key=_sort_key)
        if not keys:
            continue
        frames = [z[k] for k in keys]
        mask_series = [(float(i), f) for i, f in enumerate(frames)]
        baseline_t, baseline_contour, transects = build_baseline_transects(
            mask_series, n_transects=N_TRANSECTS, length_px=TRANSECT_LENGTH_PX)

        per_transect: list[list[tuple[float, float]]] = [[] for _ in transects]
        for key, mask in zip(keys, frames):
            yr_s, season = key.split("_")
            t_year = int(yr_s) + SEASON_FRAC.get(season, 0.5)
            contour = extract_main_contour(mask, use_spline=True, spline_smoothing=2.0)
            if contour is None:
                continue
            pts = transect_intersect(transects, contour)
            pts_baseline = transect_intersect(transects, baseline_contour)
            for i, (p, p0) in enumerate(zip(pts, pts_baseline)):
                if p is None or p0 is None:
                    continue
                dist = float(np.hypot(p[0] - p0[0], p[1] - p0[1])) * SCALE_M
                per_transect[i].append((t_year, dist))

        aoi_baseline: dict[int, dict] = {}
        for tid, series in enumerate(per_transect):
            if len(series) < 2:
                continue
            t_vals = np.array([p[0] for p in series])
            pos_vals = np.array([p[1] for p in series])
            rate, intercept = linear_regression_rate(t_vals, pos_vals)
            aoi_baseline[tid] = {
                "rate_m_per_year": rate,
                "pred_dist": rate * target_year + intercept,
                "target_year": target_year,
                "n_obs": len(series),
            }
        baseline[aoi] = aoi_baseline
        print(f"[{aoi}] LRR baseline: {len(aoi_baseline)}/{len(transects)} transects fit "
              f"(baseline_t={baseline_t}, target_year={target_year})")
    return baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-dir", default="data/offline_256_adaptive_cleaned")
    ap.add_argument("--state-dir", default="data/state")
    ap.add_argument("--target-year", type=int, default=2026)
    args = ap.parse_args()

    if not os.path.isdir(args.offline_dir):
        print(f"offline dir not found: {args.offline_dir}", file=sys.stderr)
        sys.exit(1)

    seeded = seed_history(args.offline_dir, args.state_dir)
    baseline = generate_lrr_baseline(args.offline_dir, args.state_dir, args.target_year)

    out_path = os.path.join(args.state_dir, "lrr_baseline.json")
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Wrote {out_path} ({len(baseline)} AOIs)")
    print(f"Seeded {len(seeded)} AOI(s). Done.")


if __name__ == "__main__":
    main()
