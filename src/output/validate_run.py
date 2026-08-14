#!/usr/bin/env python3
"""Output contract validation for one pipeline run (post-inference, NOT raw
GEE upstream). Validates the 5 structured files produced by src.output.geojson
against a declared schema + physical boundary values, mapping to the classic
data-engineering integration-test categories: schema drift / structural
contract, row-count reconciliation, PK uniqueness, referential integrity,
null/completeness gates, value bounds/outlier isolation, and freshness.

Runs where the pipeline's data first becomes structured (GeoJSON + metrics +
forecast docs). It is deliberately NOT applied to GEE's raw raster arrays --
those are unstructured by nature and already guarded by thresholds.json.

Usage: python -m src.output.validate_run --run-dir data/interim/run_<date>
       [--aoi-config config/aoi_points.geojson]
       [--bounds-json path/to/bounds.json]   # override any DEFAULT_BOUNDS key
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone

DEFAULT_BOUNDS = {
    "epr_min": -50.0,          # m/yr, physical envelope for EPR
    "epr_max": 50.0,           # m/yr
    "nsm_max": 3700.0,         # m  -- diagonal of a 256px patch @ 10m/px
    "n_transects_max": 40,     # matches N_TRANSECTS in src/output/geojson.py
    "aoi_buffer_m": 1500.0,    # AOI point buffer from config/aoi_points.geojson
    "max_run_age_days": 180,   # freshness SLA; older runs are flagged
    "lon_tol": 1e-4,           # lon/lat tolerance vs aoi config (referential)
}

FORECAST_REQUIRED = [
    "aoi", "aoiLon", "aoiLat", "year", "season", "runUtc",
    "meanEprMPerYr", "meanNsmM", "meanSceM", "meanLrrMPerYr",
    "classificationCounts", "nTransects", "transects",
]
TRANSECT_REQUIRED = [
    "id", "eprMPerYr", "nsmM", "sceM", "lrrMPerYr", "lrrProjectedM",
    "classification", "mcUncertainty", "modelPositions",
]
METRICS_REQUIRED = ["aoiCount", "meanNsm", "meanEpr", "meanLrr"]
GEOMETRY_TYPES = {"LineString"}
SEASONS = {"current", "predicted"}


def _load_json(path: str) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_finite(v) -> bool:
    if v is None:
        return True
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _parse_utc(run_utc: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(run_utc)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def validate_run(
    run_dir: str,
    aoi_config: str = "config/aoi_points.geojson",
    bounds: dict | None = None,
    strict: bool = True,
) -> list[str]:
    """Return a list of violations (empty = valid). One bad file raises unless
    `strict=False`, in which case the error is appended as a violation."""
    b = {**DEFAULT_BOUNDS, **(bounds or {})}
    issues: list[str] = []

    # ---------- 1. schema drift: structural contract per file ----------
    aoi_lonlat: dict[str, tuple[float, float]] = {}
    try:
        aoi_fc = _load_json(aoi_config)
        for feat in aoi_fc.get("features", []):
            name = feat.get("properties", {}).get("name")
            coords = feat.get("geometry", {}).get("coordinates")
            if name and coords:
                aoi_lonlat[name] = (coords[0], coords[1])
    except Exception as exc:
        issues.append(f"[referential] cannot load aoi_config {aoi_config}: {exc}")
        return issues

    metrics_path = os.path.join(run_dir, "metrics.json")
    forecast_path = os.path.join(run_dir, "forecast_docs.json")
    if not os.path.exists(metrics_path):
        issues.append(f"[schema] missing {metrics_path}")
    if not os.path.exists(forecast_path):
        issues.append(f"[schema] missing {forecast_path}")
    if issues:
        return issues

    metrics = _load_json(metrics_path)
    forecast_docs = _load_json(forecast_path)
    if not isinstance(forecast_docs, list):
        issues.append(f"[schema] forecast_docs.json must be a list, got {type(forecast_docs).__name__}")
        return issues

    for key in METRICS_REQUIRED:
        if key not in metrics:
            issues.append(f"[schema] metrics.json missing field '{key}'")

    # ---------- 2. row count reconciliation + null gate ----------
    aoi_count = metrics.get("aoiCount")
    if aoi_count is None:
        issues.append("[null] metrics.json missing aoiCount")
    elif int(aoi_count) != len(forecast_docs):
        issues.append(
            f"[count] metrics.aoiCount={aoi_count} != len(forecast_docs)={len(forecast_docs)}")
    if len(forecast_docs) == 0:
        issues.append("[null] forecast_docs.json is empty -- nothing to promote")

    run_utc_metrics = metrics.get("run_utc")

    # ---------- 3. PK uniqueness ----------
    seen_keys: set[tuple] = set()
    seen_transect: dict[str, set[int]] = {}
    for doc in forecast_docs:
        key = (doc.get("aoi"), doc.get("year"), doc.get("season"))
        if key in seen_keys:
            issues.append(f"[unique] duplicate forecast_docs key (aoi, year, season)={key}")
        seen_keys.add(key)

        for field in FORECAST_REQUIRED:
            if field not in doc:
                issues.append(f"[schema] forecast_docs[{doc.get('aoi')}] missing '{field}'")
            elif field in {"meanEprMPerYr", "meanNsmM", "meanSceM", "meanLrrMPerYr", "aoiLon", "aoiLat", "year", "nTransects"}:
                if not _is_finite(doc.get(field)):
                    issues.append(f"[bounds] forecast_docs[{doc.get('aoi')}].{field} is not finite")

        transects = doc.get("transects") or []
        if not isinstance(transects, list):
            issues.append(f"[schema] forecast_docs[{doc.get('aoi')}].transects not a list")
            continue
        if len(transects) == 0:
            issues.append(f"[null] forecast_docs[{doc.get('aoi')}] has 0 transects")
        if len(transects) > int(b["n_transects_max"]):
            issues.append(
                f"[bounds] forecast_docs[{doc.get('aoi')}].nTransects={len(transects)} > {b['n_transects_max']}")
        aoi_tids = seen_transect.setdefault(str(doc.get("aoi")), set())
        for t in transects:
            for field in TRANSECT_REQUIRED:
                if field not in t:
                    issues.append(
                        f"[schema] forecast_docs[{doc.get('aoi')}].transects missing '{field}'")
            tid = t.get("id")
            if tid is not None and tid in aoi_tids:
                issues.append(f"[unique] duplicate transectId={tid} in AOI {doc.get('aoi')}")
            aoi_tids.add(tid)
            epr = t.get("eprMPerYr")
            if epr is not None and not _is_finite(epr):
                issues.append(f"[bounds] AOI {doc.get('aoi')} transect {tid} eprMPerYr not finite")
            if epr is not None and not (b["epr_min"] <= float(epr) <= b["epr_max"]):
                issues.append(
                    f"[bounds] AOI {doc.get('aoi')} transect {tid} eprMPerYr={epr} outside "
                    f"[{b['epr_min']}, {b['epr_max']}] m/yr")
            nsm = t.get("nsmM")
            if nsm is not None and _is_finite(nsm) and float(nsm) > b["nsm_max"]:
                issues.append(f"[bounds] AOI {doc.get('aoi')} transect {tid} nsmM={nsm} > {b['nsm_max']} m")
            lrr = t.get("lrrMPerYr")
            if lrr is not None and _is_finite(lrr) and not (b["epr_min"] <= float(lrr) <= b["epr_max"]):
                issues.append(
                    f"[bounds] AOI {doc.get('aoi')} transect {tid} lrrMPerYr={lrr} outside EPR bounds")

        # ---------- 4. referential integrity ----------
        aoi_name = doc.get("aoi")
        if aoi_name not in aoi_lonlat:
            issues.append(f"[referential] AOI '{aoi_name}' not in {aoi_config}")
        else:
            lon_c, lat_c = aoi_lonlat[aoi_name]
            lon_d, lat_d = doc.get("aoiLon"), doc.get("aoiLat")
            if lon_d is not None and lat_d is not None:
                if abs(lon_d - lon_c) > b["lon_tol"] or abs(lat_d - lat_c) > b["lon_tol"]:
                    issues.append(
                        f"[referential] AOI '{aoi_name}' lon/lat drift: doc=({lon_d},{lat_d}) "
                        f"vs config=({lon_c},{lat_c})")

    # ---------- GeoJSON feature-level schema ----------
    for fname, expect_season in (
        ("shoreline_current.geojson", "current"),
        ("shoreline_predicted.geojson", "predicted"),
    ):
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            issues.append(f"[schema] missing {fname}")
            continue
        fc = _load_json(path)
        if fc.get("type") != "FeatureCollection":
            issues.append(f"[schema] {fname} type != FeatureCollection")
            continue
        for feat in fc.get("features", []):
            if feat.get("type") != "Feature":
                issues.append(f"[schema] {fname} has non-Feature element")
                continue
            geom = feat.get("geometry", {})
            if geom.get("type") not in GEOMETRY_TYPES:
                issues.append(f"[schema] {fname} geometry.type not in {GEOMETRY_TYPES}")
            props = feat.get("properties", {}) or {}
            if props.get("season") != expect_season:
                issues.append(f"[schema] {fname} feature season='{props.get('season')}' expected '{expect_season}'")

    # ---------- 7. freshness ----------
    run_utc = run_utc_metrics or (forecast_docs[0].get("runUtc") if forecast_docs else None)
    dt = _parse_utc(run_utc) if run_utc else None
    if dt is None:
        issues.append("[freshness] cannot parse run_utc")
    else:
        now = datetime.now(timezone.utc)
        if dt > now:
            issues.append(f"[freshness] run_utc {run_utc} is in the future")
        age_days = (now - dt).total_seconds() / 86400.0
        if age_days > float(b["max_run_age_days"]):
            issues.append(
                f"[freshness] run_utc {run_utc} is {age_days:.1f} days old "
                f"(> {b['max_run_age_days']} days)")
    if run_utc_metrics and forecast_docs:
        for doc in forecast_docs:
            if doc.get("runUtc") and doc.get("runUtc") != run_utc_metrics:
                issues.append(
                    f"[freshness] forecast_docs[{doc.get('aoi')}].runUtc differs from metrics.run_utc")

    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--aoi-config", default="config/aoi_points.geojson")
    ap.add_argument("--bounds-json", help="optional JSON file overriding DEFAULT_BOUNDS keys")
    args = ap.parse_args()

    bounds = None
    if args.bounds_json:
        with open(args.bounds_json, encoding="utf-8") as f:
            bounds = json.load(f)

    issues = validate_run(args.run_dir, aoi_config=args.aoi_config, bounds=bounds)
    if issues:
        print(f"VALIDATION FAILED ({len(issues)} issue(s)):")
        for msg in issues:
            print(f"  - {msg}")
        raise SystemExit(1)
    print("VALIDATION PASSED: output contract, bounds, and freshness OK.")


if __name__ == "__main__":
    main()