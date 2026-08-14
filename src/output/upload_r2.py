#!/usr/bin/env python3
"""Upload one run's output to a Cloudflare R2 bucket (S3-compatible) as the
raw/immutable warehouse zone. Writes, under `runs/<run_id>/`:

  shoreline_current.geojson   (unchanged, verbatim from the run)
  shoreline_predicted.geojson (verbatim)
  transects.geojson           (verbatim)
  metrics.json                (verbatim, contains run_utc + aggregate metrics)
  forecast_docs.json          (verbatim, per-AOI-per-year payload)
  run_meta.json               (verbatim from src.gee.fetch, if present)
  *_parquet/*.parquet         (one Parquet per GeoJSON via DuckDB, queryable)
  run_manifest.json           (summary + sha256 checksum of every file)

This is a landing zone ONLY -- nothing here is consumed by the frontend. The
promote step (src.pipeline_promote.py) later validates the manifest and pushes
the selected run to Sanity `production`.

SKIPs cleanly (like src.output.sanity_push) if any required R2 env var is
missing, so running the pipeline locally without R2 never breaks the flow.

Requires env vars (GitHub Secrets):
  R2_ACCOUNT_ID        -- Cloudflare account ID
  R2_ACCESS_KEY_ID     -- R2 API token access key
  R2_ACCESS_KEY_SECRET -- R2 API token secret
  R2_BUCKET            -- bucket name, e.g. "shoreline-kemujan"

Usage: python -m src.output.upload_r2 --run-dir data/interim/run_<date>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import duckdb
import s3fs

REQUIRED_ENV = ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_ACCESS_KEY_SECRET", "R2_BUCKET"]
UPLOADABLE_FILES = [
    "shoreline_current.geojson",
    "shoreline_predicted.geojson",
    "transects.geojson",
    "metrics.json",
    "forecast_docs.json",
]
OPTIONAL_FILES = ["run_meta.json"]
PARQUET_JSON_FILES = {
    "shoreline_current.geojson": "current_features",
    "shoreline_predicted.geojson": "predicted_features",
    "transects.geojson": "transects",
}
PATCH_SIZE = 256
SCALE_M = 10


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _r2_fs() -> tuple[s3fs.S3FileSystem, str, str]:
    account_id = os.environ["R2_ACCOUNT_ID"]
    bucket = os.environ["R2_BUCKET"]
    fs = s3fs.S3FileSystem(
        key=os.environ["R2_ACCESS_KEY_ID"],
        secret=os.environ["R2_ACCESS_KEY_SECRET"],
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        client_kwargs={"region_name": "auto"},
    )
    return fs, bucket, account_id


def _run_id(run_dir: str, run_utc: str) -> str:
    """Deterministic-ish id: prefer the run directory's own name when it
    already looks like run_<stamp>, else derive from run_utc."""
    base = os.path.basename(run_dir.rstrip("/\\"))
    if base.startswith("run_"):
        return base
    return "run_" + run_utc[:10].replace("-", "")


def _write_json_to_parquet(duck: duckdb.DuckDBPyConnection, path: str, out_key: str) -> str | None:
    """Flatten a GeoJSON FeatureCollection into one Parquet file via DuckDB
    (properties->columns, geometry->lon/lat-derived rows). Returns the relative
    object key written, or None if the FeatureCollection is empty."""
    with open(path, encoding="utf-8") as f:
        fc = json.load(f)
    features = fc.get("features", [])
    if not features:
        return None
    rows = []
    for feat in features:
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") == "LineString" and coords:
            row = dict(props)
            row["n_coords"] = len(coords)
            row["first_lon"] = coords[0][0] if len(coords[0]) > 0 else None
            row["first_lat"] = coords[0][1] if len(coords[0]) > 1 else None
            row["last_lon"] = coords[-1][0] if len(coords[-1]) > 0 else None
            row["last_lat"] = coords[-1][1] if len(coords[-1]) > 1 else None
            rows.append(row)
        else:
            rows.append({**props, "n_coords": 0, "first_lon": None, "first_lat": None,
                         "last_lon": None, "last_lat": None})
    duck.register("tbl", rows)
    duck.execute(f"COPY (SELECT * FROM tbl) TO '{out_key}' (FORMAT PARQUET)")
    duck.unregister("tbl")
    return out_key


def run_upload_r2(run_dir: str) -> dict:
    """Core upload logic, callable directly (Prefect task wraps this) as well
    as via the CLI `main()` below. Returns a small summary dict."""
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(f"SKIP R2 upload -- missing env var(s): {missing}.", file=sys.stderr)
        return {"uploaded": False, "reason": f"missing env vars: {missing}"}

    fs, bucket, _ = _r2_fs()

    run_utc = None
    run_meta_path = os.path.join(run_dir, "run_meta.json")
    if os.path.exists(run_meta_path):
        with open(run_meta_path, encoding="utf-8") as f:
            run_meta = json.load(f)
        run_utc = run_meta.get("run_utc")
    if not run_utc:
        run_utc = datetime.now(timezone.utc).isoformat()

    run_id = _run_id(run_dir, run_utc)
    prefix = f"runs/{run_id}"
    manifest = {
        "run_id": run_id,
        "run_utc": run_utc,
        "run_dir": os.path.basename(run_dir.rstrip("/\\")),
        "uploaded_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    duck = duckdb.connect()
    duck.execute("SET threads=1")
    try:
        for name in UPLOADABLE_FILES:
            local = os.path.join(run_dir, name)
            if not os.path.exists(local):
                print(f"WARN missing {name} -- skipping.", file=sys.stderr)
                continue
            key = f"{prefix}/{name}"
            with fs.open(f"{bucket}/{key}", "wb") as dst, open(local, "rb") as src:
                dst.write(src.read())
            manifest["files"][name] = {
                "key": key, "size": os.path.getsize(local), "sha256": _sha256(local),
            }
            print(f"Uploaded {name} -> s3://{bucket}/{key}")

        for name in OPTIONAL_FILES:
            local = os.path.join(run_dir, name)
            if os.path.exists(local):
                key = f"{prefix}/{name}"
                with fs.open(f"{bucket}/{key}", "wb") as dst, open(local, "rb") as src:
                    dst.write(src.read())
                manifest["files"][name] = {
                    "key": key, "size": os.path.getsize(local), "sha256": _sha256(local),
                }
                print(f"Uploaded {name} -> s3://{bucket}/{key}")

        for json_name, table_label in PARQUET_JSON_FILES.items():
            local = os.path.join(run_dir, json_name)
            if not os.path.exists(local):
                continue
            out_key = f"{prefix}/parquet/{json_name.replace('.geojson', '')}.parquet"
            written = _write_json_to_parquet(duck, local, f"s3://{bucket}/{out_key}")
            if written:
                manifest["files"].setdefault("parquet", {})[json_name] = {"key": out_key}
                print(f"Parquet {json_name} -> s3://{bucket}/{out_key}")

        # scalar summary for the promote step's DuckDB query
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            manifest["summary"] = {
                "aoiCount": metrics.get("aoiCount", 0),
                "meanNsm": metrics.get("meanNsm"),
                "meanEpr": metrics.get("meanEpr"),
                "meanLrr": metrics.get("meanLrr"),
            }
    finally:
        duck.close()

    manifest_key = f"{prefix}/run_manifest.json"
    with fs.open(f"{bucket}/{manifest_key}", "wb") as dst:
        dst.write(json.dumps(manifest, indent=2).encode("utf-8"))
    print(f"Manifest -> s3://{bucket}/{manifest_key}")

    return {"uploaded": True, "run_id": run_id, "bucket": bucket,
            "prefix": prefix, "n_files": len(manifest["files"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Output dir from src.output.geojson")
    args = ap.parse_args()

    run_upload_r2(args.run_dir)


if __name__ == "__main__":
    main()
