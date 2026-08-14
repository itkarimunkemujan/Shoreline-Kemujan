#!/usr/bin/env python3
"""Promote the latest valid run from the R2 warehouse to Sanity `production`.

Reads the run_manifest.json files under `runs/` in the R2 bucket (via
DuckDB/s3fs), selects the newest run that:
  - has a complete manifest (all expected files + parquet), and
  - has a non-null metrics summary (aoiCount > 0), and
  - is at least MIN_AGE_HOURS old (default 24h, so a just-tested run from a
    manual recalculate dispatch can't be pushed the same day), and
  - is newer than the last promoted run id recorded in
    `state/last_promoted.json` in the same bucket.

Then downloads the selected run's files into a temp dir and reuses
`src.output.sanity_push.run_sanity_push(run_dir, dataset="production")` --
identical push logic as the staging push, just a different dataset.

Idempotent: `state/last_promoted.json` is updated to the promoted run id, so
the same run is never promoted twice. Run with `--dry-run` to print the
selection without pushing anything.

Requires env vars (GitHub Secrets):
  R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_ACCESS_KEY_SECRET / R2_BUCKET
  SANITY_PROJECT_ID / SANITY_DATASET / SANITY_API_TOKEN   (SANITY_DATASET is
  overridden to "production" -- set SANITY_DATASET_PRODUCTION in promote.yml)

Usage: python -m src.pipeline_promote [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import s3fs

from src.output.sanity_push import run_sanity_push
from src.output.validate_run import validate_run

REQUIRED_ENV = ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_ACCESS_KEY_SECRET", "R2_BUCKET"]
MIN_AGE_HOURS = 24
STATE_KEY = "state/last_promoted.json"
PRODUCTION_DATASET = "production"


def _r2_fs() -> s3fs.S3FileSystem:
    account_id = os.environ["R2_ACCOUNT_ID"]
    return s3fs.S3FileSystem(
        key=os.environ["R2_ACCESS_KEY_ID"],
        secret=os.environ["R2_ACCESS_KEY_SECRET"],
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        client_kwargs={"region_name": "auto"},
    )


def _list_manifests(fs: s3fs.S3FileSystem, bucket: str) -> list[dict]:
    """Return sorted (by run_utc, newest first) manifests from runs/*/."""
    prefix = f"{bucket}/runs/"
    manifests = []
    for manifest_key in fs.glob(prefix + "*/run_manifest.json"):
        with fs.open(manifest_key) as f:
            manifests.append(json.load(f))
    manifests.sort(key=lambda m: m.get("run_utc", ""), reverse=True)
    return manifests


def _load_state(fs: s3fs.S3FileSystem, bucket: str) -> dict:
    try:
        with fs.open(f"{bucket}/{STATE_KEY}") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_state(fs: s3fs.S3FileSystem, bucket: str, state: dict) -> None:
    with fs.open(f"{bucket}/{STATE_KEY}", "wb") as f:
        f.write(json.dumps(state, indent=2).encode("utf-8"))


def _age_hours(run_utc: str) -> float:
    try:
        dt = datetime.fromisoformat(run_utc)
    except (TypeError, ValueError):
        return float("inf")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def _validate_manifest(manifest: dict) -> tuple[bool, str]:
    files = manifest.get("files", {})
    required = ["shoreline_current.geojson", "shoreline_predicted.geojson",
                "transects.geojson", "metrics.json", "forecast_docs.json"]
    missing = [name for name in required if name not in files]
    if missing:
        return False, f"missing files: {missing}"
    summary = manifest.get("summary") or {}
    if not summary.get("aoiCount"):
        return False, "empty summary (aoiCount == 0)"
    return True, "ok"


def run_promote(dry_run: bool = False) -> dict:
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        print(f"SKIP promote -- missing env var(s): {missing}.", file=sys.stderr)
        return {"promoted": False, "reason": f"missing env vars: {missing}"}

    fs = _r2_fs()
    bucket = os.environ["R2_BUCKET"]
    manifests = _list_manifests(fs, bucket)
    if not manifests:
        print("No runs found in R2 warehouse -- nothing to promote.")
        return {"promoted": False, "reason": "no runs in warehouse"}

    state = _load_state(fs, bucket)
    last_promoted = state.get("last_promoted_run_id")

    selected = None
    for manifest in manifests:
        if manifest.get("run_id") == last_promoted:
            break  # manifests are newest-first; everything below is older
        ok, why = _validate_manifest(manifest)
        if not ok:
            print(f"  skip {manifest.get('run_id')}: {why}")
            continue
        age = _age_hours(manifest.get("run_utc", ""))
        if age < MIN_AGE_HOURS:
            print(f"  skip {manifest.get('run_id')}: too fresh ({age:.1f}h < {MIN_AGE_HOURS}h)")
            continue
        selected = manifest
        break

    if selected is None:
        print("No eligible run to promote.")
        return {"promoted": False, "reason": "no eligible run"}

    run_id = selected["run_id"]
    print(f"Selected run: {run_id} (run_utc={selected.get('run_utc')}, "
          f"aoiCount={selected.get('summary', {}).get('aoiCount')})")
    if dry_run:
        print("--dry-run: no push performed.")
        return {"promoted": False, "dry_run": True, "run_id": run_id}

    # download the selected run into a temp dir, then run the output-contract
    # validation BEFORE pushing to production. Violations abort the promote.
    with tempfile.TemporaryDirectory() as tmp:
        for name in ["shoreline_current.geojson", "shoreline_predicted.geojson",
                     "transects.geojson", "metrics.json", "forecast_docs.json"]:
            key = selected["files"].get(name, {}).get("key")
            if not key:
                continue
            with fs.open(f"{bucket}/{key}") as src, open(os.path.join(tmp, name), "wb") as dst:
                dst.write(src.read())

        violations = validate_run(tmp)
        if violations:
            print(f"PROMOTE ABORTED: run {run_id} failed validation ({len(violations)} issue(s)):")
            for msg in violations:
                print(f"  - {msg}")
            return {"promoted": False, "run_id": run_id, "reason": "validation failed", "violations": violations}

        push = run_sanity_push(tmp, dataset=PRODUCTION_DATASET)

    if push.get("pushed"):
        state["last_promoted_run_id"] = run_id
        state["promoted_at"] = datetime.now(timezone.utc).isoformat()
        _save_state(fs, bucket, state)
        print(f"Promoted {run_id} -> Sanity {PRODUCTION_DATASET}. "
              f"last_promoted updated in s3://{bucket}/{STATE_KEY}")
    return {"promoted": push.get("pushed", False), "run_id": run_id, "push": push}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selected run without pushing to Sanity")
    args = ap.parse_args()

    run_promote(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
