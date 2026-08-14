#!/usr/bin/env python3
"""Push one run's output (from `src.output.geojson`) into Sanity: uploads the
3 GeoJSON files as file assets and `createOrReplace`s the `abrasionDataset`
singleton (main WebGIS overlay), then batch `createOrReplace`s the
`shorelineForecast` per-AOI-per-year documents (AOI marker + detail drawer).

Matches `tourism-kemujan/studio/schemaTypes/{singletons/abrasionDataset.ts,
documents/shorelineForecast.ts}` exactly -- this replaces an earlier version
of this file that pushed a `shorelineRun` document type that doesn't exist
in the frontend schema at all (dead code, never actually consumed by
anything). The authoritative source for this push shape is
`notebooks/train_256_final.ipynb`'s "CELL 11" (see also
`shoreline-kemujan/claude_result12.md`/`claude_result14.md`), ported here.

Requires env vars (GitHub Secrets):
  SANITY_PROJECT_ID   -- not secret, but env-configured for flexibility
  SANITY_DATASET      -- e.g. "production" or "development"
  SANITY_API_TOKEN    -- secret: a token with WRITE access to the dataset
  SANITY_API_VERSION  -- optional, defaults to "2024-01-01"

Usage: python -m src.output.sanity_push --run-dir data/interim/run_<date>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

SANITY_API_VERSION = os.environ.get("SANITY_API_VERSION", "2024-01-01")
ABRASION_SINGLETON_ID = "singleton-abrasionDataset"  # must match studio/schemaTypes/_singletonIds.ts


def upload_geojson_asset(upload_url: str, token: str, feature_collection: dict, filename: str) -> str | None:
    """Returns the uploaded asset's `_id`, or None if the collection is empty
    (uploading an empty FeatureCollection would just leave a useless asset)."""
    if not feature_collection.get("features"):
        return None
    content = json.dumps(feature_collection).encode("utf-8")
    resp = requests.post(
        upload_url, params={"filename": filename},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=content, timeout=120,
    )
    resp.raise_for_status()
    asset_id = resp.json()["document"]["_id"]
    print(f"Uploaded {filename} ({len(content) / 1024:.1f} KB) -> {asset_id}")
    return asset_id


def push_batch(mutate_url: str, token: str, mutations: list[dict], label: str = "") -> dict:
    resp = requests.post(mutate_url, json={"mutations": mutations},
                          headers={"Authorization": f"Bearer {token}"}, timeout=60)
    resp.raise_for_status()
    print(f"Sanity push OK: {len(mutations)} docs  {label}")
    return resp.json()


def build_abrasion_doc(
    metrics: dict,
    asset_current: str | None,
    asset_predicted: str | None,
    asset_transects: str | None,
    run_utc: str,
    pipeline_version: str,
) -> dict | None:
    if asset_current is None:
        # shorelineCurrent is `Rule.required()` in the schema -- no point
        # pushing a doc Sanity would reject anyway.
        return None
    doc = {
        "_type": "abrasionDataset",
        "_id": ABRASION_SINGLETON_ID,
        "shorelineCurrent": {"_type": "file", "asset": {"_type": "reference", "_ref": asset_current}},
        "metrics": {
            "aoiCount": metrics["aoiCount"],
            "meanNsm": metrics["meanNsm"],
            "meanEpr": metrics["meanEpr"],
            "meanLrr": metrics["meanLrr"],
        },
        "dataUpdatedAt": run_utc,
        "pipelineVersion": pipeline_version,
        "isPublished": True,
    }
    if asset_predicted:
        doc["shorelinePredicted"] = {"_type": "file", "asset": {"_type": "reference", "_ref": asset_predicted}}
    if asset_transects:
        doc["transects"] = {"_type": "file", "asset": {"_type": "reference", "_ref": asset_transects}}
    return doc


def build_forecast_mutations(forecast_docs: list[dict], pipeline_version: str) -> list[dict]:
    mutations = []
    for doc in forecast_docs:
        aoi, year = doc["aoi"], doc["year"]
        mutations.append({"createOrReplace": {
            "_type": "shorelineForecast",
            "_id": f"shorelineForecast-{aoi}-{year}-{pipeline_version}",
            **doc,
            "modelSource": pipeline_version,
            "mcUncertaintyMean": None,
        }})
    return mutations


def run_sanity_push(run_dir: str, dataset: str | None = None) -> dict:
    """Core push logic, callable directly (Prefect task wraps this) as well
    as via the CLI `main()` below. Returns a small summary dict (doc counts)
    rather than None, so a Prefect task can log/branch on the result.

    `dataset` optionally overrides `SANITY_DATASET` -- this lets the promote
    step (src.pipeline_promote.py) reuse the exact same push logic against
    the production dataset while recalculate.yml points at the staging one."""
    required_env = ["SANITY_PROJECT_ID", "SANITY_DATASET", "SANITY_API_TOKEN"]
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        print(f"SKIP Sanity push -- missing env var(s): {missing}.", file=sys.stderr)
        return {"pushed": False, "reason": f"missing env vars: {missing}"}

    project_id = os.environ["SANITY_PROJECT_ID"]
    dataset = dataset or os.environ["SANITY_DATASET"]
    token = os.environ["SANITY_API_TOKEN"]
    mutate_url = f"https://{project_id}.api.sanity.io/v{SANITY_API_VERSION}/data/mutate/{dataset}"
    upload_url = f"https://{project_id}.api.sanity.io/v{SANITY_API_VERSION}/assets/files/{dataset}"

    with open(os.path.join(run_dir, "shoreline_current.geojson")) as f:
        current_fc = json.load(f)
    with open(os.path.join(run_dir, "shoreline_predicted.geojson")) as f:
        predicted_fc = json.load(f)
    with open(os.path.join(run_dir, "transects.geojson")) as f:
        transects_fc = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics_payload = json.load(f)
    with open(os.path.join(run_dir, "forecast_docs.json")) as f:
        forecast_docs = json.load(f)

    if not forecast_docs:
        print("Nothing to push -- forecast_docs.json is empty (check src.output.geojson's run).")
        return {"pushed": False, "reason": "forecast_docs.json empty"}

    run_utc = metrics_payload.get("run_utc") or datetime.now(tz=timezone.utc).isoformat()
    # GITHUB_SHA is set automatically in Actions; falls back to the run
    # timestamp locally so this is always deterministic-ish, never blank.
    pipeline_version = os.environ.get("GITHUB_SHA", run_utc)[:12]

    run_tag = f"run{run_utc[:10].replace('-', '')}"
    asset_current = upload_geojson_asset(upload_url, token, current_fc, f"shoreline_current_{run_tag}.geojson")
    asset_predicted = upload_geojson_asset(upload_url, token, predicted_fc, f"shoreline_predicted_{run_tag}.geojson")
    asset_transects = upload_geojson_asset(upload_url, token, transects_fc, f"transects_{run_tag}.geojson")

    abrasion_doc = build_abrasion_doc(
        metrics_payload, asset_current, asset_predicted, asset_transects, run_utc, pipeline_version)
    if abrasion_doc:
        push_batch(mutate_url, token, [{"createOrReplace": abrasion_doc}], label="abrasionDataset (singleton)")
    else:
        print("SKIP abrasionDataset push -- no current-shoreline features this run.", file=sys.stderr)

    forecast_mutations = build_forecast_mutations(forecast_docs, pipeline_version)
    BATCH = 100
    for start in range(0, len(forecast_mutations), BATCH):
        push_batch(mutate_url, token, forecast_mutations[start:start + BATCH],
                   label=f"shorelineForecast batch {start // BATCH + 1}")

    print(f"Done. abrasionDataset: {'1 doc' if abrasion_doc else 'skipped'}. "
          f"shorelineForecast: {len(forecast_mutations)} docs -> {project_id}/{dataset}")
    return {
        "pushed": True,
        "abrasion_doc_pushed": bool(abrasion_doc),
        "forecast_docs_pushed": len(forecast_mutations),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Output dir from src.output.geojson")
    args = ap.parse_args()

    run_sanity_push(args.run_dir)


if __name__ == "__main__":
    main()
