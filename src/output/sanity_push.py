#!/usr/bin/env python3
"""Push each run's shoreline.geojson + metrics.json into Sanity via its HTTP
mutate API (no @sanity/client / Node dependency needed -- this is a plain
REST call, works fine from Python).

Requires env vars (GitHub Secrets, to be added later by the user):
  SANITY_PROJECT_ID   -- not secret, but env-configured for flexibility
  SANITY_DATASET      -- e.g. "production"
  SANITY_API_TOKEN     -- secret: a token with write access to the dataset
  SANITY_API_VERSION  -- optional, defaults to "2024-01-01"

Document shape pushed (one per AOI per run, id is deterministic so re-runs
UPSERT rather than duplicate):
  _type: "shorelineRun"
  _id: "shorelineRun-{aoi}-{run_utc date}"
  aoi, runUtc, meanEprMPerYr, nTransects, classificationCounts,
  trackAAvailable, geojson (the AOI's features from shoreline.geojson)

Usage: python -m src.output.sanity_push --run-dir data/interim/run_<date>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

SANITY_API_VERSION = os.environ.get("SANITY_API_VERSION", "2024-01-01")


def build_documents(geojson: dict, metrics: dict) -> list[dict]:
    run_date = metrics["run_utc"][:10]
    features_by_aoi: dict[str, list[dict]] = {}
    for feat in geojson["features"]:
        features_by_aoi.setdefault(feat["properties"]["aoi"], []).append(feat)

    docs = []
    for aoi, aoi_metrics in metrics["aoi"].items():
        docs.append({
            "_type": "shorelineRun",
            "_id": f"shorelineRun-{aoi}-{run_date}",
            "aoi": aoi,
            "runUtc": metrics["run_utc"],
            "meanEprMPerYr": aoi_metrics["mean_epr_m_per_yr"],
            "nTransects": aoi_metrics["n_transects"],
            "classificationCounts": aoi_metrics["classification_counts"],
            "trackAAvailable": aoi_metrics["track_a_available"],
            "geojson": {"type": "FeatureCollection", "features": features_by_aoi.get(aoi, [])},
        })
    return docs


def push_to_sanity(documents: list[dict]) -> None:
    project_id = os.environ["SANITY_PROJECT_ID"]
    dataset = os.environ["SANITY_DATASET"]
    token = os.environ["SANITY_API_TOKEN"]

    url = f"https://{project_id}.api.sanity.io/v{SANITY_API_VERSION}/data/mutate/{dataset}"
    mutations = [{"createOrReplace": doc} for doc in documents]
    resp = requests.post(url, json={"mutations": mutations},
                          headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    print(f"Pushed {len(documents)} document(s) to Sanity ({project_id}/{dataset})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    required_env = ["SANITY_PROJECT_ID", "SANITY_DATASET", "SANITY_API_TOKEN"]
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        print(f"SKIP Sanity push -- missing env var(s): {missing}. "
              f"Set these as GitHub Secrets once Sanity is set up.", file=sys.stderr)
        return

    with open(os.path.join(args.run_dir, "shoreline.geojson")) as f:
        geojson = json.load(f)
    with open(os.path.join(args.run_dir, "metrics.json")) as f:
        metrics = json.load(f)

    documents = build_documents(geojson, metrics)
    push_to_sanity(documents)


if __name__ == "__main__":
    main()
