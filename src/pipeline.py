#!/usr/bin/env python3
"""Prefect flow wrapping the shoreline-kemujan pipeline's 5 stages (fetch ->
mask -> Track A inference -> postprocess/GeoJSON -> Sanity push) as a real
task DAG, run EPHEMERALLY inside a single GitHub Actions job -- there is no
Prefect server/worker/agent kept running anywhere. `prefect.io` reports back
to Prefect Cloud (via PREFECT_API_KEY/PREFECT_API_URL env vars, set as GitHub
Secrets) purely as an HTTP client call per task/flow event; nothing about
this script requires infrastructure beyond the GH Actions runner itself.

This intentionally does NOT reimplement any pipeline logic -- every task
below is a thin wrapper calling the `run_*()` function already defined in
its respective module (see src/gee/fetch.py, src/preprocessing/mask.py,
src/inference/predict.py, src/output/geojson.py, src/output/sanity_push.py).
Those modules remain independently runnable via their own CLI `main()` for
manual/local debugging -- Prefect is an orchestration layer on top, not a
replacement for them.

Usage: python -m src.pipeline --run-dir data/interim/run_<date>
"""
from __future__ import annotations

import argparse
import os

from prefect import flow, get_run_logger, task

from src.gee.fetch import run_fetch
from src.preprocessing.mask import run_mask
from src.inference.predict import run_predict
from src.output.geojson import run_geojson
from src.output.sanity_push import run_sanity_push
from src.output.upload_r2 import run_upload_r2


@task(name="fetch_composite", retries=2, retry_delay_seconds=30)
def fetch_composite(aoi_config: str, run_dir: str) -> dict:
    return run_fetch(aoi_config, run_dir)


@task(name="build_mask")
def build_mask(run_dir: str, state_dir: str, thresholds_path: str) -> list[str]:
    return run_mask(run_dir, state_dir, thresholds_path)


@task(name="track_a_predict")
def track_a_predict(state_dir: str, run_dir: str, checkpoint: str, meta_path: str) -> list[str]:
    return run_predict(state_dir, run_dir, checkpoint, meta_path)


@task(name="build_payload")
def build_payload(state_dir: str, run_dir: str, aoi_config: str, lrr_baseline_path: str) -> dict[str, str]:
    return run_geojson(state_dir, run_dir, pred_dir=run_dir, aoi_config=aoi_config,
                        lrr_baseline_path=lrr_baseline_path)


@task(name="push_to_sanity", retries=1, retry_delay_seconds=15)
def push_to_sanity(run_dir: str) -> dict:
    return run_sanity_push(run_dir)


@task(name="upload_to_r2")
def upload_to_r2(run_dir: str) -> dict:
    return run_upload_r2(run_dir)


@flow(name="shoreline_pipeline", log_prints=True)
def shoreline_pipeline(
    run_dir: str,
    state_dir: str = "data/state",
    aoi_config: str = "config/aoi_points.geojson",
    thresholds_path: str = "config/thresholds.json",
    checkpoint: str = "models/checkpoint.pth",
    model_meta_path: str = "models/model_meta.json",
    lrr_baseline_path: str = "data/state/lrr_baseline.json",
) -> dict:
    logger = get_run_logger()
    os.makedirs(run_dir, exist_ok=True)

    fetch_composite(aoi_config, run_dir)
    updated_aois = build_mask(run_dir, state_dir, thresholds_path)
    logger.info(f"Mask stage updated {len(updated_aois)} AOI(s): {updated_aois}")

    predicted_aois = track_a_predict(state_dir, run_dir, checkpoint, model_meta_path)
    logger.info(f"Track A predicted {len(predicted_aois)} AOI(s): {predicted_aois}")

    output_paths = build_payload(state_dir, run_dir, aoi_config, lrr_baseline_path)
    push_result = push_to_sanity(run_dir)
    upload_result = upload_to_r2(run_dir)

    logger.info(f"Pipeline done. Sanity push result: {push_result} | R2 upload result: {upload_result}")
    return {"output_paths": output_paths, "push_result": push_result, "upload_result": upload_result}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--state-dir", default="data/state")
    ap.add_argument("--aoi-config", default="config/aoi_points.geojson")
    ap.add_argument("--thresholds", default="config/thresholds.json")
    ap.add_argument("--checkpoint", default="models/checkpoint.pth")
    ap.add_argument("--model-meta", default="models/model_meta.json")
    ap.add_argument("--lrr-baseline", default="data/state/lrr_baseline.json")
    args = ap.parse_args()

    shoreline_pipeline(
        run_dir=args.run_dir,
        state_dir=args.state_dir,
        aoi_config=args.aoi_config,
        thresholds_path=args.thresholds,
        checkpoint=args.checkpoint,
        model_meta_path=args.model_meta,
        lrr_baseline_path=args.lrr_baseline,
    )


if __name__ == "__main__":
    main()
