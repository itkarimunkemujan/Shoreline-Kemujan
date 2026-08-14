# Project Impact — Shoreline-Kemujan Monitoring

> **Honest framing.** This repo is a production **batch inference pipeline**, not a full
> ML/Data platform. It proves the *core production mechanics* end-to-end (automation,
> data contract validation, promotion gate, security) for one real use case, and the
> roadmap below is the explicit, prioritized gap between "batch inference pipeline" and
> a "full ML platform".

## 1. Overview

Production batch inference pipeline: **Sentinel-2 (Google Earth Engine) → ConvLSTM-UNet →
Sanity CMS → WebGIS**, running automatically every 4 months on GitHub Actions. Monitors
coastal erosion along Karimunjawa's Kemujan shoreline (9 monitoring points), replacing a
manual field-survey cadence, feeding the live Pokdarwis abrasion dashboard (repo
`tourism-kemujan`). Scientific basis: Suryanti et al. (2025), SWOT/IFAS shoreline
management Karimunjawa. Built as a KKN-PPM UGM program with Desa Kemujan.

Deliberately **lean**: one automated run path, minimal infrastructure, no always-on
servers.

## 2. What this project actually is

- **A single automated batch run** — `fetch_composite → build_mask → track_a_predict →
  build_payload → push_to_sanity → upload_to_r2`, orchestrated by an **ephemeral Prefect
  DAG** running inside one GitHub Actions job (no Prefect server/worker anywhere).
- **Warehouse** — Cloudflare R2 raw zone (immutable): GeoJSON + DuckDB Parquet +
  `run_manifest.json` per run under `runs/<run_id>/`, queryable via DuckDB.
- **Data contract validation** (`src/output/validate_run.py`) — this is the genuine DE
  strength: validates the 5 structured output files against a declared schema contract,
  physical bounds (EPR ±50 m/yr, NSM ≤ 3700 m, nTransects ≤ 40), null/completeness gates,
  referential integrity to `config/aoi_points.geojson`, and freshness (`run_utc`).
  Adapted from the classic "10 integration tests" playbook; runs in CI + as the gate
  before every production push.
- **Promotion gate** — a run only reaches Sanity `production` after: full manifest,
  non-null metrics, ≥24h maturity, newer than `last_promoted`, and passing the output
  contract. Idempotent — the same run is never promoted twice.

## 3. Real impact

| Dimension | What actually delivers value |
|---|---|
| Research → production | Scientific research becomes a live dashboard, not a static paper |
| Automation | Replaces manual field-survey cadence with a scheduled satellite pipeline |
| Cost | ~350KB CPU-friendly checkpoint runs in CI — **no GPU needed** for serving |
| Security | Least-privilege GEE service account, R2 token scoped to one bucket, all secrets via GitHub |
| Data quality | Contract validation blocks corrupt output before it reaches the dashboard |

## 4. Scope & Roadmap (explicit honesty)

Not built today — deliberately listed so the gap to a "full platform" is known and
prioritized:

| Missing today | Why it matters | Suggested tool | Effort |
|---|---|---|---|
| Feature store | No reusable/versioned features for retraining or online serving | Feast | Medium |
| Incremental / crawler ingest | Current ingest is a full re-fetch per run; no CDC or backfill | Custom scheduler / Airflow | High |
| Infrastructure as Code | R2, secrets, and Actions setup are manual, not reproducible | Terraform / Pulumi | Medium |
| Model registry | Only GitHub Release tags; no experiment tracking or eval harness | MLflow / W&B | Medium |
| Drift monitoring | No detection of input/output distribution drift between runs | Evidently / custom | Medium |
| Schema registry | Contract is code-only, no schema-evolution tooling | Great Expectations / Elementary | Low |

## 5. Verdict

This repo proves the mechanics that junior portfolios rarely show: a **real automated
pipeline with contract validation, a promotion gate, idempotency, and security**, wired
end-to-end from satellite data to a production web dashboard. The step to a "full ML
platform" is not magic — it is the prioritized roadmap in Section 4.