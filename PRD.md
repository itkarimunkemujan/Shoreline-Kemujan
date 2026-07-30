# PRD — Combine Landsat + Sentinel-2 training data, scheduled-sampling ConvLSTM-UNet retrain

Status: implemented in `notebooks/final_experiment.ipynb` (2026-07-29). Written for
another agent/human to pick up context quickly — not a durable architecture doc, update
or delete once superseded.

## Problem

Data collection was previously done separately for two sensors, each producing its own
offline bundle on Google Drive:

- Sentinel-2: `MyDrive/Data_experiment_shoreline/offline_256_adaptive.zip` — 8 AOI,
  seasons `S1/S2/S3` (3/year), 2018–2025, 256×256 px @ 10 m.
- Landsat: `MyDrive/shoreline_kemujan/landsat_offline.zip` — 8 AOI, seasons `L1/L2`
  (2/year), 2008–2018, 256×256 px @ 10 m (Landsat native 30 m, bilinear-resampled to
  the Sentinel grid at mask-generation time — see `ShorelineProcessorLandsat` in
  `notebooks/all_code2.py`).

Neither bundle alone covers the full 2008–2025 history, and the existing training loop
in `notebooks/all_code2.py` only ever consumed one bundle at a time, with a `seq_index`
windowing scheme hard-coded to 3 evenly-spaced Sentinel seasons/year. It also trains
with pure teacher forcing and next-step-only loss, and only ever saves final-epoch
weights — so double-descent behavior can't be inspected after training, and the
autoregressive rollout used for forecasting (`rollout_forecast` in `all_code2.py`) is
never actually trained end-to-end, only evaluated that way (exposure bias).

## Goal

Extend the model's training history back to 2008 by merging both bundles into one
chronological per-AOI mask sequence, then retrain ConvLSTM-UNet with:

1. Scheduled sampling — occasionally feed the model its own prior prediction instead
   of ground truth during training, so it learns to be stable under the same
   autoregressive conditions used at inference/rollout time.
2. Multi-step rollout loss — backprop through several future steps per training
   sample, not just the next one.
3. Full checkpoint history (periodic + best-val + last), ~500 epochs, so double
   descent can be studied post-hoc instead of only ever having whatever the last
   epoch happened to land on.

All of this lives in `notebooks/final_experiment.ipynb`, built to run standalone in
Colab (mounts Drive, authenticates, unpacks, trains, evaluates, visualizes).

## Design decisions (non-obvious, flagged for whoever revisits this)

- **Unifying `L1/L2` and `S1/S2/S3` seasons**: converted to a continuous time value
  `t = year + month/12` using fixed season→month midpoints (`L1→04`, `L2→10`,
  `S1→03`, `S2→07`, `S3→11`), rather than trying to force a shared season grid. Window
  contiguity for `build_tensor` was changed from an exact index-arithmetic check to a
  max-gap tolerance on `t`, since Landsat/Sentinel cadence differs and real data gaps
  exist in both.
- **2018 overlap**: both sensors have data in 2018. Sentinel is kept (native 10 m,
  no resampling) and the Landsat frame for that slot is dropped; this is logged
  per-AOI so it's auditable, not silent.
- **Scheduled sampling ramp**: linear/inverse-sigmoid from `0` (pure teacher forcing)
  up to a capped max (default `0.5` by ~epoch 250) — standard practice default, not a
  value the user specified; tunable at the top of the training cell.
- **Checkpoint cadence**: every 10 epochs + a running best-by-val-Dice + final —
  trades Drive storage for double-descent visibility. Adjustable if Drive space
  becomes a concern (~50 periodic checkpoints over 500 epochs, per run).
- **Visualization reuse**: rollout/transect/EPR/LRR plotting (`full_analysis_plot`,
  `yearly_grid_plot_extended`, `diagnose_rollout_collapse`, `compute_lrr_and_compare`,
  contour/transect helpers) is reused verbatim from `notebooks/all_code2.py`, only
  swapping the season-index helper (`seq_index`/`SEASON_ORDER`) for the new
  continuous-time ordering so contours still sort correctly across the 2018
  Landsat→Sentinel boundary. The ConvLSTM-UNet architecture itself is also reused
  unchanged — it was already debugged in `all_code2.py` (see its skip-connection
  fix comments); only the loss/training loop changed.

## Out of scope

- Re-running Earth Engine data collection — both bundles already exist on Drive and
  are treated as the source of truth.
- Changing the model architecture (`ConvLSTMUNet`) — reused as-is.
- Wiring this into the `shoreline-kemujan-monitoring` inference pipeline described in
  `README.md` — that repo's Track A inference expects a single trained checkpoint;
  this notebook is where that checkpoint gets produced, promotion to the monitoring
  repo remains a manual step per `README.md`'s existing "Training tetap manual" note.
