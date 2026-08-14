#!/usr/bin/env python3
"""Track A entry point: load the committed checkpoint, run one next-step
(and optionally a short rollout) prediction per AOI from the rolling mask
history preprocess.py maintains. CPU-only -- Actions runners have no GPU,
and this is a single forward pass, not training.

Model shape (IN_CH/LOOKBACK/BASE_CH) is read from `model_meta.json`, shipped
alongside the checkpoint in the same GitHub Release -- NOT hardcoded. This is
the promotion contract between the notebook (free to experiment with
different architectures/channel counts any time -- see
notebooks/train_256_final.ipynb, and the "2aug" multichannel design that was
tried and abandoned earlier) and this production pipeline (which must stay
correct across whatever the notebook lands on next, without needing a code
edit here every time). If `model_meta.json` is missing (e.g. an older
checkpoint predating this convention), falls back to IN_CH=1/LOOKBACK=3/
BASE_CH=16 -- the current production model's actual shape -- rather than
failing outright.

Usage: python -m src.inference.predict --state-dir data/state --out-dir data/interim/run_<date>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.model.convlstm import ConvLSTMUNet
from src.model.dataset import build_inference_input
from src.preprocessing.mask import load_history

DEFAULT_MODEL_META = {"IN_CH": 1, "LOOKBACK": 3, "BASE_CH": 16}


def load_model_meta(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        print(f"model_meta.json not found at {meta_path} -- using defaults {DEFAULT_MODEL_META}")
        return dict(DEFAULT_MODEL_META)
    with open(meta_path) as f:
        meta = json.load(f)
    return {**DEFAULT_MODEL_META, **meta}


def rollout_forecast(model: ConvLSTMUNet, seed_frames: list[np.ndarray],
                      static_stack: np.ndarray | None, n_steps: int, device: torch.device) -> list[np.ndarray]:
    window = list(seed_frames)
    predictions = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = build_inference_input(window, static_stack, device)
            prob = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
            pred = (prob > 0.5).astype(np.float32)
            predictions.append(pred)
            window = window[1:] + [pred]
    return predictions


def run_predict(state_dir: str, out_dir: str, checkpoint: str = "models/checkpoint.pth",
                 meta_path: str = "models/model_meta.json",
                 static_features_path: str = "models/static_features.npz",
                 rollout_steps: int = 0) -> list[str]:
    """Core Track A logic, callable directly (Prefect task wraps this) as well
    as via the CLI `main()` below. Returns the list of AOI names that got a
    fresh prediction this run (used by the pipeline to know what's available
    downstream without re-scanning the filesystem)."""
    meta = load_model_meta(meta_path)
    in_ch, lookback, base_ch = meta["IN_CH"], meta["LOOKBACK"], meta["BASE_CH"]

    device = torch.device("cpu")
    model = ConvLSTMUNet(in_ch=in_ch, base_ch=base_ch, mc_dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    static_features = {}
    if os.path.exists(static_features_path):
        z = np.load(static_features_path)
        static_features = {aoi: z[aoi] for aoi in z.files}

    os.makedirs(out_dir, exist_ok=True)
    predicted_aois = []
    for aoi_dir in sorted(f for f in os.listdir(state_dir) if f.endswith("_history.npz")):
        aoi = aoi_dir.replace("_history.npz", "")
        history = load_history(state_dir, aoi)
        if len(history) < lookback:
            print(f"[{aoi}] SKIP -- only {len(history)}/{lookback} frame(s) of history so far")
            continue
        try:
            static_stack = static_features.get(aoi)
            x = build_inference_input(history[-lookback:], static_stack, device)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
            pred = (prob > 0.5).astype(np.float32)
            np.save(os.path.join(out_dir, f"{aoi}_pred_mask.npy"), pred)
            np.save(os.path.join(out_dir, f"{aoi}_pred_prob.npy"), prob)

            if rollout_steps > 0:
                future = rollout_forecast(model, history[-lookback:], static_stack, rollout_steps, device)
                np.savez_compressed(os.path.join(out_dir, f"{aoi}_rollout.npz"),
                                     **{f"step_{i}": f for i, f in enumerate(future)})
            predicted_aois.append(aoi)
            print(f"[{aoi}] prediction OK")
        except Exception as exc:  # noqa: BLE001 -- one bad AOI shouldn't sink the whole run
            print(f"[{aoi}] SKIP -- unexpected error: {exc}")
            continue
    return predicted_aois


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="data/state")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default="models/checkpoint.pth")
    ap.add_argument("--meta", default="models/model_meta.json")
    ap.add_argument("--static-features", default="models/static_features.npz",
                     help="Optional; no-op unless the checkpoint was trained with static "
                          "per-pixel channels (the production checkpoint isn't).")
    ap.add_argument("--rollout-steps", type=int, default=0, help="0 = next-step only")
    args = ap.parse_args()

    run_predict(args.state_dir, args.out_dir, args.checkpoint, args.meta,
                args.static_features, args.rollout_steps)


if __name__ == "__main__":
    main()
