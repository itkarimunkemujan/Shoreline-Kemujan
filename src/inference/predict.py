#!/usr/bin/env python3
"""Track A entry point: load the committed checkpoint, run one next-step
(and optionally a short rollout) prediction per AOI from the rolling mask
history preprocess.py maintains. CPU-only -- Actions runners have no GPU,
and this is a single forward pass, not training.

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="data/state")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default="models/checkpoint.pth")
    ap.add_argument("--meta", default="models/model_meta.json")
    ap.add_argument("--static-features", default="models/static_features.npz")
    ap.add_argument("--rollout-steps", type=int, default=0, help="0 = next-step only")
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)
    in_ch, lookback = meta["IN_CH"], meta["LOOKBACK"]

    device = torch.device("cpu")
    model = ConvLSTMUNet(in_ch=in_ch, base_ch=16, mc_dropout_p=0.0).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    static_features = {}
    if os.path.exists(args.static_features):
        z = np.load(args.static_features)
        static_features = {aoi: z[aoi] for aoi in z.files}

    os.makedirs(args.out_dir, exist_ok=True)
    for aoi_dir in sorted(f for f in os.listdir(args.state_dir) if f.endswith("_history.npz")):
        aoi = aoi_dir.replace("_history.npz", "")
        history = load_history(args.state_dir, aoi)
        if len(history) < lookback:
            print(f"[{aoi}] SKIP -- only {len(history)}/{lookback} frame(s) of history so far")
            continue
        static_stack = static_features.get(aoi)
        x = build_inference_input(history[-lookback:], static_stack, device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).cpu().numpy()[0, 0]
        pred = (prob > 0.5).astype(np.float32)
        np.save(os.path.join(args.out_dir, f"{aoi}_pred_mask.npy"), pred)
        np.save(os.path.join(args.out_dir, f"{aoi}_pred_prob.npy"), prob)

        if args.rollout_steps > 0:
            future = rollout_forecast(model, history[-lookback:], static_stack, args.rollout_steps, device)
            np.savez_compressed(os.path.join(args.out_dir, f"{aoi}_rollout.npz"),
                                 **{f"step_{i}": f for i, f in enumerate(future)})
        print(f"[{aoi}] prediction OK")


if __name__ == "__main__":
    main()
