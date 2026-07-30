"""Tensor-building: turns rolling mask history + static per-pixel features
into the (B, LOOKBACK, IN_CH, H, W) tensor ConvLSTMUNet expects. Shared by
inference (single AOI, single window) and any future retraining (many AOI,
many windows, gap-tolerant) -- ported from claude_result.md Cell 8.
"""
from __future__ import annotations

import numpy as np
import torch


def build_inference_input(history_frames: list[np.ndarray], static_stack: np.ndarray | None,
                           device: torch.device) -> torch.Tensor:
    """history_frames: LOOKBACK masks, oldest first, each (H, W).
    static_stack: (n_static_ch, H, W) or None (mask-only model, in_ch=1).
    Returns (1, LOOKBACK, IN_CH, H, W) float tensor."""
    mask_stack = np.stack([f[np.newaxis] for f in history_frames])  # (LOOKBACK, 1, H, W)
    if static_stack is not None:
        lookback = len(history_frames)
        static_bcast = np.broadcast_to(static_stack[np.newaxis], (lookback, *static_stack.shape))
        x = np.concatenate([mask_stack, static_bcast], axis=1)  # (LOOKBACK, 1+C, H, W)
    else:
        x = mask_stack
    return torch.from_numpy(x[np.newaxis]).float().to(device)


def build_multistep_tensor(combined_masks: dict, static_features: dict, lookback: int,
                            rollout_steps: int, max_gap_years: float):
    """Training-time windowing with a max-gap tolerance instead of an exact
    fixed-cadence check (Landsat ~1/2yr vs Sentinel ~1/3yr don't share a
    grid). Kept here for reuse by any future retraining script; not used by
    the inference path above. See claude_result.md Cell 8 for the original,
    fully-annotated version this was ported from."""
    import collections

    channel_counts = collections.Counter(v.shape[0] for v in static_features.values())
    n_static_ch = channel_counts.most_common(1)[0][0] if channel_counts else 0

    X_list, y_list, meta = [], [], []
    n_needed = lookback + rollout_steps
    for aoi, frames in combined_masks.items():
        static_stack = static_features.get(aoi)
        use_static = n_static_ch > 0 and static_stack is not None and static_stack.shape[0] == n_static_ch
        if n_static_ch > 0 and not use_static:
            continue
        ts = sorted(frames.keys())
        for i in range(len(ts) - n_needed + 1):
            window_ts = ts[i:i + n_needed]
            gaps = [window_ts[j + 1] - window_ts[j] for j in range(len(window_ts) - 1)]
            if max(gaps) > max_gap_years:
                continue
            in_ts, target_ts = window_ts[:lookback], window_ts[lookback:lookback + rollout_steps]
            mask_stack = np.stack([frames[t][np.newaxis] for t in in_ts])
            if use_static:
                static_bcast = np.broadcast_to(static_stack[np.newaxis], (lookback, *static_stack.shape))
                x_sample = np.concatenate([mask_stack, static_bcast], axis=1)
            else:
                x_sample = mask_stack
            X_list.append(x_sample)
            y_list.append(np.stack([frames[t][np.newaxis] for t in target_ts]))
            meta.append({"aoi": aoi, "input_t": list(in_ts), "target_t": list(target_ts),
                         "target_year": int(target_ts[0])})

    return np.stack(X_list), np.stack(y_list), meta
