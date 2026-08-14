"""Turn a downloaded MNDWI array into a binary water mask using a FIXED,
pre-committed per-AOI threshold (config/thresholds.json) -- no threshold is
ever recomputed at runtime here. Recomputing per run is what caused the
training-data volatility diagnosed during the ConvLSTM training session.

Also owns the rolling LOOKBACK-frame history each AOI needs as ConvLSTM input
for the next scheduled run.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage


def fit_patch(mask_arr: np.ndarray, patch_size: int) -> np.ndarray:
    t = patch_size
    h, w = mask_arr.shape
    if h > t:
        mask_arr = mask_arr[(h - t) // 2:(h - t) // 2 + t, :]
    if w > t:
        mask_arr = mask_arr[:, (w - t) // 2:(w - t) // 2 + t]
    h, w = mask_arr.shape
    if h < t or w < t:
        pad = np.zeros((t, t), dtype=mask_arr.dtype)
        pad[:h, :w] = mask_arr
        mask_arr = pad
    return mask_arr


def load_thresholds(path: str) -> dict[str, float]:
    with open(path) as f:
        data = json.load(f)
    thresholds = data["thresholds"]
    missing = [aoi for aoi, val in thresholds.items() if val is None]
    if missing:
        raise ValueError(
            f"config/thresholds.json has unpopulated (null) thresholds for: {missing}. "
            f"Fill these from a training notebook run before running preprocess in production."
        )
    return thresholds


def apply_threshold(mndwi_arr: np.ndarray, threshold: float, closing_radius: int = 2) -> np.ndarray:
    """Threshold + morphological closing, mirroring the GEE-side
    focalMode/focalMax/focalMin sequence in all_code2.py's _water_mask, done
    client-side since preprocess.py has no GEE dependency by design."""
    water = mndwi_arr > threshold
    water = ndimage.binary_opening(water, structure=np.ones((3, 3)))
    water = ndimage.binary_closing(water, structure=np.ones((closing_radius * 2 + 1,) * 2))
    return water.astype(np.float32)


def history_path(state_dir: str, aoi: str) -> str:
    return os.path.join(state_dir, f"{aoi}_history.npz")


def load_history(state_dir: str, aoi: str) -> list[np.ndarray]:
    path = history_path(state_dir, aoi)
    if not os.path.exists(path):
        return []
    z = np.load(path)
    keys = sorted(z.files, key=lambda k: int(k.split("_")[1]))
    return [z[k] for k in keys]


def update_history(state_dir: str, aoi: str, new_mask: np.ndarray, lookback: int) -> list[np.ndarray]:
    os.makedirs(state_dir, exist_ok=True)
    frames = load_history(state_dir, aoi) + [new_mask]
    frames = frames[-lookback:]
    np.savez_compressed(history_path(state_dir, aoi), **{f"frame_{i}": f for i, f in enumerate(frames)})
    return frames


def run_mask(in_dir: str, state_dir: str = "data/state", thresholds_path: str = "config/thresholds.json",
             patch_size: int = 256, lookback: int = 3) -> list[str]:
    """Core preprocess logic, callable directly (Prefect task wraps this) as
    well as via the CLI `main()` below. Returns the list of AOI names whose
    rolling history got updated this run."""
    thresholds = load_thresholds(thresholds_path)
    with open(os.path.join(in_dir, "run_meta.json")) as f:
        run_meta = json.load(f)

    updated_aois = []
    for entry in run_meta["aoi"]:
        name = entry["name"]
        if name not in thresholds:
            print(f"[{name}] SKIP -- no threshold configured")
            continue
        try:
            mndwi = np.load(os.path.join(in_dir, f"{name}_mndwi.npy"))
            water = apply_threshold(mndwi, thresholds[name])
            water = fit_patch(water, patch_size)
            frames = update_history(state_dir, name, water, lookback)
            updated_aois.append(name)
            print(f"[{name}] mask OK, history now {len(frames)}/{lookback} frame(s)")
        except Exception as exc:  # noqa: BLE001 -- one bad AOI shouldn't sink the whole run
            print(f"[{name}] SKIP -- unexpected error: {exc}")
            continue
    return updated_aois


def main() -> None:
    """Entry point: python -m src.preprocessing.mask --in-dir <fetch output> --state-dir data/state"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Output dir from src.gee.fetch")
    ap.add_argument("--state-dir", default="data/state")
    ap.add_argument("--thresholds", default="config/thresholds.json")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--lookback", type=int, default=3)
    args = ap.parse_args()

    run_mask(args.in_dir, args.state_dir, args.thresholds, args.patch_size, args.lookback)


if __name__ == "__main__":
    main()
