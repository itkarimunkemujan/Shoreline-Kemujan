"""Track B: independent, model-free shoreline change baseline (README's
"guaranteed floor" -- keeps producing useful output even if Track A's
ConvLSTM checkpoint is stale, missing, or degraded).

DSAS-style transects + linear regression rate (LRR) on the observed mask
history, with an optional 1D Kalman filter to smooth each transect's
position time series before fitting (reduces sensitivity to a single noisy
observation without needing a full second model).
"""
from __future__ import annotations

import numpy as np

from src.analysis.metrics import classify_epr
from src.analysis.spatial import extract_main_contour, transect_intersect
from src.analysis.transect import build_baseline_transects, linear_regression_rate


def kalman_smooth_1d(observations: np.ndarray, process_var: float = 1e-3,
                      measurement_var: float = 1.0) -> np.ndarray:
    """Minimal scalar Kalman filter (no external deps). Smooths a per-transect
    position time series before LRR fitting; skips smoothing entirely
    (returns observations unchanged) for fewer than 3 points."""
    if len(observations) < 3:
        return observations
    est = observations[0]
    err = 1.0
    smoothed = [est]
    for z in observations[1:]:
        err += process_var
        k = err / (err + measurement_var)
        est = est + k * (z - est)
        err = (1 - k) * err
        smoothed.append(est)
    return np.array(smoothed)


def track_b_analysis(aoi: str, mask_series: list[tuple[float, np.ndarray]], scale_m: float,
                      n_transects: int = 40, length_px: float = 8, use_kalman: bool = True) -> dict:
    """mask_series: sorted list of (t, mask), earliest first, from
    preprocessing/mask.py's rolling history (or a longer archival series if
    available). Returns a JSON-serializable dict with one entry per transect."""
    baseline_t, baseline_contour, transects = build_baseline_transects(
        mask_series, n_transects=n_transects, length_px=length_px)

    per_transect_series: list[list[tuple[float, float]]] = [[] for _ in transects]
    for t, mask in mask_series:
        contour = extract_main_contour(mask, use_spline=True, spline_smoothing=2.0)
        if contour is None:
            continue
        pts = transect_intersect(transects, contour)
        pts_baseline = transect_intersect(transects, baseline_contour)
        for i, (p, p0) in enumerate(zip(pts, pts_baseline)):
            if p is None or p0 is None:
                continue
            dist = float(np.hypot(p[0] - p0[0], p[1] - p0[1])) * scale_m
            per_transect_series[i].append((t, dist))

    results = []
    for i, series in enumerate(per_transect_series):
        if len(series) < 2:
            results.append({"transect_id": i, "rate_m_per_year": None, "classification": "Tidak valid",
                             "n_obs": len(series)})
            continue
        t_vals = np.array([p[0] for p in series])
        pos_vals = np.array([p[1] for p in series])
        if use_kalman:
            pos_vals = kalman_smooth_1d(pos_vals)
        rate, _ = linear_regression_rate(t_vals, pos_vals)
        label, color = classify_epr(rate)
        p1, p2 = transects[i]
        results.append({"transect_id": i, "rate_m_per_year": rate, "classification": label,
                         "color": color, "n_obs": len(series),
                         "line_px": [[p1[1], p1[0]], [p2[1], p2[0]]]})

    return {"aoi": aoi, "baseline_t": baseline_t, "n_transects": len(transects), "transects": results}
