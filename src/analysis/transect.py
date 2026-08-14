"""Ties spatial.py (geometry) + metrics.py (EPR/classification) together into
the actual per-AOI transect analysis: baseline transects from the earliest
contour, EPR per subsequent year relative to that baseline, LRR (linear
regression rate) per transect. Pure computation, no plotting -- ported from
compute_lrr_and_compare / yearly_grid_plot_extended in claude_result.md.
"""
from __future__ import annotations

import numpy as np

from src.analysis.metrics import compute_epr
from src.analysis.spatial import extract_main_contour, make_transects


def build_baseline_transects(mask_series: list[tuple[float, np.ndarray]],
                              n_transects: int = 40, length_px: float = 8,
                              spline_smoothing: float = 2.0):
    """mask_series: sorted list of (t, mask) for one AOI, earliest first."""
    for t, mask in mask_series:
        contour = extract_main_contour(mask, use_spline=True, spline_smoothing=spline_smoothing)
        if contour is not None:
            return t, contour, make_transects(contour, n_transects=n_transects, length_px=length_px)
    raise ValueError("No frame in mask_series produced a valid contour")


def epr_per_year(mask_series: list[tuple[float, np.ndarray]], transects, baseline_t: float,
                  baseline_contour: np.ndarray, scale_m: float, spline_smoothing: float = 2.0):
    """Returns {t: [epr_per_transect]} relative to baseline_t."""
    result = {}
    for t, mask in mask_series:
        contour = extract_main_contour(mask, use_spline=True, spline_smoothing=spline_smoothing)
        dt = t - baseline_t
        if contour is None:
            result[t] = [None] * len(transects)
        elif dt == 0:
            result[t] = [0.0] * len(transects)
        else:
            result[t] = compute_epr(transects, baseline_contour, contour, dt, px_to_m=scale_m)
    return result


def linear_regression_rate(t_values: np.ndarray, positions: np.ndarray) -> tuple[float, float]:
    """Returns (slope_per_year, intercept). positions: signed distance from baseline."""
    slope, intercept = np.polyfit(t_values, positions, 1)
    return float(slope), float(intercept)
