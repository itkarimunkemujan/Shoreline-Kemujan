"""EPR/NSM/SCE shoreline-change metrics and DSAS-style classification bins.

Ported from claude_result.md Cell 10 / all_code2.py. Pure numpy -- no I/O.
"""
from __future__ import annotations

import numpy as np

from src.analysis.spatial import transect_intersect

CHANGE_BINS = [
    (-np.inf, -2.0, "Erosi Parah", "#67001f"),
    (-2.0, -0.5, "Erosi", "#d6604d"),
    (-0.5, 0.5, "Stabil", "#f7f7f7"),
    (0.5, 2.0, "Akresi", "#4393c3"),
    (2.0, np.inf, "Akresi Kuat", "#053061"),
]


def classify_epr(epr: float | None) -> tuple[str, str]:
    if epr is None or (isinstance(epr, float) and np.isnan(epr)):
        return "Tidak valid", "#cccccc"
    for lo, hi, label, color in CHANGE_BINS:
        if lo <= epr < hi:
            return label, color
    return "Tidak valid", "#cccccc"


def compute_epr(transects, contour_first: np.ndarray, contour_last: np.ndarray,
                 time_years: float, px_to_m: float) -> list[float | None]:
    pts_first = transect_intersect(transects, contour_first)
    pts_last = transect_intersect(transects, contour_last)
    eprs = []
    for (p1, p2), pf, pl in zip(transects, pts_first, pts_last):
        if pf is None or pl is None:
            eprs.append(None)
            continue
        disp = np.array([pl[0] - pf[0], pl[1] - pf[1]])
        direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        signed_dist_px = np.dot(disp, direction)
        eprs.append(signed_dist_px * px_to_m / (time_years + 1e-9))
    return eprs


def nsm(epr_series: list[float], total_dt_years: float) -> float:
    """Net Shoreline Movement: EPR at the final observation times elapsed years."""
    return epr_series[-1] * total_dt_years if epr_series else float("nan")


def sce(epr_matrix: np.ndarray, total_dt_years: float) -> np.ndarray:
    """Shoreline Change Envelope: max-min EPR across the time series, per transect."""
    if epr_matrix.size == 0:
        return np.array([])
    return (np.nanmax(epr_matrix, axis=0) - np.nanmin(epr_matrix, axis=0)) * total_dt_years
