"""Pure-geometry helpers: pixel<->lonlat conversion, contour extraction,
transect construction. No I/O, no GEE, no torch -- shared by both the
model-based (Track A) and statistics-only (Track B) pipelines.

Ported from claude_result.md Cell 10 (notebooks/all_code2.py originally).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString
from skimage import measure


def patch_bounds(lon: float, lat: float, patch_size: int, scale_m: float):
    half_m = patch_size * scale_m / 2
    dlat = half_m / 111320
    dlon = half_m / (111320 * np.cos(np.radians(lat)))
    return [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]


def pixel_to_lonlat(row: float, col: float, lon_c: float, lat_c: float, patch_size: int, scale_m: float):
    bounds = patch_bounds(lon_c, lat_c, patch_size, scale_m)
    south, west = bounds[0]
    north, east = bounds[1]
    lat = north - (row / (patch_size - 1)) * (north - south)
    lon = west + (col / (patch_size - 1)) * (east - west)
    return lon, lat


def smooth_contour_spline(contour: np.ndarray, smoothing: float = 2.0, n_points: int = 300) -> np.ndarray:
    if len(contour) < 4:
        return contour
    y, x = contour[:, 0], contour[:, 1]
    try:
        tck, _ = splprep([x, y], s=smoothing, per=False)
        u_fine = np.linspace(0, 1, n_points)
        x_fine, y_fine = splev(u_fine, tck)
        return np.column_stack([y_fine, x_fine])
    except Exception:
        return contour


def extract_main_contour(mask: np.ndarray, use_spline: bool = True, spline_smoothing: float = 2.0):
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return None
    main = max(contours, key=len)
    if use_spline:
        main = smooth_contour_spline(main, smoothing=spline_smoothing)
    return main


def make_transects(contour_px: np.ndarray, n_transects: int = 40, length_px: float = 8):
    coords_xy = contour_px[:, ::-1]
    line = LineString(coords_xy)
    total_len = line.length
    positions = np.linspace(0.08, 0.92, n_transects) * total_len
    transects = []
    for pos in positions:
        pt = line.interpolate(pos)
        pt2 = line.interpolate(min(pos + 0.5, total_len))
        dx, dy = pt2.x - pt.x, pt2.y - pt.y
        norm = np.hypot(dx, dy) + 1e-9
        nx, ny = -dy / norm, dx / norm
        p1 = (pt.x - nx * length_px, pt.y - ny * length_px)
        p2 = (pt.x + nx * length_px, pt.y + ny * length_px)
        transects.append((p1, p2))
    return transects


def transect_intersect(transects, contour_px: np.ndarray):
    coords_xy = contour_px[:, ::-1]
    shoreline = LineString(coords_xy)
    pts = []
    for p1, p2 in transects:
        t = LineString([p1, p2])
        inter = t.intersection(shoreline)
        if inter.is_empty:
            pts.append(None)
        elif inter.geom_type == "Point":
            pts.append((inter.x, inter.y))
        else:
            pts.append((list(inter.geoms)[0].x, list(inter.geoms)[0].y))
    return pts
