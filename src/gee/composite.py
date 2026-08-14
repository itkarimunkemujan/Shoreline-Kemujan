"""Composite fetching: ONE query for the union bbox of all AOI, cropped many
times client-side -- not one query per AOI. This is the cost-control pattern
established during training data collection (all_code2.py's
fetch_landsat_composite_once); keeping it for scheduled runs matters since
this is what runs monthly indefinitely, not a one-off notebook cell.
"""
from __future__ import annotations

import ee

from src.preprocessing.ndwi import add_mndwi_sentinel


def get_union_bbox(aoi_features: list[dict], pad_deg: float = 0.02) -> ee.Geometry:
    lons = [f["geometry"]["coordinates"][0] for f in aoi_features]
    lats = [f["geometry"]["coordinates"][1] for f in aoi_features]
    return ee.Geometry.Rectangle([min(lons) - pad_deg, min(lats) - pad_deg,
                                   max(lons) + pad_deg, max(lats) + pad_deg])


def fetch_sentinel_composite(union_geom: ee.Geometry, start: str, end: str,
                              cloud_max: float = 30) -> tuple[ee.Image | None, int]:
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(union_geom)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_max)))
    n = coll.size().getInfo()
    if n == 0:
        return None, 0
    composite = add_mndwi_sentinel(coll.median()).clip(union_geom)
    return composite, n
