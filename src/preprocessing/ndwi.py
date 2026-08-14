"""MNDWI computation + cloud masking on Earth Engine images.

Server-side (ee.Image) computation only -- this runs before any pixel array
is downloaded, since MNDWI needs the raw spectral bands and only the final
mask/MNDWI array is worth transferring off GEE.
"""
from __future__ import annotations

import ee

SENTINEL_BANDS = {"green": "B3", "swir1": "B11"}

LANDSAT_BANDS = {
    "LANDSAT/LC08/C02/T1_L2": {"green": "SR_B3", "swir1": "SR_B6", "qa": "QA_PIXEL",
                                "scale_factor": 0.0000275, "offset": -0.2},
    "LANDSAT/LE07/C02/T1_L2": {"green": "SR_B2", "swir1": "SR_B5", "qa": "QA_PIXEL",
                                "scale_factor": 0.0000275, "offset": -0.2},
}


def add_mndwi_sentinel(image: ee.Image) -> ee.Image:
    mndwi = image.normalizedDifference([SENTINEL_BANDS["green"], SENTINEL_BANDS["swir1"]]).rename("MNDWI")
    return image.addBands(mndwi)


def mask_landsat_clouds(image: ee.Image, collection_id: str) -> ee.Image:
    """QA_PIXEL Collection 2: bit1=dilated cloud, bit3=cloud, bit4=cloud shadow."""
    qa = image.select(LANDSAT_BANDS[collection_id]["qa"])
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    shadow = qa.bitwiseAnd(1 << 4).neq(0)
    dilated = qa.bitwiseAnd(1 << 1).neq(0)
    mask = cloud.Or(shadow).Or(dilated).Not()
    return image.updateMask(mask)


def add_mndwi_landsat(image: ee.Image, collection_id: str) -> ee.Image:
    bands = LANDSAT_BANDS[collection_id]
    green = image.select(bands["green"])
    swir1 = image.select(bands["swir1"]).resample("bilinear").reproject(crs=green.projection(), scale=10)
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI")
    return image.addBands(mndwi)
