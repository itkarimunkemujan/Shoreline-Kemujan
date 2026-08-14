"""Download a single-band patch from a GEE image as a numpy array, with
retry+backoff (transient getDownloadURL/requests failures are common)."""
from __future__ import annotations

import io
import time

import ee
import numpy as np
import requests


def download_patch(image: ee.Image, band: str, region: ee.Geometry, scale: float,
                    max_retry: int = 3) -> np.ndarray:
    last_err: Exception | None = None
    for attempt in range(max_retry):
        try:
            url = image.select(band).getDownloadURL({"region": region, "scale": scale, "format": "NPY"})
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return np.load(io.BytesIO(r.content))[band].astype(np.float32)
        except Exception as e:
            last_err = e
            if attempt < max_retry - 1:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to download band {band}: {last_err}")
