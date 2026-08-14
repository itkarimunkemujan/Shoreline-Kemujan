#!/usr/bin/env python3
"""Mini end-to-end tests for src/output/validate_run.py -- output-contract
validation of one pipeline run. Uses only stdlib unittest (no extra tech
stack): builds a synthetic valid run_dir in a temp dir, then corrupts each
file to assert every check catches its target issue.

Run:  python -m unittest discover tests -v
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone

from src.output.validate_run import validate_run

AOI_CONFIG = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Titik_02"},
            "geometry": {"type": "Point", "coordinates": [110.4666041, -5.7915568]},
        },
        {
            "type": "Feature",
            "properties": {"name": "Titik_07"},
            "geometry": {"type": "Point", "coordinates": [110.4930955, -5.8154436]},
        },
    ],
}


def _make_run_dir(root: str) -> str:
    run_dir = os.path.join(root, "run_test")
    os.makedirs(run_dir, exist_ok=True)
    run_utc = datetime.now(timezone.utc).isoformat()

    current_fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[110.466, -5.791], [110.466, -5.792]]},
             "properties": {"aoi": "Titik_02", "year": 2026, "season": "current", "source": "real"}},
        ],
    }
    predicted_fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[110.466, -5.791], [110.466, -5.792]]},
             "properties": {"aoi": "Titik_02", "year": 2027, "season": "predicted", "source": "rollout"}},
        ],
    }
    transects_fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[110.466, -5.791], [110.466, -5.792]]},
             "properties": {"aoi": "Titik_02", "transectId": 0, "year": 2026,
                            "eprMPerYr": 1.2, "classification": "Akresi"}},
        ],
    }

    def _transect(tid, epr=1.2, nsm=3.0):
        return {
            "id": tid, "eprMPerYr": epr, "nsmM": nsm, "sceM": 5.0,
            "lrrMPerYr": None, "lrrProjectedM": None, "classification": "Akresi",
            "mcUncertainty": None, "modelPositions": [],
        }

    metrics = {
        "run_utc": run_utc,
        "aoiCount": 1,
        "meanNsm": 3.0,
        "meanEpr": 1.2,
        "meanLrr": 0,
    }
    forecast_docs = [{
        "aoi": "Titik_02", "aoiLon": 110.4666041, "aoiLat": -5.7915568,
        "year": 2026, "season": "current", "runUtc": run_utc,
        "meanEprMPerYr": 1.2, "meanNsmM": 3.0, "meanSceM": 5.0, "meanLrrMPerYr": None,
        "classificationCounts": [{"label": "Akresi", "count": 1}],
        "nTransects": 1,
        "transects": [_transect(0)],
    }]

    with open(os.path.join(run_dir, "shoreline_current.geojson"), "w") as f:
        json.dump(current_fc, f)
    with open(os.path.join(run_dir, "shoreline_predicted.geojson"), "w") as f:
        json.dump(predicted_fc, f)
    with open(os.path.join(run_dir, "transects.geojson"), "w") as f:
        json.dump(transects_fc, f)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(run_dir, "forecast_docs.json"), "w") as f:
        json.dump(forecast_docs, f)
    return run_dir


class ValidateRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp()
        cls._aoi = os.path.join(cls._tmp, "aoi_points.geojson")
        with open(cls._aoi, "w") as f:
            json.dump(AOI_CONFIG, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def setUp(self):
        self.run_dir = _make_run_dir(self._tmp)

    def _violations(self):
        return validate_run(self.run_dir, aoi_config=self._aoi)

    # -- each test asserts one check category catches its target issue --

    def test_valid_run_passes(self):
        self.assertEqual(self._violations(), [])

    def test_missing_required_field_schema_drift(self):
        path = os.path.join(self.run_dir, "forecast_docs.json")
        with open(path) as f:
            docs = json.load(f)
        del docs[0]["aoiLon"]
        with open(path, "w") as f:
            json.dump(docs, f)
        violations = self._violations()
        self.assertTrue(any("missing 'aoiLon'" in v for v in violations))

    def test_epr_out_of_bounds(self):
        path = os.path.join(self.run_dir, "forecast_docs.json")
        with open(path) as f:
            docs = json.load(f)
        docs[0]["transects"][0]["eprMPerYr"] = 999.0
        with open(path, "w") as f:
            json.dump(docs, f)
        violations = self._violations()
        self.assertTrue(any("eprMPerYr=999.0 outside" in v for v in violations))

    def test_duplicate_forecast_key(self):
        path = os.path.join(self.run_dir, "forecast_docs.json")
        with open(path) as f:
            docs = json.load(f)
        docs.append(dict(docs[0]))
        with open(path, "w") as f:
            json.dump(docs, f)
        violations = self._violations()
        self.assertTrue(any("duplicate forecast_docs key" in v for v in violations))

    def test_aoi_not_in_config_referential(self):
        path = os.path.join(self.run_dir, "forecast_docs.json")
        with open(path) as f:
            docs = json.load(f)
        docs[0]["aoi"] = "Titik_99"
        with open(path, "w") as f:
            json.dump(docs, f)
        violations = self._violations()
        self.assertTrue(any("not in" in v and "Titik_99" in v for v in violations))

    def test_nan_detected(self):
        path = os.path.join(self.run_dir, "forecast_docs.json")
        with open(path) as f:
            docs = json.load(f)
        docs[0]["meanEprMPerYr"] = math.nan
        with open(path, "w") as f:
            json.dump(docs, f)
        violations = self._violations()
        self.assertTrue(any("not finite" in v for v in violations))

    def test_count_mismatch(self):
        path = os.path.join(self.run_dir, "metrics.json")
        with open(path) as f:
            metrics = json.load(f)
        metrics["aoiCount"] = 5
        with open(path, "w") as f:
            json.dump(metrics, f)
        violations = self._violations()
        self.assertTrue(any("aoiCount=5 != len(forecast_docs)" in v for v in violations))

    def test_future_run_utc_freshness(self):
        future = datetime.now(timezone.utc).timestamp() + 86400 * 3
        future_iso = datetime.fromtimestamp(future, tz=timezone.utc).isoformat()
        path = os.path.join(self.run_dir, "metrics.json")
        with open(path) as f:
            metrics = json.load(f)
        metrics["run_utc"] = future_iso
        with open(path, "w") as f:
            json.dump(metrics, f)
        violations = self._violations()
        self.assertTrue(any("in the future" in v for v in violations))


if __name__ == "__main__":
    unittest.main()