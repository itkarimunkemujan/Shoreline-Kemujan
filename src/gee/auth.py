"""GEE authentication. Production (Actions) path: service-account JSON from
an env var (GitHub Secret). Local-dev fallback: interactive ee.Authenticate().
"""
from __future__ import annotations

import json
import os

import ee


def initialize_ee(project: str) -> None:
    key_data = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    if key_data:
        info = json.loads(key_data)
        credentials = ee.ServiceAccountCredentials(info["client_email"], key_data=key_data)
        ee.Initialize(credentials, project=project)
        return
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
