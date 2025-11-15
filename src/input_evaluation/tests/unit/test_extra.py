from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import api
from api.logging import setup_logging, get_logger


def test_healthz_endpoint():
    client = TestClient(api)
    resp = client.get("/_healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_setup_logging_and_get_logger():
    # Ensure calling setup_logging doesn't raise and returns a logger
    logger = setup_logging()
    assert logger is not None
    # get_logger should return a bound logger instance we can call methods on
    l2 = get_logger()
    assert hasattr(l2, "info")
    # Basic call should not raise
    l2.info("unit-test-log", test_key="value")
