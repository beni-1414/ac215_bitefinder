from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import api
from api.logging import setup_logging, get_logger
import api.services.gcs_io as gcs_io


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


def test_parse_gs_uri_normal():
    uri = "gs://my-bucket/some/path/to/object.jpg"
    bucket, blob = gcs_io.parse_gs_uri(uri)
    assert bucket == "my-bucket"
    assert blob == "some/path/to/object.jpg"


def test_read_bytes_gcs_normal(monkeypatch):
    # Arrange: create a dummy client/bucket/blob that returns known bytes
    data = b"dummy-bytes"

    class DummyBlob:
        def __init__(self, payload):
            self._payload = payload

        def download_as_bytes(self):
            return self._payload

    class DummyBucket:
        def __init__(self, payload):
            self._payload = payload

        def blob(self, name):
            # ensure blob name is passed through
            assert name == "some/path/to/object.jpg"
            return DummyBlob(self._payload)

    class DummyClient:
        def __init__(self, payload):
            self._payload = payload

        def bucket(self, name):
            assert name == "my-bucket"
            return DummyBucket(self._payload)

    monkeypatch.setattr(gcs_io, "get_client", lambda: DummyClient(data))

    # Act
    out = gcs_io.read_bytes_gcs("gs://my-bucket/some/path/to/object.jpg")

    # Assert
    assert out == data
