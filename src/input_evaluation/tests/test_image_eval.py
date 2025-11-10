from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
import app.routes.image_eval as image_route
from app.services.image_quality import IQAMetrics


def _dummy_metrics():
    return IQAMetrics(
        blur_laplacian_var=120.0,
        exposure_hist_entropy=4.2,
        under_over_exposed_ratio=0.05,
        noise_estimate_sigma=8.0,
        compression_artifacts_score=0.02,
        motion_blur_index=0.1,
        skin_patch_detected=True,
        skin_area_ratio=0.4,
        exif_orientation=1,
    )


def test_image_eval_handles_gcs_json(monkeypatch):
    # GCS is used; decide returns a failure; metrics come from _dummy_metrics()
    captured = {}

    def fake_read(uri: str) -> bytes:
        captured["uri"] = uri
        return b"fake-bytes"

    monkeypatch.setattr(image_route, "read_bytes_gcs", fake_read)
    monkeypatch.setattr(image_route, "compute_metrics", lambda _: _dummy_metrics())
    monkeypatch.setattr(image_route, "decide", lambda m: (False, "Too blurry", "blur_laplacian_var"))

    client = TestClient(app)
    resp = client.post(
        "/v1/evaluate/image",
        json={"image_gcs_uri": "gs://bucket/bite.jpg"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "gcs"
    assert data["usable"] is False
    assert data["improve_message"] == "Too blurry"
    assert data["metrics"]["blur_laplacian_var"] == 120.0  # from _dummy_metrics()
    assert captured["uri"] == "gs://bucket/bite.jpg"