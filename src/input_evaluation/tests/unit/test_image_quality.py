from __future__ import annotations

from dataclasses import replace

from app.config import settings
from app.services import image_quality


def _base_metrics():
    return image_quality.IQAMetrics(
        blur_laplacian_var=settings.THRESHOLDS.MIN_LAPLACIAN_VAR + 10,
        exposure_hist_entropy=settings.THRESHOLDS.MIN_EXPOSURE_ENTROPY + 1,
        under_over_exposed_ratio=settings.THRESHOLDS.MAX_EXPOSURE_CLIP_RATIO - 0.1,
        noise_estimate_sigma=settings.THRESHOLDS.MAX_NOISE_SIGMA - 5,
        compression_artifacts_score=settings.THRESHOLDS.MAX_BLOCKINESS - 0.05,
        motion_blur_index=settings.THRESHOLDS.MAX_MOTION_BLUR - 0.2,
        skin_patch_detected=True,
        skin_area_ratio=settings.THRESHOLDS.MIN_SKIN_AREA_RATIO + 0.1,
        exif_orientation=1,
    )


def test_decide_detects_blur_first():
    metrics = replace(
        _base_metrics(),
        blur_laplacian_var=settings.THRESHOLDS.MIN_LAPLACIAN_VAR - 0.1,
    )
    usable, msg, culprit = image_quality.decide(metrics)
    assert usable is False
    assert "focus" in msg
    assert culprit == "blur_laplacian_var"


def test_decide_accepts_clean_image():
    usable, msg, culprit = image_quality.decide(_base_metrics())
    assert usable is True
    assert msg == "OK"
    assert culprit == ""
