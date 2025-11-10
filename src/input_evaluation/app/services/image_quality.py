from __future__ import annotations
import io
import csv
import math
import argparse
from dataclasses import dataclass
from typing import Counter, Dict, Tuple

import numpy as np
import cv2
from PIL import Image, ExifTags

from app.config import settings

@dataclass
class IQAMetrics:
    blur_laplacian_var: float
    exposure_hist_entropy: float
    under_over_exposed_ratio: float
    noise_estimate_sigma: float
    compression_artifacts_score: float
    motion_blur_index: float
    skin_patch_detected: bool
    skin_area_ratio: float
    exif_orientation: int


def _read_exif_orientation(pil_im: Image.Image) -> int:
    try:
        exif = pil_im._getexif() or {}
        for k, v in ExifTags.TAGS.items():
            if v == 'Orientation':
                return int(exif.get(k, 1))
    except Exception:
        pass
    return 1


def load_image_bytes_to_bgr(img_bytes: bytes) -> Tuple[np.ndarray, int]:
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    orient = _read_exif_orientation(pil)
    arr = np.asarray(pil)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, orient


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_entropy(gray: np.ndarray) -> Tuple[float, float]:
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).ravel()
    p = hist / (hist.sum() + 1e-8)
    entropy = -float(np.nansum(p * np.log2(p + 1e-12)))
    clip_bins = int(256 * 0.02)
    under = p[:clip_bins].sum()
    over = p[-clip_bins:].sum()
    return entropy, float(under + over)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    # Fast wavelet-based approximation via Laplacian MAD
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = 1.4826 * np.median(np.abs(lap - np.median(lap)))
    return float(sigma)


def blockiness_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vert = gray[:, 7::8].astype(np.float32)
    horiz = gray[7::8, :].astype(np.float32)
    if vert.size == 0 or horiz.size == 0:
        return 0.0
    v_score = float(np.mean(np.abs(np.diff(vert, axis=1))))
    h_score = float(np.mean(np.abs(np.diff(horiz, axis=0))))
    raw = (v_score + h_score) / 2.0                   # ~0..255
    return float(raw / 255.0)                         # normalize to 0..1


def motion_blur_index(gray: np.ndarray) -> float:
    # Ratio of low-frequency energy after directional sobel filtering
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    edges = mag > np.percentile(mag, 90)
    # If many edges but Laplacian low, likely motion blur → higher index
    lap_var = variance_of_laplacian(gray)
    idx = float(np.clip((edges.mean() * 100.0) / (lap_var + 1e-6), 0, 5.0))
    return idx


def detect_skin(bgr: np.ndarray) -> Tuple[bool, float]:
    # Simple HSV rules + morphology
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 180, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([179, 180, 255], dtype=np.uint8)
    mask = cv2.bitwise_or(mask1, cv2.inRange(hsv, lower2, upper2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    ratio = float(mask.mean() / 255.0)
    return (ratio > 0.05), ratio


def compute_metrics(img_bytes: bytes) -> IQAMetrics:
    bgr, orient = load_image_bytes_to_bgr(img_bytes)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blur = variance_of_laplacian(gray)
    ent, clip_ratio = exposure_entropy(gray)
    noise = estimate_noise_sigma(gray)
    block = blockiness_score(bgr)
    motion = motion_blur_index(gray)
    skin_ok, skin_ratio = detect_skin(bgr)

    return IQAMetrics(
        blur_laplacian_var=blur,
        exposure_hist_entropy=ent,
        under_over_exposed_ratio=clip_ratio,
        noise_estimate_sigma=noise,
        compression_artifacts_score=block,
        motion_blur_index=motion,
        skin_patch_detected=skin_ok,
        skin_area_ratio=skin_ratio,
        exif_orientation=orient,
    )


def decide(metrics: IQAMetrics) -> Tuple[bool, str, str]:
    t = settings.THRESHOLDS
    # order matters: first failing metric drives the guidance message
    if metrics.blur_laplacian_var < t.MIN_LAPLACIAN_VAR:
        return False, "Image looks out of focus. Hold steady and tap to focus on the bite.", "blur_laplacian_var"
    if metrics.motion_blur_index > t.MAX_MOTION_BLUR:
        return False, "There may be motion blur. Keep still for a second after tapping to focus.", "motion_blur_index"
    if metrics.exposure_hist_entropy < t.MIN_EXPOSURE_ENTROPY or metrics.under_over_exposed_ratio > t.MAX_EXPOSURE_CLIP_RATIO:
        return False, "Lighting is poor. Take the photo in bright, even light (near a window or shade).", "exposure_hist_entropy"
    if metrics.noise_estimate_sigma > t.MAX_NOISE_SIGMA:
        return False, "Image is noisy. Improve lighting or avoid digital zoom.", "noise_estimate_sigma"
    if metrics.compression_artifacts_score > t.MAX_BLOCKINESS:
        return False, "Compression artifacts detected. Use original photo quality and avoid screenshots.", "compression_artifacts_score"
    if not metrics.skin_patch_detected or metrics.skin_area_ratio < t.MIN_SKIN_AREA_RATIO:
        return False, "Ensure the bite area fills about one-third of the frame with plain background.", "skin_area_ratio"
    return True, "OK", ""

# --- CLI for batch scoring ---

def _save_csv(rows: list[dict], out_path: str):
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _iter_local_images(path: str):
    import pathlib
    p = pathlib.Path(path)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for fp in p.rglob(ext):
            yield fp


def cli():
    """CLI to batch score local images and save results to CSV.
    Usage: python image_quality.py --input /path/to/images --out results.csv
    
    NOTE: The dataset is very blurry, so around 250 images in the dataset fail the blur check. This is expected, but present day phones should do better.
    """
    ap = argparse.ArgumentParser(description="Batch score images for heuristic metrics")
    ap.add_argument("--input", required=True, help="Folder of images (local)")
    ap.add_argument("--out", required=True, help="Output CSV file")
    args = ap.parse_args()
    print(f"Scoring images in {args.input}...")

    rows = []
    for fp in _iter_local_images(args.input):
        try:
            with open(fp, 'rb') as f:
                m = compute_metrics(f.read())
            usable, _, culprit = decide(m)
            rows.append({
                "path": str(fp),
                "usable": usable,
                "culprit": culprit,
                **m.__dict__,
            })
        except Exception as e:
            rows.append({"path": str(fp), "error": str(e)})
    if rows:
        _save_csv(rows, args.out)
        # Print a summary of the results, with quantiles of each variable and total number of true-falses
        print("Quantiles of each variable:")
        for key in rows[0].keys():
            if key in ("path", "usable", "error", 'skin_patch_detected','exif_orientation'):
                continue
            vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
            if not vals:
                continue
            q25 = np.percentile(vals, 25)
            q50 = np.percentile(vals, 50)
            q75 = np.percentile(vals, 75)
            print(f"  {key}: Q1={q25:.3f}, Median={q50:.3f}, Q3={q75:.3f}")
        # Count the distinct numbers of culprits
        culprit_counts = Counter(r.get("culprit") for r in rows if "culprit" in r)
        for culprit, count in culprit_counts.items():
            print(f"    {culprit}: {count}")
        false_count = sum(1 for r in rows if r.get("usable") is False)
        print(f"Usable: {len(rows) - false_count}, Not usable: {false_count}")
        print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    cli()