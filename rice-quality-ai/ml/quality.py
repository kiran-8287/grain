"""
Image quality assessment module.

Computes image-quality indicators AFTER segmentation:
- Blur score (Laplacian variance)
- Median grain size in pixels
- Segmentation confidence
- Background separation quality
- Illumination uniformity
- Proportion of uncertain grains

Produces quality tiers: GOOD, FAIR, POOR, UNRELIABLE
These are engineering heuristics, NOT scientifically validated scores.
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from ml.config import get_threshold

logger = logging.getLogger(__name__)


def assess_image_quality(
    image_rgb: np.ndarray,
    grain_areas: List[float],
    grain_confidences: List[float],
    uncertain_count: int,
    total_count: int,
) -> Dict:
    """
    Compute image quality assessment.
    
    This happens AFTER grain detection/segmentation, not before.
    Total megapixels is used for logging only — not for rejection.
    
    Args:
        image_rgb: The input image
        grain_areas: List of grain areas in pixels
        grain_confidences: List of segmentation confidences
        uncertain_count: Number of uncertain/merged grain instances
        total_count: Total accepted grain instances
    """
    h, w = image_rgb.shape[:2]
    megapixels = (h * w) / 1_000_000
    
    # 1. Blur score (Laplacian variance)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(np.var(laplacian))
    
    # 2. Median grain size
    median_grain_pixels = float(np.median(grain_areas)) if grain_areas else 0
    
    # 3. Mean segmentation confidence
    mean_confidence = float(np.mean(grain_confidences)) if grain_confidences else 0
    
    # 4. Uncertain grain fraction
    uncertain_fraction = uncertain_count / total_count if total_count > 0 else 0
    
    # 5. Illumination uniformity (std of block means)
    block_size = max(32, min(h, w) // 8)
    block_means = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            block_means.append(float(np.mean(block)))
    illumination_uniformity = 1.0 - min(1.0, float(np.std(block_means)) / 50.0) if block_means else 0.5
    
    # 6. Saturation/clipping check
    clipped_dark = float(np.sum(gray < 5) / (h * w))
    clipped_bright = float(np.sum(gray > 250) / (h * w))
    
    # Quality thresholds
    blur_good = get_threshold("image_quality", "blur_threshold_good", 100)
    blur_fair = get_threshold("image_quality", "blur_threshold_fair", 50)
    blur_poor = get_threshold("image_quality", "blur_threshold_poor", 20)
    
    grain_px_good = get_threshold("image_quality", "min_grain_pixels_good", 400)
    grain_px_fair = get_threshold("image_quality", "min_grain_pixels_fair", 100)
    grain_px_poor = get_threshold("image_quality", "min_grain_pixels_poor", 25)
    
    uncertain_warning = get_threshold("image_quality", "uncertain_grain_fraction_warning", 0.2)
    
    # Determine quality tier
    reasons = []
    scores = []
    
    # Blur
    if blur_score >= blur_good:
        scores.append(1.0)
    elif blur_score >= blur_fair:
        scores.append(0.7)
        reasons.append("Image appears slightly blurry.")
    elif blur_score >= blur_poor:
        scores.append(0.4)
        reasons.append("Image is noticeably blurry.")
    else:
        scores.append(0.1)
        reasons.append("Image is very blurry — measurements may be unreliable.")
    
    # Grain pixel size
    if median_grain_pixels >= grain_px_good:
        scores.append(1.0)
    elif median_grain_pixels >= grain_px_fair:
        scores.append(0.7)
        reasons.append("Many grains occupy relatively few pixels.")
    elif median_grain_pixels >= grain_px_poor:
        scores.append(0.4)
        reasons.append("Many grains occupy very few pixels.")
    else:
        scores.append(0.1)
        reasons.append("Grains are extremely small in the image — measurements unreliable.")
    
    # Uncertain grains
    if uncertain_fraction > uncertain_warning:
        scores.append(0.5)
        reasons.append(f"Several grains appear merged ({uncertain_count} uncertain).")
    else:
        scores.append(1.0)
    
    # Illumination
    if illumination_uniformity < 0.5:
        scores.append(0.5)
        reasons.append("Lighting is highly uneven.")
    else:
        scores.append(min(1.0, illumination_uniformity + 0.2))
    
    # Clipping
    if clipped_dark > 0.1 or clipped_bright > 0.1:
        scores.append(0.6)
        reasons.append("Image has significant clipping (very dark or very bright regions).")
    else:
        scores.append(1.0)
    
    # Overall score
    overall_score = float(np.mean(scores))
    
    if overall_score >= 0.8:
        quality_level = "GOOD"
    elif overall_score >= 0.6:
        quality_level = "FAIR"
    elif overall_score >= 0.35:
        quality_level = "POOR"
    else:
        quality_level = "UNRELIABLE"
    
    return {
        "quality_level": quality_level,
        "quality_score": round(overall_score, 4),
        "blur_score": round(blur_score, 2),
        "median_grain_pixels": round(median_grain_pixels, 1),
        "mean_segmentation_confidence": round(mean_confidence, 4),
        "uncertain_grain_fraction": round(uncertain_fraction, 4),
        "illumination_uniformity": round(illumination_uniformity, 4),
        "clipped_dark_fraction": round(clipped_dark, 4),
        "clipped_bright_fraction": round(clipped_bright, 4),
        "megapixels": round(megapixels, 2),
        "image_dimensions": [w, h],
        "reasons": reasons,
        "_note": "Quality tiers are engineering heuristics, not scientifically validated scores.",
    }
