"""
Per-grain geometry analysis module.

Computes: area, perimeter, centroid, orientation, bounding box,
fitted ellipse, minAreaRect, major/minor axes, solidity, aspect ratio,
Length, Breadth, L/B ratio.

Converts to mm only if calibration exists, otherwise uses pixels.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GrainGeometry:
    """Per-grain geometric measurements."""
    grain_id: int
    area_pixels: float
    perimeter_pixels: float
    centroid: Tuple[float, float]
    orientation_deg: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    fitted_ellipse: Optional[Tuple] = None  # ((cx,cy), (MA,ma), angle)
    min_area_rect: Optional[Tuple] = None   # ((cx,cy), (w,h), angle)
    major_axis_pixels: float = 0.0
    minor_axis_pixels: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 0.0
    length_pixels: float = 0.0
    breadth_pixels: float = 0.0
    length_mm: Optional[float] = None
    breadth_mm: Optional[float] = None
    lb_ratio: Optional[float] = None
    measurement_quality: str = "good"
    is_anomalous: bool = False
    anomaly_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "grain_id": self.grain_id,
            "area_pixels": round(self.area_pixels, 2),
            "perimeter_pixels": round(self.perimeter_pixels, 2),
            "centroid": [round(c, 2) for c in self.centroid],
            "orientation_deg": round(self.orientation_deg, 2),
            "bbox": list(self.bbox),
            "major_axis_pixels": round(self.major_axis_pixels, 2),
            "minor_axis_pixels": round(self.minor_axis_pixels, 2),
            "solidity": round(self.solidity, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "length_pixels": round(self.length_pixels, 2),
            "breadth_pixels": round(self.breadth_pixels, 2),
            "length_mm": round(self.length_mm, 3) if self.length_mm is not None else None,
            "breadth_mm": round(self.breadth_mm, 3) if self.breadth_mm is not None else None,
            "lb_ratio": round(self.lb_ratio, 4) if self.lb_ratio is not None else None,
            "measurement_quality": self.measurement_quality,
            "is_anomalous": self.is_anomalous,
            "anomaly_reasons": self.anomaly_reasons,
        }


def compute_grain_geometry(
    mask: Optional[np.ndarray] = None,
    grain_id: int = 1,
    pixels_per_mm: Optional[float] = None,
    **kwargs,
) -> GrainGeometry:
    """
    Compute geometric properties for a single grain mask.
    
    Args:
        mask: Binary mask (uint8, 0/255) for one grain
        grain_id: Unique grain identifier
        pixels_per_mm: If available, convert to mm
        
    Returns:
        GrainGeometry object
    """
    # Support positional or keyword arguments regardless of order
    if mask is None and "grain_id" in kwargs:
        grain_id = kwargs["grain_id"]
    if mask is None:
        mask = kwargs.get("mask", np.zeros((10, 10), dtype=np.uint8))

    def _empty_geometry(reasons: List[str]) -> GrainGeometry:
        return GrainGeometry(
            grain_id=grain_id,
            area_pixels=0.0,
            perimeter_pixels=0.0,
            centroid=(0.0, 0.0),
            orientation_deg=0.0,
            bbox=(0, 0, 0, 0),
            major_axis_pixels=0.0,
            minor_axis_pixels=0.0,
            solidity=0.0,
            aspect_ratio=0.0,
            length_pixels=0.0,
            breadth_pixels=0.0,
            length_mm=None,
            breadth_mm=None,
            lb_ratio=None,
            measurement_quality="poor",
            is_anomalous=True,
            anomaly_reasons=reasons,
        )

    if mask is None or mask.size == 0 or np.sum(mask) == 0:
        return _empty_geometry(["Empty mask"])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return _empty_geometry(["No contours found in mask"])
    
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 5:
        return _empty_geometry(["Contour area < 5 pixels"])
    
    perimeter = cv2.arcLength(contour, True)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return _empty_geometry(["Zero moment contour"])
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Fitted ellipse (requires >= 5 points)
    fitted_ellipse = None
    major_axis = 0.0
    minor_axis = 0.0
    orientation = 0.0
    
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            fitted_ellipse = ellipse
            (ecx, ecy), (ma, MA), angle = ellipse
            # OpenCV fitEllipse: MA is the larger axis, ma is the smaller
            # But the naming can be confusing — MA/ma are width/height of ellipse
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
            orientation = angle
        except cv2.error:
            major_axis = max(w, h)
            minor_axis = min(w, h)
            orientation = 0.0
    else:
        major_axis = max(w, h)
        minor_axis = min(w, h)
    
    # Min area rectangle
    min_rect = None
    if len(contour) >= 5:
        min_rect = cv2.minAreaRect(contour)
        rect_w, rect_h = min_rect[1]
        # Cross-check with fitted ellipse
    
    # Convex hull for solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    # Length = major dimension, Breadth = minor dimension
    length_pixels = major_axis
    breadth_pixels = minor_axis
    
    # Aspect ratio
    aspect_ratio = length_pixels / breadth_pixels if breadth_pixels > 0 else 0.0
    
    # L/B ratio (protected against division by zero)
    lb_ratio = None
    if breadth_pixels > 0:
        lb_ratio = length_pixels / breadth_pixels
    
    # Convert to mm if calibration exists
    length_mm = None
    breadth_mm = None
    if pixels_per_mm is not None and pixels_per_mm > 0:
        length_mm = length_pixels / pixels_per_mm
        breadth_mm = breadth_pixels / pixels_per_mm
    
    # Anomaly detection
    is_anomalous = False
    anomaly_reasons = []
    
    if lb_ratio is not None and lb_ratio > 10:
        is_anomalous = True
        anomaly_reasons.append(f"Extreme L/B ratio: {lb_ratio:.2f}")
    
    if solidity < 0.5:
        is_anomalous = True
        anomaly_reasons.append(f"Very low solidity: {solidity:.3f}")
    
    if area < 20:
        is_anomalous = True
        anomaly_reasons.append(f"Very small area: {area:.0f} px")
    
    # Measurement quality
    measurement_quality = "good"
    if area < 100:
        measurement_quality = "limited"
    if area < 25:
        measurement_quality = "poor"
    if is_anomalous:
        measurement_quality = "unreliable"
    
    return GrainGeometry(
        grain_id=grain_id,
        area_pixels=area,
        perimeter_pixels=perimeter,
        centroid=(cx, cy),
        orientation_deg=orientation,
        bbox=(x, y, w, h),
        fitted_ellipse=fitted_ellipse,
        min_area_rect=min_rect,
        major_axis_pixels=major_axis,
        minor_axis_pixels=minor_axis,
        solidity=solidity,
        aspect_ratio=aspect_ratio,
        length_pixels=length_pixels,
        breadth_pixels=breadth_pixels,
        length_mm=length_mm,
        breadth_mm=breadth_mm,
        lb_ratio=lb_ratio,
        measurement_quality=measurement_quality,
        is_anomalous=is_anomalous,
        anomaly_reasons=anomaly_reasons,
    )


def compute_robust_whole_kernel_length(
    grain_lengths: List[float],
    threshold_fraction: float = 0.75,
    robust_filter_fraction: float = 0.85,
    robust_iterations: int = 3,
) -> Tuple[Optional[float], str]:
    """
    Compute robust whole-kernel length estimate using iterative median.
    
    Official conceptual rule: broken kernel is below three-fourths
    of the whole-kernel length.
    
    DO NOT use plain mean. Use robust estimator:
    1. L_whole = median(all valid grain lengths)
    2. Iterate 3 times:
       - select grains with length > 0.85 * L_whole
       - recompute L_whole from those grains
    
    For very small samples (N <= 2): returns None with explanation.
    
    Args:
        grain_lengths: List of grain lengths
        threshold_fraction: Fraction of whole-kernel length below which grain is broken (default 0.75)
        robust_filter_fraction: Filter fraction for robust estimation (default 0.85)
        robust_iterations: Number of refinement iterations (default 3)
        
    Returns:
        (whole_kernel_length, status) — status is 'reliable', 'limited', or 'undetermined'
    """
    if not grain_lengths:
        return None, "undetermined"
    
    n = len(grain_lengths)
    
    if n <= 2:
        return None, "undetermined"
    
    lengths = np.array(grain_lengths)
    l_whole = float(np.median(lengths))
    
    for _ in range(robust_iterations):
        selected = lengths[lengths > robust_filter_fraction * l_whole]
        if len(selected) < 3:
            break
        l_whole = float(np.median(selected))
    
    status = "reliable" if n >= 30 else "limited"
    
    return l_whole, status


def classify_broken(
    length: float,
    whole_kernel_length: Optional[float],
    threshold_fraction: float = 0.75,
    small_broken_fraction: float = 0.25,
) -> Dict:
    """
    Classify whether a grain is broken.
    
    Args:
        length: Grain length
        whole_kernel_length: Robust whole-kernel length estimate
        threshold_fraction: Broken if length < this fraction of whole kernel
        small_broken_fraction: Very small broken threshold
        
    Returns:
        Dict with broken status, confidence, and method
    """
    if whole_kernel_length is None or whole_kernel_length <= 0:
        return {
            "broken_label": "undetermined",
            "broken_ratio": None,
            "is_small_broken": None,
            "confidence": 0.0,
            "method": "robust_whole_kernel_estimator",
            "reason": "Cannot be reliably determined from this sample",
        }
    
    ratio = length / whole_kernel_length
    is_broken = ratio < threshold_fraction
    is_small_broken = ratio < small_broken_fraction
    
    return {
        "broken_label": "broken" if is_broken else "whole",
        "broken_ratio": round(ratio, 4),
        "is_small_broken": is_small_broken,
        "confidence": min(1.0, abs(ratio - threshold_fraction) / 0.2 + 0.5),
        "method": "robust_whole_kernel_estimator",
        "threshold_fraction": threshold_fraction,
        "whole_kernel_length_ref": round(whole_kernel_length, 2),
    }
