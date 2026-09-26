"""
Calibration module for rice grain measurements.

Supports two modes:
- MODE A: Physical reference (ArUco markers or known-size object)
- MODE B: No reference — measurements in pixels, L/B ratio still valid

Does NOT invent mm values when no calibration reference exists.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import get_threshold

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Calibration analysis result."""
    calibrated: bool
    mode: str  # 'aruco', 'manual', 'none'
    pixels_per_mm: Optional[float] = None
    marker_count: int = 0
    homography: Optional[np.ndarray] = None
    rectified: bool = False
    message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "calibrated": self.calibrated,
            "mode": self.mode,
            "pixels_per_mm": round(self.pixels_per_mm, 4) if self.pixels_per_mm else None,
            "marker_count": self.marker_count,
            "rectified": self.rectified,
            "message": self.message,
        }


def detect_calibration(
    image_rgb: np.ndarray,
    manual_scale: Optional[Dict] = None,
) -> CalibrationResult:
    """
    Detect calibration reference in image.
    
    Checks for ArUco markers first. If not found, checks for
    manual scale entry. Otherwise returns uncalibrated.
    
    Args:
        image_rgb: Input RGB image
        manual_scale: Optional manual scale {'reference_pixels': N, 'reference_mm': M}
    """
    # Try ArUco detection
    aruco_result = _detect_aruco(image_rgb)
    if aruco_result.calibrated:
        return aruco_result
    
    # Try manual scale
    if manual_scale:
        ref_px = manual_scale.get("reference_pixels", 0)
        ref_mm = manual_scale.get("reference_mm", 0)
        if ref_px > 0 and ref_mm > 0:
            ppmm = ref_px / ref_mm
            return CalibrationResult(
                calibrated=True,
                mode="manual",
                pixels_per_mm=ppmm,
                message=f"Manual calibration: {ppmm:.2f} pixels/mm "
                        f"({ref_px} px = {ref_mm} mm)",
            )
    
    # No calibration
    return CalibrationResult(
        calibrated=False,
        mode="none",
        message="Metric calibration unavailable — measurements shown in pixels. "
                "L/B ratio does not require absolute scale and is valid.",
    )


def _detect_aruco(image_rgb: np.ndarray) -> CalibrationResult:
    """Detect ArUco markers for automatic calibration."""
    try:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Get ArUco dictionary
        dict_name = get_threshold("calibration", "aruco_dictionary", "DICT_4X4_50")
        aruco_dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return CalibrationResult(
                calibrated=False,
                mode="none",
                message="No ArUco markers detected.",
            )
        
        marker_size_mm = get_threshold("calibration", "marker_size_mm", 20.0)
        n_markers = len(ids)
        
        # Calculate pixels per mm from marker size
        # Use the first marker's side length
        marker_corners = corners[0][0]
        side_lengths = []
        for i in range(4):
            p1 = marker_corners[i]
            p2 = marker_corners[(i + 1) % 4]
            side_lengths.append(np.linalg.norm(p2 - p1))
        
        avg_side_pixels = np.mean(side_lengths)
        pixels_per_mm = avg_side_pixels / marker_size_mm
        
        # If 4+ markers, compute homography for perspective correction
        rectified = False
        homography = None
        
        if n_markers >= 4:
            try:
                # Sort markers by ID
                src_points = []
                for i in range(min(4, n_markers)):
                    center = np.mean(corners[i][0], axis=0)
                    src_points.append(center)
                
                if len(src_points) >= 4:
                    src_pts = np.array(src_points[:4], dtype=np.float32)
                    # Compute perspective transform
                    rect_w = 500
                    rect_h = 500
                    dst_pts = np.array([
                        [0, 0], [rect_w, 0],
                        [rect_w, rect_h], [0, rect_h]
                    ], dtype=np.float32)
                    
                    homography, _ = cv2.findHomography(src_pts, dst_pts)
                    rectified = True
            except Exception as e:
                logger.warning(f"Homography computation failed: {e}")
        
        return CalibrationResult(
            calibrated=True,
            mode="aruco",
            pixels_per_mm=float(pixels_per_mm),
            marker_count=n_markers,
            homography=homography,
            rectified=rectified,
            message=f"ArUco calibration: {n_markers} markers detected, "
                    f"{pixels_per_mm:.2f} pixels/mm"
                    + (", perspective-rectified" if rectified else ""),
        )
        
    except Exception as e:
        logger.warning(f"ArUco detection failed: {e}")
        return CalibrationResult(
            calibrated=False,
            mode="none",
            message=f"ArUco detection failed: {e}",
        )
