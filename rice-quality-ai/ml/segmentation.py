"""
Instance segmentation module for rice grains.

Primary method: Mask R-CNN (when trained weights available)
Fallback: Classical CV pipeline (watershed, contours, connected components)

The fallback is clearly labelled as "Classical CV fallback" —
does NOT pretend results came from Mask R-CNN.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import get_threshold, get_model_info, get_project_root

logger = logging.getLogger(__name__)


@dataclass
class GrainInstance:
    """A single segmented grain instance."""
    grain_id: int
    mask: np.ndarray              # Binary mask (full image size)
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    centroid: Tuple[float, float]
    contour: Optional[np.ndarray] = None
    is_touching: bool = False
    is_overlapping: bool = False
    segmentation_quality: str = "good"  # good, uncertain, poor
    method: str = "classical_cv_fallback"

    def mask_area(self) -> int:
        return int(np.sum(self.mask > 0))


@dataclass
class SegmentationResult:
    """Result of grain segmentation."""
    grains: List[GrainInstance]
    detected_count: int
    accepted_count: int
    uncertain_count: int
    rejected_count: int
    estimated_merged_count: int
    method: str
    processing_time_seconds: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "detected_count": self.detected_count,
            "accepted_count": self.accepted_count,
            "uncertain_count": self.uncertain_count,
            "rejected_count": self.rejected_count,
            "estimated_merged_count": self.estimated_merged_count,
            "method": self.method,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "warnings": self.warnings,
        }


def segment_grains(
    image_rgb: np.ndarray,
    method: str = "auto",
) -> SegmentationResult:
    """
    Segment individual rice grains from an image.
    
    Args:
        image_rgb: RGB image (numpy array)
        method: 'mask_rcnn', 'classical_cv', or 'auto'
        
    Returns:
        SegmentationResult with individual grain masks
    """
    start_time = time.time()
    
    # Check if Mask R-CNN weights are available
    model_info = get_model_info("segmentation")
    mask_rcnn_available = (
        model_info.get("status") == "trained" and
        (get_project_root() / model_info.get("weights_path", "")).exists()
    )
    
    if method == "auto":
        method = "mask_rcnn" if mask_rcnn_available else "classical_cv"
    
    if method == "mask_rcnn" and mask_rcnn_available:
        result = _segment_mask_rcnn(image_rgb)
    else:
        if method == "mask_rcnn" and not mask_rcnn_available:
            logger.warning(
                "Mask R-CNN weights not available. "
                "Using Classical CV fallback. "
                "Train segmentation model with: python training/train_segmentation.py"
            )
        result = _segment_classical_cv(image_rgb)
    
    result.processing_time_seconds = time.time() - start_time
    return result


def _segment_mask_rcnn(image_rgb: np.ndarray) -> SegmentationResult:
    """
    Segment using Mask R-CNN.
    Only called when trained weights are available.
    """
    # This will be implemented when model is trained
    # For now, fall back to classical CV with a warning
    logger.warning("Mask R-CNN inference not yet implemented. Using classical CV fallback.")
    result = _segment_classical_cv(image_rgb)
    result.warnings.append("Mask R-CNN model loaded but inference path pending integration.")
    return result


def extract_foreground_mask(
    image_rgb: np.ndarray,
) -> Tuple[np.ndarray, bool, Dict]:
    """
    Extract binary foreground mask (grains = 255, background = 0)
    from an RGB image, automatically handling dark, light, or colored backgrounds.
    
    Uses perimeter border pixel sampling combined with Otsu thresholding.
    
    Returns:
        binary: uint8 array (h, w), 255 for foreground, 0 for background
        is_dark_bg: True if background is dark, False if light
        meta: dictionary with threshold statistics
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Check for blank / zero-variance image
    if int(np.max(gray)) == int(np.min(gray)) or float(np.std(gray)) < 1.0:
        return (
            np.zeros((h, w), dtype=np.uint8),
            True,
            {"is_blank": True, "bg_gray": round(float(np.median(gray)), 2)},
        )
        
    margin = max(3, min(h, w) // 50)
    border_pixels = np.concatenate([
        gray[:margin, :].ravel(),
        gray[-margin:, :].ravel(),
        gray[:, :margin].ravel(),
        gray[:, -margin:].ravel(),
    ])
    
    otsu_val, thresh_bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_dark = cv2.bitwise_not(thresh_bright)
    
    dark_border_frac = float(np.mean(border_pixels < otsu_val))
    bright_fg_ratio = float(np.sum(thresh_bright > 0) / (h * w))
    dark_fg_ratio = float(np.sum(thresh_dark > 0) / (h * w))
    
    # Border fraction is the primary physical indicator of background polarity:
    # In grain imaging, background surrounds the perimeter of the sample.
    if dark_border_frac >= 0.5:
        # Dark background (e.g. black, navy, dark tray) -> grains are bright
        # Sanity check: grains rarely occupy > 70% of the field of view
        if bright_fg_ratio > 0.70 and dark_fg_ratio < 0.30:
            binary = thresh_dark
            is_dark_bg = False
        else:
            binary = thresh_bright
            is_dark_bg = True
    else:
        # Light background (e.g. white paper, light tray) -> grains are dark
        if dark_fg_ratio > 0.70 and bright_fg_ratio < 0.30:
            binary = thresh_bright
            is_dark_bg = True
        else:
            binary = thresh_dark
            is_dark_bg = False
            
    meta = {
        "otsu_val": float(otsu_val),
        "dark_border_frac": round(dark_border_frac, 3),
        "is_dark_bg": is_dark_bg,
        "fg_ratio": round(float(np.sum(binary > 0) / (h * w)), 4),
        # Median border intensity — used by the rice gate as the background reference.
        "bg_gray": round(float(np.median(border_pixels)), 2),
    }
    return binary, is_dark_bg, meta


def _segment_classical_cv(image_rgb: np.ndarray) -> SegmentationResult:
    """
    Classical CV fallback pipeline for grain segmentation.
    
    Steps:
    1. Robust background estimation via perimeter sampling
    2. Foreground extraction (Otsu thresholding calibrated to background polarity)
    3. Morphological operations
    4. Distance transform
    5. Adaptive per-component watershed splitting for touching grains
    6. Contour extraction
    7. Connected components analysis
    
    This is clearly labelled as "Classical CV fallback".
    """
    h, w = image_rgb.shape[:2]
    warnings = []
    
    binary, is_dark_bg, meta = extract_foreground_mask(image_rgb)
    if meta.get("is_blank", False) or np.sum(binary > 0) == 0:
        return SegmentationResult(
            grains=[],
            detected_count=0,
            accepted_count=0,
            uncertain_count=0,
            rejected_count=0,
            estimated_merged_count=0,
            method="classical_cv_fallback",
            processing_time_seconds=0.0,
            warnings=["No foreground grain objects detected in image."],
        )
        
    if is_dark_bg:
        warnings.append("Detected dark background — segmented bright grain foreground")
    else:
        warnings.append("Detected light background — segmented dark grain foreground")
    
    # Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove small noise and fill small interior holes
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Distance transform for watershed
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    
    # Adaptive sure foreground markers per connected component
    # Using per-component peaks prevents large clumps from swallowing small grains
    num_cc, cc_labels = cv2.connectedComponents(closed)
    sure_fg = np.zeros_like(closed)
    
    for lab in range(1, num_cc):
        comp_mask = (cc_labels == lab)
        max_d = np.max(dist_transform[comp_mask])
        if max_d > 0:
            t = max(2.0, 0.40 * max_d)
            sure_fg[comp_mask & (dist_transform >= t)] = 255
    
    # Sure background (dilated foreground)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_bg = cv2.dilate(closed, kernel_bg, iterations=3)
    
    # Unknown region between sure background and sure foreground
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Connected components on sure foreground markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background is 1, not 0
    markers[unknown == 255] = 0  # Unknown region is 0
    
    # Watershed segmentation
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(image_bgr, markers)
    
    # Extract individual grain masks
    min_area = get_threshold("segmentation", "min_grain_area_pixels", 50)
    merge_area_ratio = get_threshold("segmentation", "merge_detection_area_ratio", 2.5)
    merge_aspect_ratio = get_threshold("segmentation", "merge_detection_aspect_ratio", 4.0)
    
    grains = []
    uncertain_count = 0
    rejected_count = 0
    estimated_merged = 0
    grain_id = 0
    
    # Collect all grain areas for median computation
    all_areas = []
    label_areas = {}
    for label in range(2, num_labels + 1):
        grain_mask = np.zeros((h, w), dtype=np.uint8)
        grain_mask[markers == label] = 255
        area = np.sum(grain_mask > 0)
        if area >= min_area:
            all_areas.append(area)
            label_areas[label] = area
    
    median_area = float(np.median(all_areas)) if all_areas else 0
    
    for label in range(2, num_labels + 1):
        grain_mask = np.zeros((h, w), dtype=np.uint8)
        grain_mask[markers == label] = 255
        
        area = np.sum(grain_mask > 0)
        
        if area < min_area:
            rejected_count += 1
            continue
        
        # Find contour
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            rejected_count += 1
            continue
        
        contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            rejected_count += 1
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        grain_id += 1
        
        # Check for potential merge
        is_touching = False
        seg_quality = "good"
        
        if median_area > 0:
            area_ratio = area / median_area
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            
            if area_ratio > merge_area_ratio:
                is_touching = True
                seg_quality = "uncertain"
                uncertain_count += 1
                estimated_merged += 1
                warnings.append(f"Grain #{grain_id}: possibly merged (area ratio {area_ratio:.1f}x median)")
            elif aspect > merge_aspect_ratio:
                is_touching = True
                seg_quality = "uncertain"
                uncertain_count += 1
        
        # Segmentation confidence — heuristic based on solidity and area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        confidence = min(1.0, solidity * 0.7 + 0.3)  # Heuristic confidence
        
        grain = GrainInstance(
            grain_id=grain_id,
            mask=grain_mask,
            bbox=(x, y, bw, bh),
            confidence=round(confidence, 4),
            centroid=(cx, cy),
            contour=contour,
            is_touching=is_touching,
            segmentation_quality=seg_quality,
            method="classical_cv_fallback",
        )
        grains.append(grain)
    
    detected_count = grain_id + rejected_count
    accepted_count = len(grains)
    
    if not grains:
        warnings.append("No grain instances found after segmentation.")
    
    return SegmentationResult(
        grains=grains,
        detected_count=detected_count,
        accepted_count=accepted_count,
        uncertain_count=uncertain_count,
        rejected_count=rejected_count,
        estimated_merged_count=estimated_merged,
        method="classical_cv_fallback",
        processing_time_seconds=0.0,
        warnings=warnings,
    )


def detect_rice_presence(image_rgb: np.ndarray) -> Dict:
    """
    Decide whether the image contains RICE grains (Case 1 rice-presence gate).

    Delegates to ml.rice_gate.evaluate_rice_presence — the single central decision
    point — so that "objects were detected" is never confused with "rice was
    detected". Only detections whose class is the model configuration's rice class
    (models/segmentation/class_mapping.json -> class_id 1 "rice_grain") whose
    confidence passes rice_gate.rice_confidence_threshold count as rice; every
    other object keeps the project's foreign-matter class vocabulary.

    The public signature and the legacy summary keys are preserved for existing
    callers (`has_rice`, `confidence`, `estimated_grain_count`,
    `total_objects_detected`, `message`, `method`).
    """
    # Imported lazily: ml.rice_gate re-uses extract_foreground_mask from this
    # module, so a module-level import would be circular.
    from ml.rice_gate import evaluate_rice_presence

    return evaluate_rice_presence(image_rgb)
