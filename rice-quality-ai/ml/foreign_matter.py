"""
Foreign matter detection module.

Uses YOLO (when trained) or heuristic fallback to detect non-rice objects.
Foreign matter is detected on the FULL image, not only on rice-grain crops.

Official percentage limit is weight-based. Image-derived quantity is labelled:
"Image-based count/area estimate" — not official laboratory weight percentage.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import get_model_info, get_project_root, get_threshold

logger = logging.getLogger(__name__)


@dataclass
class ForeignObject:
    """A detected foreign matter object."""
    foreign_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_pixels: int = 0

    def to_dict(self) -> Dict:
        return {
            "foreign_id": self.foreign_id,
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "area_pixels": self.area_pixels,
        }


@dataclass
class ForeignMatterResult:
    """Result of foreign matter detection."""
    objects: List[ForeignObject]
    foreign_object_count: int
    foreign_matter_fraction_estimate: float
    total_image_area: int
    method: str
    processing_time_seconds: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "foreign_object_count": self.foreign_object_count,
            "foreign_matter_fraction_estimate": round(self.foreign_matter_fraction_estimate, 6),
            "total_image_area": self.total_image_area,
            "method": self.method,
            "basis": "Image-based count/area estimate — not official laboratory weight percentage",
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "warnings": self.warnings,
        }


def detect_foreign_matter(
    image_rgb: np.ndarray,
    grain_masks: Optional[List[np.ndarray]] = None,
) -> ForeignMatterResult:
    """
    Detect foreign matter in the full image.
    
    Args:
        image_rgb: Full RGB image
        grain_masks: Optional list of grain masks (to exclude rice regions)
    """
    start_time = time.time()
    
    model_info = get_model_info("foreign_matter")
    model_path = get_project_root() / model_info.get("weights_path", "")
    
    if model_path.exists() and model_info.get("status") == "trained":
        result = _detect_foreign_yolo(image_rgb, model_path, grain_masks)
    else:
        result = _detect_foreign_heuristic(image_rgb, grain_masks)
    
    result.processing_time_seconds = time.time() - start_time
    return result


def _detect_foreign_yolo(
    image_rgb: np.ndarray,
    model_path,
    grain_masks: Optional[List[np.ndarray]],
) -> ForeignMatterResult:
    """YOLO-based foreign matter detection."""
    try:
        from ultralytics import YOLO
        
        model = YOLO(str(model_path))
        results = model(image_rgb, verbose=False)
        
        h, w = image_rgb.shape[:2]
        total_area = h * w
        objects = []
        total_foreign_area = 0
        
        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, "other_foreign_matter")
                
                bw = int(x2 - x1)
                bh = int(y2 - y1)
                area = bw * bh
                total_foreign_area += area
                
                objects.append(ForeignObject(
                    foreign_id=i + 1,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), bw, bh),
                    area_pixels=area,
                ))
        
        fraction = total_foreign_area / total_area if total_area > 0 else 0
        
        return ForeignMatterResult(
            objects=objects,
            foreign_object_count=len(objects),
            foreign_matter_fraction_estimate=fraction,
            total_image_area=total_area,
            method="yolo11n",
            processing_time_seconds=0.0,
        )
    except Exception as e:
        logger.warning(f"YOLO inference failed: {e}. Using heuristic fallback.")
        return _detect_foreign_heuristic(image_rgb, grain_masks)


def _detect_foreign_heuristic(
    image_rgb: np.ndarray,
    grain_masks: Optional[List[np.ndarray]],
) -> ForeignMatterResult:
    """
    Heuristic fallback for foreign matter detection.
    Uses colour and shape analysis to identify non-rice objects.
    Clearly labelled as heuristic.
    """
    h, w = image_rgb.shape[:2]
    total_area = h * w
    warnings = [
        "YOLO foreign matter model not available. Using heuristic fallback. "
        "Run: python training/train_foreign_matter.py"
    ]
    
    # Create a combined rice mask
    rice_mask = np.zeros((h, w), dtype=np.uint8)
    if grain_masks:
        for gm in grain_masks:
            if gm.shape == (h, w):
                rice_mask = cv2.bitwise_or(rice_mask, gm)
    
    # Look for objects that are NOT rice using the same background-aware foreground mask
    from ml.segmentation import extract_foreground_mask
    binary, _, _ = extract_foreground_mask(image_rgb)
    
    # Remove rice grain regions from the binary
    non_rice = cv2.bitwise_and(binary, cv2.bitwise_not(rice_mask))
    
    # Find non-rice contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(non_rice, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    total_foreign_area = 0
    fid = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Skip tiny noise
            continue
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # Classify by colour
        roi = image_rgb[y:y+ch, x:x+cw]
        if roi.size == 0:
            continue
        
        mean_color = np.mean(roi.reshape(-1, 3), axis=0)
        
        # Simple heuristic classification
        if mean_color[0] < 80 and mean_color[1] < 80 and mean_color[2] < 80:
            cls_name = "stone"
        elif mean_color[1] > mean_color[0] + 30:
            cls_name = "organic"
        else:
            cls_name = "other_foreign_matter"
        
        fid += 1
        total_foreign_area += area
        
        objects.append(ForeignObject(
            foreign_id=fid,
            class_name=cls_name,
            confidence=0.3,  # Low confidence for heuristic
            bbox=(x, y, cw, ch),
            area_pixels=int(area),
        ))
    
    fraction = total_foreign_area / total_area if total_area > 0 else 0
    
    return ForeignMatterResult(
        objects=objects,
        foreign_object_count=len(objects),
        foreign_matter_fraction_estimate=fraction,
        total_image_area=total_area,
        method="heuristic_fallback",
        processing_time_seconds=0.0,
        warnings=warnings,
    )
