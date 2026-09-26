"""
Rice Quality Analysis Pipeline.

Orchestrates the complete image-based quality analysis:
1. Validates and loads image
2. Detects rice presence (stops if no rice)
3. Segments individual grains (Mask R-CNN or Classical CV fallback)
4. Detects calibration reference (ArUco or manual, else uncalibrated)
5. Computes per-grain geometry (Length, Breadth, L/B ratio, Area, Solidity, etc.)
6. Classifies 8 defect types per grain (multi-label)
7. Detects full-image foreign matter (YOLO or heuristic)
8. Computes sample-level admixture of lower class
9. Computes sample-level summary statistics
10. Evaluates image quality indicators & assigns tier
11. Compares with official India KMS 2026-27 raw rice standards
12. Generates annotated image with colored masks, Grain IDs, and foreign matter boxes
"""

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ml.admixture import detect_admixture
from ml.calibration import detect_calibration
from ml.classifiers import (
    batch_classify_damaged,
    batch_classify_sprouted_weevilled,
    classify_damaged,
    classify_immature_shrunken,
    classify_sprouted_weevilled,
)
from ml.colour import (
    analyze_dehusked,
    analyze_discoloured,
    analyze_red,
    compute_reference_lab,
    extract_grain_lab_pixels,
)
from ml.config import get_threshold, load_standards
from ml.foreign_matter import detect_foreign_matter
from ml.geometry import (
    classify_broken,
    compute_grain_geometry,
    compute_robust_whole_kernel_length,
)
from ml.preprocessing import (
    ImageValidationError,
    load_image,
    resize_for_inference,
)
from ml.quality import assess_image_quality
from ml.segmentation import (
    GrainInstance,
    detect_rice_presence,
    segment_grains,
)
from ml.standards import compare_with_standards
from ml.texture import (
    extract_chalky_features,
    heuristic_chalky_classification,
)

logger = logging.getLogger(__name__)


class RiceQualityPipeline:
    """
    End-to-end analyzer for rice grain quality.
    """

    def __init__(self, small_sample_threshold: int = 30):
        self.small_sample_threshold = small_sample_threshold

    def analyze(
        self,
        image_source: str | bytes,
        filename: str = "image.jpg",
        manual_scale: Optional[Dict] = None,
        grade: str = "grade_a",
    ) -> Dict[str, Any]:
        """
        Execute full analysis pipeline on an image.

        Args:
            image_source: File path or raw image bytes
            filename: Original filename (for extension check / logging)
            manual_scale: Optional {'reference_pixels': N, 'reference_mm': M}
            grade: 'grade_a' or 'common'

        Returns:
            Dict matching the complete project result schema.
        """
        start_time = time.time()
        warnings: List[str] = []

        # 1. Load and validate image
        try:
            image_rgb, image_info = load_image(image_source, filename=filename)
        except ImageValidationError as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "ValidationError",
                "warnings": [str(e)],
            }
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return {
                "success": False,
                "error": f"Image decoding error: {str(e)}",
                "error_type": "DecodeError",
                "warnings": [f"Image decoding error: {str(e)}"],
            }

        h, w = image_rgb.shape[:2]
        megapixels = (h * w) / 1_000_000

        # 2. Check rice presence
        rice_presence = detect_rice_presence(image_rgb)
        if not rice_presence["has_rice"]:
            logger.info("No rice detected in image.")
            return {
                "success": True,
                "rice_detected": False,
                "message": "No rice grains detected. Please upload an image containing rice grains.",
                "image": image_info.to_dict(),
                "sample": {
                    "total_detected": 0,
                    "analysed": 0,
                    "uncertain": 0,
                    "rejected": 0,
                },
                "grains": [],
                "foreign_matter": [],
                "admixture": {
                    "admixture_status": "not_applicable",
                    "reason": "No rice grains present",
                },
                "summary": {},
                "standards": {
                    "status": "Not evaluated — no rice grains detected",
                },
                "warnings": [
                    "No rice grains detected. Please upload an image containing rice grains."
                ],
                "processing_time_seconds": round(time.time() - start_time, 3),
            }

        if rice_presence["confidence"] < 0.5:
            warnings.append(
                "Rice may be present, but confidence is low. Results may be unreliable."
            )

        # 3. Detect calibration
        calibration_res = detect_calibration(image_rgb, manual_scale=manual_scale)
        pixels_per_mm = calibration_res.pixels_per_mm

        # 4. Grain Instance Segmentation
        # For huge images, we do segmentation with memory-safe sizing if needed
        seg_result = segment_grains(image_rgb, method="auto")
        warnings.extend(seg_result.warnings)

        grains = seg_result.grains
        n_grains = len(grains)

        if n_grains == 0:
            return {
                "success": True,
                "rice_detected": True,
                "message": "Rice detected but segmentation yielded 0 valid grain instances.",
                "image": image_info.to_dict(),
                "sample": {
                    "total_detected": seg_result.detected_count,
                    "analysed": 0,
                    "uncertain": seg_result.uncertain_count,
                    "rejected": seg_result.rejected_count,
                },
                "grains": [],
                "foreign_matter": [],
                "admixture": {"admixture_status": "not reliably estimable"},
                "summary": {},
                "standards": {},
                "warnings": warnings + ["Zero grain instances accepted."],
                "processing_time_seconds": round(time.time() - start_time, 3),
            }

        # 5. Per-grain Geometry
        geometries = []
        grain_areas = []
        grain_confidences = []
        lengths = []

        for grain in grains:
            geom = compute_grain_geometry(
                grain_id=grain.grain_id,
                mask=grain.mask,
                pixels_per_mm=pixels_per_mm,
            )
            geometries.append(geom)
            grain_areas.append(geom.area_pixels)
            grain_confidences.append(grain.confidence)
            effective_length = (
                geom.length_mm if geom.length_mm is not None else geom.length_pixels
            )
            lengths.append(effective_length)

        # 6. Broken Grain Reference Calculation
        whole_kernel_len, whole_len_status = compute_robust_whole_kernel_length(
            lengths
        )
        if whole_kernel_len is None:
            warnings.append(
                "Whole-kernel length reference could not be reliably estimated (sample size <= 2)."
            )

        # Build LAB reference for discolouration
        all_labs = []
        for grain in grains:
            g_lab = extract_grain_lab_pixels(image_rgb, grain.mask)
            if g_lab is not None and len(g_lab) > 0:
                all_labs.append(g_lab)
        ref_lab = compute_reference_lab(all_labs)

        # Population stats for immature/shrunken
        pop_breadths = [
            g.breadth_pixels for g in geometries if g.breadth_pixels > 0
        ]
        population_stats = {
            "median_breadth": (
                float(np.median(pop_breadths)) if pop_breadths else 1.0
            ),
            "median_area": (
                float(np.median(grain_areas)) if grain_areas else 1.0
            ),
        }

        # 7. Multi-Label Defect Classification per grain
        classified_grains = []
        broken_labels = []
        damaged_count = 0
        discoloured_count = 0
        chalky_count = 0
        red_count = 0
        dehusked_count = 0
        immature_count = 0
        sprouted_count = 0
        broken_count = 0

        # Run batch ML classifiers once across all grains (avoids N separate model forward passes)
        grain_masks_list = [g.mask for g in grains]
        damaged_results = batch_classify_damaged(image_rgb, grain_masks_list)
        sprouted_results = batch_classify_sprouted_weevilled(image_rgb, grain_masks_list)

        for i, (grain, geom) in enumerate(zip(grains, geometries)):
            # Broken
            g_len = (
                geom.length_mm if geom.length_mm is not None else geom.length_pixels
            )
            broken_res = classify_broken(
                length=g_len,
                whole_kernel_length=whole_kernel_len,
            )
            broken_labels.append(broken_res["broken_label"])
            if broken_res["broken_label"] == "broken":
                broken_count += 1

            # Chalky
            chalky_feats = extract_chalky_features(image_rgb, grain.mask)
            chalky_res = heuristic_chalky_classification(chalky_feats)
            if chalky_res["chalky_label"] == "chalky":
                chalky_count += 1

            # Damaged (from pre-computed batch)
            damaged_res = damaged_results[i]
            if damaged_res.get("damaged_label") in ("damaged", "slightly_damaged"):
                damaged_count += 1

            # Discoloured
            discoloured_res = analyze_discoloured(
                image_rgb, grain.mask, reference_lab=ref_lab
            )
            if discoloured_res.get("discoloured_label") == "discoloured":
                discoloured_count += 1

            # Red
            red_res = analyze_red(image_rgb, grain.mask)
            if red_res.get("red_label") == "red":
                red_count += 1

            # Dehusked
            dehusked_res = analyze_dehusked(image_rgb, grain.mask)
            if dehusked_res.get("dehusked_label") == "dehusked":
                dehusked_count += 1

            # Immature / Shrunken
            immature_res = classify_immature_shrunken(
                geom.to_dict(), population_stats=population_stats
            )
            if immature_res.get("immature_shrunken_status") == "immature_shrunken":
                immature_count += 1

            # Sprouted / Weevilled (from pre-computed batch)
            sprouted_res = sprouted_results[i]
            if sprouted_res.get("sprouted_weevilled_label") == "sprouted_weevilled":
                sprouted_count += 1

            # RLE or polygon contour for grain mask representation
            contour_pts = []
            if grain.contour is not None:
                # Subsample contour for clean JSON export
                step = max(1, len(grain.contour) // 30)
                contour_pts = [
                    [int(pt[0][0]), int(pt[0][1])]
                    for pt in grain.contour[::step]
                ]

            grain_dict = {
                "id": grain.grain_id,
                "bbox": list(grain.bbox),
                "centroid": [round(c, 2) for c in grain.centroid],
                "confidence": grain.confidence,
                "segmentation_quality": grain.segmentation_quality,
                "is_touching": grain.is_touching,
                "contour": contour_pts,
                "geometry": geom.to_dict(),
                "defects": {
                    "broken": broken_res,
                    "damaged": damaged_res,
                    "discoloured": discoloured_res,
                    "chalky": chalky_res,
                    "red": red_res,
                    "dehusked": dehusked_res,
                    "immature_shrunken": immature_res,
                    "sprouted_weevilled": sprouted_res,
                },
                "sample_level_parameters": {
                    "foreign_matter": "Sample-level parameter",
                    "admixture": "Sample-level parameter",
                    "total_count": "Sample-level parameter",
                },
            }
            classified_grains.append(grain_dict)

        # 8. Foreign Matter Detection (Full Image)
        grain_masks_list = [g.mask for g in grains]
        fm_result = detect_foreign_matter(image_rgb, grain_masks=grain_masks_list)
        warnings.extend(fm_result.warnings)

        # 9. Admixture Detection (Sample-Level)
        seg_qualities = [g.segmentation_quality for g in grains]
        geom_dicts = [g.to_dict() for g in geometries]
        admixture_res = detect_admixture(
            geom_dicts,
            broken_labels=broken_labels,
            segmentation_qualities=seg_qualities,
        )

        # 10. Sample Statistics & Edge Cases
        if n_grains < self.small_sample_threshold:
            if n_grains == 1:
                warnings.append(
                    "Sample size: 1 grain. Individual-grain analysis is possible, "
                    "but sample-level quality percentages are not representative of a larger rice lot."
                )
            else:
                warnings.append(
                    f"Small sample ({n_grains} grains): grain-level classifications are available, "
                    "but sample-level percentages may not represent the composition of a larger rice lot."
                )

        # Metric or pixel lengths
        len_vals = [
            g.length_mm if g.length_mm is not None else g.length_pixels
            for g in geometries
        ]
        brd_vals = [
            g.breadth_mm if g.breadth_mm is not None else g.breadth_pixels
            for g in geometries
        ]
        lb_vals = [g.lb_ratio for g in geometries if g.lb_ratio is not None]

        unit = "mm" if calibration_res.calibrated else "pixels"

        # Safe percentage helper
        def to_pct(count: int, total: int) -> float:
            return round((count / total) * 100.0, 2) if total > 0 else 0.0

        sample_summary = {
            "total_rice_grains": n_grains,
            "uncertain_grains": seg_result.uncertain_count,
            "rejected_grains": seg_result.rejected_count,
            "foreign_matter_count": fm_result.foreign_object_count,
            "admixture_percentage": admixture_res.get("admixture_percentage", 0.0),
            "broken_count": broken_count,
            "broken_percent": to_pct(broken_count, n_grains),
            "damaged_count": damaged_count,
            "damaged_percent": to_pct(damaged_count, n_grains),
            "discoloured_count": discoloured_count,
            "discoloured_percent": to_pct(discoloured_count, n_grains),
            "chalky_count": chalky_count,
            "chalky_percent": to_pct(chalky_count, n_grains),
            "red_count": red_count,
            "red_percent": to_pct(red_count, n_grains),
            "dehusked_count": dehusked_count,
            "dehusked_percent": to_pct(dehusked_count, n_grains),
            "immature_count": immature_count,
            "immature_percent": to_pct(immature_count, n_grains),
            "sprouted_count": sprouted_count,
            "sprouted_percent": to_pct(sprouted_count, n_grains),
            "average_length": round(float(np.mean(len_vals)), 2) if len_vals else 0.0,
            "median_length": round(float(np.median(len_vals)), 2) if len_vals else 0.0,
            "average_breadth": round(float(np.mean(brd_vals)), 2) if brd_vals else 0.0,
            "median_breadth": round(float(np.median(brd_vals)), 2) if brd_vals else 0.0,
            "average_lb_ratio": round(float(np.mean(lb_vals)), 2) if lb_vals else 0.0,
            "median_lb_ratio": round(float(np.median(lb_vals)), 2) if lb_vals else 0.0,
            "measurement_unit": unit,
            "calibration_mode": calibration_res.mode,
            "pixels_per_mm": calibration_res.pixels_per_mm,
            "is_small_sample": n_grains < self.small_sample_threshold,
            "label_note": (
                "Observed sample fraction"
                if n_grains < self.small_sample_threshold
                else "Sample percentage"
            ),
        }

        # 11. Image Quality Assessment
        quality_res = assess_image_quality(
            image_rgb=image_rgb,
            grain_areas=grain_areas,
            grain_confidences=grain_confidences,
            uncertain_count=seg_result.uncertain_count,
            total_count=n_grains,
        )

        # 12. Official Standards Comparison (India KMS 2026-27)
        standards_res = compare_with_standards(
            sample_stats=sample_summary,
            total_count=n_grains,
            calibration_mode=calibration_res.mode,
            grade=grade,
        )
        warnings.extend(standards_res.get("warnings", []))

        # 13. Generate Annotated Image
        annotated_b64 = self._create_annotated_image_b64(
            image_rgb=image_rgb,
            grains=grains,
            geometries=geometries,
            foreign_objects=fm_result.objects,
        )

        total_elapsed = time.time() - start_time

        # Build clean deduplicated warnings list
        unique_warnings = list(dict.fromkeys(warnings))

        return {
            "success": True,
            "rice_detected": True,
            "image": image_info.to_dict(),
            "sample": {
                "total_detected": seg_result.detected_count,
                "analysed": n_grains,
                "uncertain": seg_result.uncertain_count,
                "rejected": seg_result.rejected_count,
                "estimated_merged": seg_result.estimated_merged_count,
            },
            "calibration": calibration_res.to_dict(),
            "quality": quality_res,
            "grains": classified_grains,
            "foreign_matter": fm_result.to_dict()["objects"],
            "admixture": admixture_res,
            "summary": sample_summary,
            "standards": standards_res,
            "annotated_image_base64": annotated_b64,
            "warnings": unique_warnings,
            "processing_time_seconds": round(total_elapsed, 3),
        }

    def _create_annotated_image_b64(
        self,
        image_rgb: np.ndarray,
        grains: List[GrainInstance],
        geometries: List[Any],
        foreign_objects: List[Any],
    ) -> str:
        """Draw masks, bounding boxes, Grain IDs and foreign matter boxes."""
        overlay = image_rgb.copy()
        h, w = image_rgb.shape[:2]

        # Colors
        colors = [
            (34, 197, 94),   # Green
            (59, 130, 246),  # Blue
            (236, 72, 153),  # Pink
            (168, 85, 247),  # Purple
            (20, 184, 166),  # Teal
            (249, 115, 22),  # Orange
        ]

        # Draw grain masks and IDs
        for i, (grain, geom) in enumerate(zip(grains, geometries)):
            color = colors[i % len(colors)]
            mask_bool = grain.mask > 0

            # Blend color mask
            overlay[mask_bool] = (
                overlay[mask_bool] * 0.6 + np.array(color) * 0.4
            ).astype(np.uint8)

            # Draw contour if present
            if grain.contour is not None:
                cv2.drawContours(overlay, [grain.contour], -1, color, 1)

            # Draw Grain #ID text
            cx, cy = int(grain.centroid[0]), int(grain.centroid[1])
            # Only draw label if reasonable size
            if geom.area_pixels >= 40:
                text = f"#{grain.grain_id}"
                font_scale = 0.4 if min(w, h) < 800 else 0.5
                thickness = 1
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                cv2.rectangle(
                    overlay,
                    (cx - tw // 2 - 2, cy - th // 2 - 2),
                    (cx + tw // 2 + 2, cy + th // 2 + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    overlay,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        # Draw Foreign Matter objects (in Red / Yellow with bounding box)
        for fo in foreign_objects:
            fx, fy, fw, fh = fo.bbox
            cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), (239, 68, 68), 2)
            label = f"FM:{fo.class_name}"
            cv2.putText(
                overlay,
                label,
                (fx, max(12, fy - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (239, 68, 68),
                1,
                cv2.LINE_AA,
            )

        # Encode to JPEG base64
        pil_img = Image.fromarray(overlay)
        # Limit preview resolution to avoid giant responses if > 2000px
        if max(w, h) > 1600:
            pil_img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
