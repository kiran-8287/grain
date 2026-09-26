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
8. Computes a geometry outlier diagnostic; lower-class admixture is unsupported
9. Computes sample-level summary statistics
10. Evaluates image quality indicators & assigns tier
11. Compares observed image fractions with historical/reference rice limits
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
from ml.foreign_matter import detect_foreign_matter, merge_gate_foreign_objects
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
from ml.rice_gate import (
    MESSAGE_NO_ANALYSABLE_RICE,
    STATUS_NO_ANALYSABLE_RICE,
    STATUS_NO_RICE_CLASS,
    STATUS_NOT_RICE,
)
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

    @staticmethod
    def _filter_grains_to_rice(
        grains: List[Any], rice_gate: Optional[Dict[str, Any]]
    ) -> List[Any]:
        """Keep only segmentation results whose boxes match actual rice detections."""
        if not rice_gate or not rice_gate.get("detections"):
            return grains

        rice_boxes = [
            tuple(d.get("bbox") or [])
            for d in rice_gate.get("detections", [])
            if d.get("is_rice") and len(d.get("bbox") or []) == 4
        ]
        if not rice_boxes:
            return []

        def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            x1 = max(ax, bx)
            y1 = max(ay, by)
            x2 = min(ax + aw, bx + bw)
            y2 = min(ay + ah, by + bh)
            return x2 > x1 and y2 > y1

        filtered = []
        for grain in grains:
            grain_box = tuple(grain.bbox)
            if any(_overlaps(grain_box, box) for box in rice_boxes):
                filtered.append(grain)
        return filtered

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

        # 2. Rice-presence gate (Case 1): "objects detected" != "rice detected".
        #    Only detections whose class is the model configuration's rice class
        #    (models/segmentation/class_mapping.json -> class_id 1 "rice_grain")
        #    with confidence >= rice_gate.rice_confidence_threshold count as rice.
        #    Foreign-matter detections are never treated as rice.
        rice_gate = detect_rice_presence(image_rgb)
        gate_status = rice_gate.get("status")
        if gate_status in (
            STATUS_NOT_RICE,
            STATUS_NO_ANALYSABLE_RICE,
            STATUS_NO_RICE_CLASS,
        ) or not rice_gate.get("has_rice", False):
            logger.info(
                "Rice gate FAILED — rice analysis not executed. %s",
                rice_gate.get("debug", ""),
            )
            return self._build_no_rice_response(
                image_info=image_info,
                image_rgb=image_rgb,
                rice_gate=rice_gate,
                warnings=warnings,
                start_time=start_time,
            )

        logger.info("Rice gate PASSED. %s", rice_gate.get("debug", ""))
        warnings.extend(rice_gate.get("warnings", []))

        if rice_gate["confidence"] < 0.5:
            warnings.append(
                "Rice may be present, but confidence is low. Results may be unreliable."
            )

        # 3. Detect calibration
        calibration_res = detect_calibration(image_rgb, manual_scale=manual_scale)
        pixels_per_mm = calibration_res.pixels_per_mm

        # 4. Grain Instance Segmentation
        # For huge images, we do segmentation with memory-safe sizing if needed.
        # IMPORTANT: only rice-gate detections are allowed to survive into the
        # per-grain rice analysis. Non-rice detections remain in the foreign-matter
        # channel and must never become GrainInstance objects.
        seg_result = segment_grains(image_rgb, method="auto")
        warnings.extend(seg_result.warnings)

        grains = self._filter_grains_to_rice(seg_result.grains, rice_gate)
        if len(grains) != len(seg_result.grains):
            warnings.append(
                "Filtered non-rice segmented objects before per-grain rice analysis. "
                "Only rice-gate detections remain in the rice-grain pipeline."
            )
        n_grains = len(grains)

        if n_grains == 0:
            # Rice-class detections existed, but no analysable grain instance was
            # produced: the rice-analysis pipeline must not run on zero grains.
            logger.info(
                "No analysable rice grain instances (%s rice detections) — stopping.",
                rice_gate.get("rice_detections", 0),
            )
            stopped_gate = dict(rice_gate)
            stopped_gate["status"] = STATUS_NO_ANALYSABLE_RICE
            stopped_gate["analysis_stopped"] = True
            stopped_gate["message"] = MESSAGE_NO_ANALYSABLE_RICE
            return self._build_no_rice_response(
                image_info=image_info,
                image_rgb=image_rgb,
                rice_gate=stopped_gate,
                warnings=warnings + ["Zero grain instances accepted."],
                start_time=start_time,
                message=MESSAGE_NO_ANALYSABLE_RICE,
                sample_overrides={
                    "total_detected": seg_result.detected_count,
                    "uncertain": seg_result.uncertain_count,
                    "rejected": seg_result.rejected_count,
                },
            )

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
        chalky_analyzed_count = 0
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
            if chalky_res["chalky_label"] in ("chalky", "not_chalky"):
                chalky_analyzed_count += 1

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
        #    The gate's non-rice detections are merged in so foreign matter is always
        #    reported separately from rice (CASE B: rice + foreign matter).
        grain_masks_list = [g.mask for g in grains]
        fm_result = merge_gate_foreign_objects(
            detect_foreign_matter(image_rgb, grain_masks=grain_masks_list),
            rice_gate,
        )
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

        def analyzed_labels(defect_key: str, label_key: str, known_labels: Tuple[str, ...]) -> int:
            return sum(
                grain["defects"][defect_key].get(label_key) in known_labels
                for grain in classified_grains
            )

        broken_analyzed_count = sum(label in ("broken", "whole") for label in broken_labels)
        damaged_analyzed_count = analyzed_labels(
            "damaged", "damaged_label", ("damaged", "slightly_damaged", "normal")
        )
        discoloured_analyzed_count = analyzed_labels(
            "discoloured", "discoloured_label", ("discoloured", "normal")
        )
        red_analyzed_count = analyzed_labels("red", "red_label", ("red", "not_red"))
        dehusked_analyzed_count = analyzed_labels(
            "dehusked", "dehusked_label", ("dehusked", "not_dehusked")
        )
        immature_analyzed_count = analyzed_labels(
            "immature_shrunken", "immature_shrunken_status", ("immature_shrunken", "normal")
        )
        sprouted_analyzed_count = analyzed_labels(
            "sprouted_weevilled", "sprouted_weevilled_label", ("sprouted_weevilled", "normal")
        )
        admixture_analyzed_count = (
            int(admixture_res.get("valid_grains_used", 0))
            if admixture_res.get("admixture_status") in ("detected", "not_detected")
            else 0
        )

        sample_summary = {
            "total_rice_grains": n_grains,
            "uncertain_grains": seg_result.uncertain_count,
            "rejected_grains": seg_result.rejected_count,
            "foreign_matter_count": fm_result.foreign_object_count,
            "admixture_percentage": (
                admixture_res.get("admixture_percentage")
                if admixture_analyzed_count
                else None
            ),
            "admixture_percent": (
                admixture_res.get("admixture_percentage")
                if admixture_analyzed_count
                else None
            ),
            "admixture_count": admixture_res.get("admixture_count", 0),
            "admixture_analyzed_count": admixture_analyzed_count,
            "admixture_status": admixture_res.get("admixture_status", "unsupported"),
            "geometry_outlier_count": admixture_res.get("geometry_outlier_count"),
            "geometry_outlier_fraction": admixture_res.get("geometry_outlier_fraction"),
            "broken_count": broken_count,
            "broken_percent": to_pct(broken_count, broken_analyzed_count) if broken_analyzed_count else None,
            "broken_analyzed_count": broken_analyzed_count,
            "damaged_count": damaged_count,
            "damaged_percent": to_pct(damaged_count, damaged_analyzed_count) if damaged_analyzed_count else None,
            "damaged_analyzed_count": damaged_analyzed_count,
            "discoloured_count": discoloured_count,
            "discoloured_percent": to_pct(discoloured_count, discoloured_analyzed_count) if discoloured_analyzed_count else None,
            "discoloured_analyzed_count": discoloured_analyzed_count,
            "chalky_count": chalky_count,
            "chalky_percent": (
                to_pct(chalky_count, chalky_analyzed_count)
                if chalky_analyzed_count
                else None
            ),
            "chalky_analyzed_count": chalky_analyzed_count,
            "red_count": red_count,
            "red_percent": to_pct(red_count, red_analyzed_count) if red_analyzed_count else None,
            "red_analyzed_count": red_analyzed_count,
            "dehusked_count": dehusked_count,
            "dehusked_percent": to_pct(dehusked_count, dehusked_analyzed_count) if dehusked_analyzed_count else None,
            "dehusked_analyzed_count": dehusked_analyzed_count,
            "immature_count": immature_count,
            "immature_percent": to_pct(immature_count, immature_analyzed_count) if immature_analyzed_count else None,
            "immature_analyzed_count": immature_analyzed_count,
            "sprouted_count": sprouted_count,
            "sprouted_percent": to_pct(sprouted_count, sprouted_analyzed_count) if sprouted_analyzed_count else None,
            "sprouted_analyzed_count": sprouted_analyzed_count,
            "foreign_matter_percent": None,
            "foreign_matter_analyzed_count": 0,
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
        grain_quality_mask = np.zeros((h, w), dtype=np.uint8)
        for grain_mask in grain_masks_list:
            if grain_mask.shape == grain_quality_mask.shape:
                grain_quality_mask = cv2.bitwise_or(grain_quality_mask, grain_mask)

        quality_res = assess_image_quality(
            image_rgb=image_rgb,
            grain_areas=grain_areas,
            grain_confidences=grain_confidences,
            uncertain_count=seg_result.uncertain_count,
            total_count=n_grains,
            segmentation_qualities=seg_qualities,
            grain_mask=grain_quality_mask,
        )

        # 12. Historical/reference screening with sample- and quality-based suppression.
        standards_res = compare_with_standards(
            sample_stats=sample_summary,
            total_count=n_grains,
            calibration_mode=calibration_res.mode,
            grade=grade,
            quality_tier=quality_res["quality_level"],
            minimum_sample_size=max(30, self.small_sample_threshold),
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
            "rice_gate": rice_gate,
            "annotated_image_base64": annotated_b64,
            "warnings": unique_warnings,
            "processing_time_seconds": round(total_elapsed, 3),
        }

    def _build_no_rice_response(
        self,
        image_info: Any,
        image_rgb: np.ndarray,
        rice_gate: Dict[str, Any],
        warnings: List[str],
        start_time: float,
        message: Optional[str] = None,
        sample_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        CASE 1 response: no valid rice-class detection, so the rice-analysis
        pipeline (grain segmentation, per-grain parameter extraction, defect
        classification, grading, admixture) is NOT executed.

        Foreign matter is still reported separately — foreign matter is NOT rice,
        and the foreign-matter channel must stay available for later work.
        """
        no_rice_message = message or (
            "No rice grains detected. Please upload an image containing rice grains."
        )

        # Foreign matter is detected on the full image (no rice masks exist here)
        # so the non-rice objects remain identifiable even though rice analysis stops.
        fm_result = merge_gate_foreign_objects(
            detect_foreign_matter(image_rgb), rice_gate
        )
        fm_summary = fm_result.to_dict()
        fm_objects = fm_summary.pop("objects")

        sample: Dict[str, Any] = {
            "total_detected": rice_gate.get("total_detections", 0),
            "analysed": 0,
            "uncertain": 0,
            "rejected": 0,
            "rice_detections": rice_gate.get("rice_detections", 0),
            "foreign_matter_detections": rice_gate.get("foreign_matter_detections", 0),
        }
        if sample_overrides:
            sample.update(sample_overrides)

        annotated_b64 = self._create_annotated_image_b64(
            image_rgb=image_rgb,
            grains=[],
            geometries=[],
            foreign_objects=fm_result.objects,
        )

        unique_warnings = list(
            dict.fromkeys(
                warnings
                + list(rice_gate.get("warnings", []))
                + [
                    no_rice_message,
                    "Rice analysis stopped before grain segmentation: no grain "
                    "measurements, defect classification, grading or admixture were computed.",
                ]
            )
        )

        return {
            "success": True,
            "rice_detected": False,
            "message": no_rice_message,
            "image": image_info.to_dict(),
            "sample": sample,
            "calibration": {
                "calibrated": False,
                "mode": "not_evaluated",
                "pixels_per_mm": None,
                "marker_count": 0,
                "message": "Calibration not performed — rice analysis was not executed.",
            },
            "quality": None,
            "grains": [],
            "foreign_matter": fm_objects,
            "foreign_matter_summary": fm_summary,
            "admixture": {
                "admixture_status": "not_applicable",
                "reason": "No rice grains detected — rice analysis was not executed.",
            },
            "summary": {},
            "standards": {"status": "Not evaluated — no rice grains detected"},
            "rice_gate": rice_gate,
            "annotated_image_base64": annotated_b64,
            "warnings": unique_warnings,
            "processing_time_seconds": round(time.time() - start_time, 3),
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
