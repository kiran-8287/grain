"""Regression checks for single-grain Case 3 behavior."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from ml.classifiers import _choose_sprouted_class
from ml.admixture import detect_admixture
from ml.texture import extract_chalky_features, heuristic_chalky_classification
from ml.pipeline import RiceQualityPipeline
from ml.quality import assess_image_quality
from ml.standards import compare_with_standards
from training.train_chalky import _load_chalky_manifest
from training.train_sprouted import ManifestSproutedDataset


def test_chalky_features_ignore_dark_or_bright_background():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.ellipse(mask, (60, 60), (32, 12), 24, 0, 360, 255, -1)

    dark_background = np.zeros((120, 120, 3), dtype=np.uint8)
    bright_background = np.full((120, 120, 3), 255, dtype=np.uint8)
    grain_color = np.array([232, 225, 214], dtype=np.uint8)
    dark_background[mask > 0] = grain_color
    bright_background[mask > 0] = grain_color

    dark_features = extract_chalky_features(dark_background, mask)
    bright_features = extract_chalky_features(bright_background, mask)

    assert dark_features == bright_features


def test_brightness_only_does_not_emit_chalky_probability():
    result = heuristic_chalky_classification(
        {
            "mean_l_star": 245.0,
            "bright_pixel_fraction": 1.0,
            "chalky_pixel_fraction": 1.0,
        }
    )

    assert result["chalky_label"] == "undetermined"
    assert result["chalky_probability"] is None
    assert "synthetic feature vectors" in result["limitation"]


def test_single_clean_rice_grain_has_no_lot_compliance_verdict():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.ellipse(image, (150, 150), (70, 24), 20, 0, 360, (235, 235, 235), -1)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded

    result = RiceQualityPipeline().analyze(buffer.tobytes())

    assert result["rice_detected"] is True
    assert result["sample"]["analysed"] == 1
    assert len(result["grains"]) == 1
    chalky = result["grains"][0]["defects"]["chalky"]
    assert chalky["chalky_label"] == "undetermined"
    assert chalky["chalky_probability"] is None
    assert result["summary"]["chalky_percent"] is None
    assert result["standards"]["official_grade"]["status"] == (
        "Not determinable from this sample"
    )
    broken_standard = result["standards"]["screening"]["broken"]
    assert broken_standard["detected_count"] == 0
    assert broken_standard["analyzed_count"] == 0
    assert broken_standard["observed_fraction"] is None
    assert all(
        item["status"] != "EXCEEDS REFERENCE LIMIT"
        for item in result["standards"]["screening"].values()
    )
    chalky_standard = result["standards"]["screening"]["chalky"]
    assert chalky_standard["detected_count"] == 0
    assert chalky_standard["analyzed_count"] == 0
    assert chalky_standard["observed_fraction"] is None


def test_two_clean_rice_grains_remain_individual_and_noncompliant_sample_is_indeterminate():
    image = np.zeros((300, 360, 3), dtype=np.uint8)
    cv2.ellipse(image, (90, 150), (48, 17), 20, 0, 360, (235, 235, 235), -1)
    cv2.ellipse(image, (270, 150), (48, 17), -20, 0, 360, (232, 230, 225), -1)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded

    result = RiceQualityPipeline().analyze(buffer.tobytes())

    assert result["rice_detected"] is True
    assert result["sample"]["analysed"] == 2
    assert len(result["grains"]) == 2
    assert all(g["defects"]["chalky"]["chalky_label"] == "undetermined" for g in result["grains"])
    assert all(g["defects"]["damaged"]["model_status"] == "experimental" for g in result["grains"])
    assert all(g["defects"]["sprouted_weevilled"]["model_status"] == "experimental / uncalibrated" for g in result["grains"])
    assert result["standards"]["official_grade"]["status"] == (
        "Not determinable from this sample"
    )


def test_saved_case3_image_reports_model_confidence_and_not_compliance():
    case3_image = PROJECT_ROOT / "data" / "runs" / "025_26_09_26" / "input_image.png"
    result = RiceQualityPipeline().analyze(str(case3_image))

    assert result["rice_detected"] is True
    assert result["sample"]["analysed"] == 1
    grain = result["grains"][0]
    assert grain["defects"]["chalky"]["chalky_label"] == "undetermined"
    assert grain["defects"]["chalky"]["chalky_probability"] is None
    assert grain["defects"]["damaged"]["model_status"] == "experimental"
    assert "not calibrated" in grain["defects"]["damaged"]["confidence_basis"]
    sprouted = grain["defects"]["sprouted_weevilled"]
    assert sprouted["sprouted_weevilled_label"] == "normal"
    assert sprouted["confidence"] == sprouted["probability"]
    assert sprouted["class_mapping"] == {
        "0": "normal",
        "1": "sprouted_weevilled",
    }
    assert "not calibrated" in sprouted["confidence_basis"]
    assert result["summary"]["measurement_unit"] == "pixels"
    assert result["standards"]["official_grade"]["status"] == (
        "Not determinable from this sample"
    )
    assert all(
        item["status"] != "EXCEEDS REFERENCE LIMIT"
        for item in result["standards"]["screening"].values()
    )


def test_unreliable_image_suppresses_official_compliance_statuses():
    result = compare_with_standards(
        sample_stats={
            "broken_count": 12,
            "broken_percent": 30.0,
            "broken_analyzed_count": 40,
        },
        total_count=40,
        quality_tier="UNRELIABLE",
    )

    assert result["official_grade"]["status"] == "Not determinable from this sample"
    assert result["official_grade"]["exceeding_parameters"] == []
    assert result["screening"]["broken"]["status"] == "NOT DETERMINABLE"
    assert result["screening"]["broken"]["detected_count"] == 12
    assert result["screening"]["broken"]["analyzed_count"] == 40
    assert result["screening"]["broken"]["observed_fraction"] == 0.3
    assert result["screening"]["dehusked"]["status"] == "NOT ASSESSABLE"


def test_image_quality_unreliable_is_reachable_by_explicit_hard_failures():
    quality = assess_image_quality(
        np.zeros((64, 64, 3), dtype=np.uint8),
        grain_areas=[10],
        grain_confidences=[0.1],
        uncertain_count=1,
        total_count=1,
        segmentation_qualities=["unreliable"],
    )

    assert quality["quality_level"] == "UNRELIABLE"
    assert quality["quality_score"] >= 0.35
    assert quality["thresholds_used"]["mean_segmentation_confidence_used_for_tier"] is True
    assert quality["unreliable_reasons"]


def test_image_quality_clipping_ignores_intentional_black_background():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    grain_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.ellipse(grain_mask, (50, 50), (30, 12), 20, 0, 360, 255, -1)
    image[grain_mask > 0] = (235, 230, 220)

    quality = assess_image_quality(
        image,
        grain_areas=[int(np.count_nonzero(grain_mask))],
        grain_confidences=[0.95],
        uncertain_count=0,
        total_count=1,
        segmentation_qualities=["good"],
        grain_mask=grain_mask,
    )

    assert quality["clipped_dark_fraction"] == 0
    assert quality["clipping_basis"] == "accepted grain-mask pixels"
    assert quality["quality_level"] != "UNRELIABLE"


def test_image_quality_metrics_are_background_invariant_inside_grain_mask():
    grain_mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.ellipse(grain_mask, (60, 60), (32, 12), 24, 0, 360, 255, -1)
    black_background = np.zeros((120, 120, 3), dtype=np.uint8)
    white_background = np.full((120, 120, 3), 255, dtype=np.uint8)
    grain_pixels = np.indices(grain_mask.shape).sum(axis=0) % 2 == 0
    grain = np.where(grain_pixels[..., None], (224, 220, 212), (231, 226, 218)).astype(np.uint8)
    black_background[grain_mask > 0] = grain[grain_mask > 0]
    white_background[grain_mask > 0] = grain[grain_mask > 0]
    arguments = {
        "grain_areas": [int(np.count_nonzero(grain_mask))],
        "grain_confidences": [0.95],
        "uncertain_count": 0,
        "total_count": 1,
        "segmentation_qualities": ["good"],
        "grain_mask": grain_mask,
    }

    black_quality = assess_image_quality(black_background, **arguments)
    white_quality = assess_image_quality(white_background, **arguments)

    for key in (
        "blur_score",
        "clipped_dark_fraction",
        "clipped_bright_fraction",
        "illumination_uniformity",
        "quality_score",
        "quality_level",
    ):
        assert black_quality[key] == white_quality[key]


def test_sprouted_prediction_uses_mapping_not_assumed_index():
    class_index, confidence = _choose_sprouted_class(
        np.array([0.8, 0.2]),
        {"0": "sprouted_weevilled", "1": "normal"},
        0.5,
    )

    assert class_index == 0
    assert confidence == pytest.approx(0.8)


def test_geometry_outlier_diagnostic_is_not_reported_as_lower_class_admixture():
    geometries = [
        {
            "length_pixels": 10 + index,
            "breadth_pixels": 4 + index * 0.1,
            "lb_ratio": 2.5,
            "area_pixels": 40 + index,
            "solidity": 0.9,
        }
        for index in range(12)
    ]
    result = detect_admixture(geometries, ["whole"] * 12, ["good"] * 12)

    assert result["admixture_status"] == "unsupported"
    assert result["admixture_percentage"] is None
    assert result["geometry_outlier_status"] == "diagnostic_only"


def test_training_refuses_to_generate_synthetic_model_metrics(tmp_path):
    with pytest.raises(FileNotFoundError, match="real labeled grain images"):
        _load_chalky_manifest(tmp_path)
    with pytest.raises(FileNotFoundError, match="real labeled grain images"):
        ManifestSproutedDataset(tmp_path)


def test_sprouted_training_rejects_a_dataset_missing_one_supported_condition(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_path,mask_path,label,source_group\n"
        "normal.jpg,normal.png,normal,sample-a\n"
        "sprout.jpg,sprout.png,sprouted,sample-b\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both sprouted and weevilled"):
        ManifestSproutedDataset(tmp_path)


def test_damaged_demo_trainer_refuses_to_overwrite_synthetic_artifact():
    from training.train_damaged import train_damaged_model

    with pytest.raises(FileNotFoundError, match="Synthetic crop training is disabled"):
        train_damaged_model()