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
from ml.texture import extract_chalky_features, heuristic_chalky_classification
from ml.pipeline import RiceQualityPipeline
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


def test_saved_case3_image_reports_model_confidence_and_not_compliance():
    case3_image = PROJECT_ROOT / "data" / "runs" / "025_26_09_26" / "input_image.png"
    result = RiceQualityPipeline().analyze(str(case3_image))

    assert result["rice_detected"] is True
    assert result["sample"]["analysed"] == 1
    grain = result["grains"][0]
    assert grain["defects"]["chalky"]["chalky_label"] == "undetermined"
    assert grain["defects"]["chalky"]["chalky_probability"] is None
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


def test_sprouted_prediction_uses_mapping_not_assumed_index():
    class_index, confidence = _choose_sprouted_class(
        np.array([0.8, 0.2]),
        {"0": "sprouted_weevilled", "1": "normal"},
        0.5,
    )

    assert class_index == 0
    assert confidence == pytest.approx(0.8)


def test_training_refuses_to_generate_synthetic_model_metrics(tmp_path):
    with pytest.raises(FileNotFoundError, match="real labeled grain images"):
        _load_chalky_manifest(tmp_path)
    with pytest.raises(FileNotFoundError, match="real labeled grain images"):
        ManifestSproutedDataset(tmp_path)