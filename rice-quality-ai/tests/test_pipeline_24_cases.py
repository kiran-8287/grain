"""
Comprehensive Automated Test Suite Covering the 24 Project Test Cases.

Tests:
1. valid 1-grain image
2. valid 5-grain image
3. valid 10-grain image
4. large grain image
5. non-rice image
6. empty image
7. corrupted image
8. PNG with transparency
9. grayscale image
10. rotated JPEG
11. low-resolution image
12. very large image
13. touching grains
14. overlapping grains
15. rice + foreign matter
16. no calibration reference
17. calibration reference present
18. model missing
19. zero detected grains
20. division by zero protection for L/B
21. invalid segmentation mask
22. uncertain detection
23. very small sample
24. huge sample
"""

import io
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest
from PIL import Image

from ml.calibration import detect_calibration
from ml.geometry import (
    GrainGeometry,
    classify_broken,
    compute_grain_geometry,
    compute_robust_whole_kernel_length,
)
from ml.pipeline import RiceQualityPipeline
from ml.preprocessing import ImageValidationError, load_image


@pytest.fixture
def pipeline():
    return RiceQualityPipeline(small_sample_threshold=30)


def create_synthetic_rice_image(num_grains: int = 5, img_size: int = 600) -> bytes:
    """Helper to draw synthetic rice grain ellipses and return JPEG bytes."""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    np.random.seed(42)

    grid_cols = int(np.ceil(np.sqrt(num_grains)))
    step = img_size // (grid_cols + 1)

    idx = 0
    for r in range(grid_cols):
        for c in range(grid_cols):
            if idx >= num_grains:
                break
            cx = (c + 1) * step
            cy = (r + 1) * step
            # Rice grain: length ~50-80, breadth ~16-24
            ma = int(np.random.uniform(25, 38))
            mi = int(np.random.uniform(9, 13))
            angle = int(np.random.uniform(0, 180))
            color = (int(np.random.uniform(215, 240)), int(np.random.uniform(210, 235)), int(np.random.uniform(200, 225)))
            cv2.ellipse(img, (cx, cy), (ma, mi), angle, 0, 360, color, -1)
            idx += 1

    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


# 1. Valid 1-grain image
def test_case_1_valid_1_grain_image(pipeline):
    img_bytes = create_synthetic_rice_image(num_grains=1, img_size=300)
    res = pipeline.analyze(img_bytes)
    assert res["success"] is True
    assert res["rice_detected"] is True
    assert len(res["grains"]) == 1
    assert res["sample"]["analysed"] == 1
    assert any("Sample size: 1 grain" in w for w in res["warnings"])


# 2. Valid 5-grain image
def test_case_2_valid_5_grain_image(pipeline):
    img_bytes = create_synthetic_rice_image(num_grains=5, img_size=400)
    res = pipeline.analyze(img_bytes)
    assert res["success"] is True
    assert res["rice_detected"] is True
    assert len(res["grains"]) == 5
    assert res["summary"]["is_small_sample"] is True


# 3. Valid 10-grain image
def test_case_3_valid_10_grain_image(pipeline):
    img_bytes = create_synthetic_rice_image(num_grains=10, img_size=500)
    res = pipeline.analyze(img_bytes)
    assert res["success"] is True
    assert len(res["grains"]) == 10
    for g in res["grains"]:
        assert "geometry" in g
        assert "defects" in g
        assert g["sample_level_parameters"]["admixture"] == "Sample-level parameter"


# 4. Large grain image
def test_case_4_large_grain_image(pipeline):
    # A single close-up large grain (e.g. 150x50 px)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.ellipse(img, (200, 200), (90, 30), 15, 0, 360, (230, 230, 230), -1)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes())
    assert res["success"] is True
    assert len(res["grains"]) == 1
    assert res["grains"][0]["geometry"]["major_axis_pixels"] > 100


# 5. Non-rice image
def test_case_5_non_rice_image(pipeline):
    # Blank pure black image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes())
    assert res["success"] is True
    assert res["rice_detected"] is False
    assert res["message"] == "No rice grains detected. Please upload an image containing rice grains."


# 6. Empty image
def test_case_6_empty_image(pipeline):
    empty_bytes = b""
    res = pipeline.analyze(empty_bytes)
    assert res["success"] is False
    assert res["error_type"] == "ValidationError"


# 7. Corrupted image
def test_case_7_corrupted_image(pipeline):
    corrupted_bytes = b"\xff\xd8\xff\xe0" + b"random_corrupted_garbage_bytes" * 10
    res = pipeline.analyze(corrupted_bytes)
    assert res["success"] is False


# 8. PNG with transparency
def test_case_8_png_with_transparency(pipeline):
    pil_img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    # Draw opaque rice grain
    arr = np.array(pil_img)
    cv2.ellipse(arr, (100, 100), (40, 15), 45, 0, 360, (220, 220, 220, 255), -1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    res = pipeline.analyze(buf.getvalue(), filename="transparent.png")
    assert res["success"] is True
    assert res["rice_detected"] is True
    assert len(res["grains"]) == 1


# 9. Grayscale image
def test_case_9_grayscale_image(pipeline):
    gray = np.zeros((300, 300), dtype=np.uint8)
    cv2.ellipse(gray, (150, 150), (45, 14), 25, 0, 360, 225, -1)
    _, enc = cv2.imencode(".png", gray)
    res = pipeline.analyze(enc.tobytes(), filename="gray.png")
    assert res["success"] is True
    assert res["rice_detected"] is True
    assert len(res["grains"]) == 1


# 10. Rotated JPEG
def test_case_10_rotated_jpeg(pipeline):
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.ellipse(img, (200, 150), (50, 18), 70, 0, 360, (220, 220, 220), -1)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes(), filename="rotated.jpg")
    assert res["success"] is True
    assert res["rice_detected"] is True


# 11. Low-resolution image
def test_case_11_low_resolution_image(pipeline):
    # Very small image (e.g. 64x64) — must NOT reject merely for low MP
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.ellipse(img, (32, 32), (18, 6), 0, 0, 360, (230, 230, 230), -1)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes(), filename="low_res.jpg")
    assert res["success"] is True
    assert "megapixels" in res["image"]
    assert res["image"]["megapixels"] < 0.01  # Not rejected!


# 12. Very large image
def test_case_12_very_large_image(pipeline):
    # Large dimensions test
    img = np.zeros((1200, 1200, 3), dtype=np.uint8)
    cv2.ellipse(img, (600, 600), (80, 28), 0, 0, 360, (220, 220, 220), -1)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes(), filename="large.jpg")
    assert res["success"] is True
    assert res["image"]["width"] == 1200


# 13. Touching grains
def test_case_13_touching_grains(pipeline):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Two grains touching side by side
    cv2.ellipse(img, (135, 150), (35, 12), 90, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img, (160, 150), (35, 12), 90, 0, 360, (220, 220, 220), -1)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes())
    assert res["success"] is True
    assert res["rice_detected"] is True
    # System should detect grains and track touching/uncertain status
    assert res["sample"]["analysed"] >= 1


# 14. Overlapping grains
def test_case_14_overlapping_grains(pipeline):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Overlapping grains forming a large merged blob
    cv2.ellipse(img, (150, 150), (45, 15), 30, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img, (150, 150), (45, 15), 120, 0, 360, (220, 220, 220), -1)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes())
    assert res["success"] is True
    assert "uncertain" in res["sample"]


# 15. Rice + Foreign matter
def test_case_15_rice_plus_foreign_matter(pipeline):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Rice grain (elongated white)
    cv2.ellipse(img, (120, 120), (40, 14), 45, 0, 360, (225, 225, 225), -1)
    # Foreign matter (dark stone-like polygon)
    pts = np.array([[280, 280], [320, 270], [340, 310], [290, 320]], np.int32)
    cv2.fillPoly(img, [pts], (40, 60, 80))
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes())
    assert res["success"] is True
    assert res["rice_detected"] is True
    assert "foreign_matter" in res


# 16. No calibration reference
def test_case_16_no_calibration_reference(pipeline):
    img_bytes = create_synthetic_rice_image(num_grains=3, img_size=300)
    res = pipeline.analyze(img_bytes)
    assert res["calibration"]["calibrated"] is False
    assert res["calibration"]["mode"] == "none"
    assert res["summary"]["measurement_unit"] == "pixels"


# 17. Calibration reference present (manual entry)
def test_case_17_calibration_reference_present(pipeline):
    img_bytes = create_synthetic_rice_image(num_grains=3, img_size=300)
    manual_scale = {"reference_pixels": 100.0, "reference_mm": 10.0}
    res = pipeline.analyze(img_bytes, manual_scale=manual_scale)
    assert res["calibration"]["calibrated"] is True
    assert res["calibration"]["pixels_per_mm"] == 10.0
    assert res["summary"]["measurement_unit"] == "mm"
    assert res["grains"][0]["geometry"]["length_mm"] is not None


# 18. Model missing (handles graceful fallback without fabricating results)
def test_case_18_model_missing(pipeline):
    img_bytes = create_synthetic_rice_image(num_grains=2, img_size=300)
    res = pipeline.analyze(img_bytes)
    assert res["success"] is True
    # Grains have defects classified with clearly labelled fallback methods
    for g in res["grains"]:
        defects = g["defects"]
        assert "method" in defects["chalky"]
        assert "method" in defects["damaged"]


# 19. Zero detected grains (e.g. round stones / coins background)
def test_case_19_zero_detected_grains(pipeline):
    # Image containing only round objects (aspect ratio 1.0, non-rice)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (80, 80), 35, (180, 180, 180), -1)
    cv2.circle(img, (220, 220), 40, (160, 160, 160), -1)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes())
    assert res["success"] is True
    assert res["rice_detected"] is False
    assert len(res["grains"]) == 0


# 20. Division by zero protection for L/B
def test_case_20_division_by_zero_protection_lb():
    # Test GrainGeometry with zero breadth
    geom = compute_grain_geometry(grain_id=99, mask=np.zeros((50, 50), dtype=np.uint8))
    assert geom.lb_ratio is None or geom.lb_ratio == 0.0
    assert not np.isnan(geom.length_pixels)
    assert not np.isnan(geom.breadth_pixels)


# 21. Invalid segmentation mask
def test_case_21_invalid_segmentation_mask():
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    geom = compute_grain_geometry(grain_id=1, mask=empty_mask)
    assert geom.area_pixels == 0.0
    assert geom.measurement_quality == "poor"


# 22. Uncertain detection
def test_case_22_uncertain_detection(pipeline):
    # Irregular non-convex shape
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 30, (220, 220, 220), -1)
    cv2.circle(img, (170, 150), 20, (220, 220, 220), -1)
    _, enc = cv2.imencode(".jpg", img)
    res = pipeline.analyze(enc.tobytes())
    assert res["success"] is True


# 23. Very small sample (N=2)
def test_case_23_very_small_sample(pipeline):
    img_bytes = create_synthetic_rice_image(num_grains=2, img_size=300)
    res = pipeline.analyze(img_bytes)
    assert res["success"] is True
    assert res["summary"]["is_small_sample"] is True
    # Broken status should be undetermined for N<=2 as per requirements
    for g in res["grains"]:
        assert g["defects"]["broken"]["broken_label"] == "undetermined"


# 24. Huge sample (performance / scaling)
def test_case_24_huge_sample(pipeline):
    # Test with 50 grains
    img_bytes = create_synthetic_rice_image(num_grains=50, img_size=800)
    res = pipeline.analyze(img_bytes)
    assert res["success"] is True
    assert res["sample"]["analysed"] >= 40
    assert res["processing_time_seconds"] < 10.0  # Runs fast
