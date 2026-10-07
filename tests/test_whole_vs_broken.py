"""
Unit and integration tests for Whole vs Broken grain classification.

Tests requirements:
A: 1 whole + trusted profile -> whole
B: 1 broken + trusted profile -> broken
C: 2 whole + trusted profile -> both whole
D: 10 whole -> 10 whole
E: 10 whole + 1 broken -> 10 whole, 1 broken
F: 10 whole + 2 broken -> 10 whole, 2 broken
G: 10 whole + 5 broken -> 10 whole, 5 broken
H: 100 broken + trusted profile -> 100 broken
I: 50 broken + 1 whole + trusted profile -> 50 broken, 1 whole
J: only broken + NO trusted profile -> undetermined, not whole
K: only one grain + NO trusted profile -> undetermined
L: stones/non-rice image -> existing rice-gate behavior unchanged
M: verify summary exposes: total_count, whole_count, broken_count, broken_percent
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from ml.quality.geometry import (
    classify_broken,
    compute_grain_geometry,
    resolve_whole_kernel_reference,
)
from ml.quality.profiles import GrainProfile, load_grain_profile
from ml.segmentation.pipeline import RiceQualityPipeline
from ml.segmentation.segmentation import detect_rice_presence


@pytest.fixture
def trusted_profile():
    return GrainProfile(
        profile_name="test_profile",
        reference_unit="pixels",
        whole_kernel_length=100.0,
        whole_kernel_breadth=30.0,
        whole_kernel_lb_ratio=3.33,
        source="unit_test",
        reference_count=30,
    )


def test_case_a_single_whole_with_trusted_profile(trusted_profile):
    """Test A: 1 whole + trusted profile -> whole"""
    measured_len = 98.0
    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=[measured_len],
        profile=trusted_profile,
    )
    assert ref_len == 100.0
    assert src == "profile"
    assert status == "reliable"

    result = classify_broken(
        length=measured_len,
        whole_kernel_length=ref_len,
        reference_source=src,
        reference_status=status,
    )
    assert result["broken_label"] == "whole"
    assert result["length_ratio"] == 0.98
    assert result["whole_kernel_length_ref"] == 100.0


def test_case_b_single_broken_with_trusted_profile(trusted_profile):
    """Test B: 1 broken + trusted profile -> broken"""
    measured_len = 60.0  # 60% of 100 -> < 75%
    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=[measured_len],
        profile=trusted_profile,
    )
    assert ref_len == 100.0
    assert src == "profile"

    result = classify_broken(
        length=measured_len,
        whole_kernel_length=ref_len,
        reference_source=src,
        reference_status=status,
    )
    assert result["broken_label"] == "broken"
    assert result["length_ratio"] == 0.60
    assert result["is_small_broken"] is False


def test_case_c_two_whole_with_trusted_profile(trusted_profile):
    """Test C: 2 whole + trusted profile -> both whole"""
    lengths = [95.0, 102.0]
    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=lengths,
        profile=trusted_profile,
    )
    assert ref_len == 100.0

    labels = [
        classify_broken(l, ref_len, reference_source=src, reference_status=status)["broken_label"]
        for l in lengths
    ]
    assert labels == ["whole", "whole"]


def test_case_d_ten_whole_sample_derived():
    """Test D: 10 whole -> 10 whole via validated sample-derived reference"""
    # 10 intact whole grains around 100 px (tight CV)
    lengths = [98.0, 101.0, 99.0, 102.0, 100.0, 97.0, 103.0, 101.0, 99.0, 100.0]
    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=lengths,
        profile=None,
    )
    assert ref_len is not None
    assert src == "sample_derived"
    assert status in ("reliable", "limited")
    assert 99.0 <= ref_len <= 101.0

    labels = [
        classify_broken(l, ref_len, reference_source=src, reference_status=status)["broken_label"]
        for l in lengths
    ]
    assert all(lbl == "whole" for lbl in labels)


def test_case_e_ten_whole_one_broken():
    """Test E: 10 whole + 1 broken -> 10 whole, 1 broken"""
    whole_lens = [98.0, 101.0, 99.0, 102.0, 100.0, 97.0, 103.0, 101.0, 99.0, 100.0]
    broken_lens = [62.0]
    all_lens = whole_lens + broken_lens

    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=all_lens,
        profile=None,
    )
    assert ref_len is not None
    assert src == "sample_derived"
    assert 98.0 <= ref_len <= 102.0

    results = [
        classify_broken(l, ref_len, reference_source=src, reference_status=status)["broken_label"]
        for l in all_lens
    ]
    assert results.count("whole") == 10
    assert results.count("broken") == 1


def test_case_f_ten_whole_two_broken():
    """Test F: 10 whole + 2 broken -> 10 whole, 2 broken"""
    whole_lens = [98.0, 101.0, 99.0, 102.0, 100.0, 97.0, 103.0, 101.0, 99.0, 100.0]
    broken_lens = [55.0, 68.0]
    all_lens = whole_lens + broken_lens

    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=all_lens,
        profile=None,
    )
    assert ref_len is not None
    assert src == "sample_derived"

    results = [
        classify_broken(l, ref_len, reference_source=src, reference_status=status)["broken_label"]
        for l in all_lens
    ]
    assert results.count("whole") == 10
    assert results.count("broken") == 2


def test_case_g_ten_whole_five_broken():
    """Test G: 10 whole + 5 broken -> 10 whole, 5 broken"""
    whole_lens = [98.0, 101.0, 99.0, 102.0, 100.0, 97.0, 103.0, 101.0, 99.0, 100.0]
    broken_lens = [52.0, 58.0, 64.0, 69.0, 71.0]
    all_lens = whole_lens + broken_lens

    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=all_lens,
        profile=None,
    )
    assert ref_len is not None
    assert src == "sample_derived"

    results = [
        classify_broken(l, ref_len, reference_source=src, reference_status=status)["broken_label"]
        for l in all_lens
    ]
    assert results.count("whole") == 10
    assert results.count("broken") == 5


def test_case_h_hundred_broken_with_trusted_profile(trusted_profile):
    """Test H: 100 broken + trusted profile -> all 100 broken (does not collapse to whole!)"""
    np.random.seed(42)
    broken_lens = list(np.random.uniform(45.0, 72.0, 100))  # all < 75 px

    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=broken_lens,
        profile=trusted_profile,
    )
    assert ref_len == 100.0
    assert src == "profile"

    results = [
        classify_broken(l, ref_len, reference_source=src, reference_status=status)["broken_label"]
        for l in broken_lens
    ]
    assert results.count("broken") == 100
    assert results.count("whole") == 0


def test_case_i_fifty_broken_one_whole_with_trusted_profile(trusted_profile):
    """Test I: 50 broken + 1 whole + trusted profile -> 50 broken, 1 whole"""
    np.random.seed(42)
    broken_lens = list(np.random.uniform(45.0, 70.0, 50))
    all_lens = broken_lens + [102.0]

    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=all_lens,
        profile=trusted_profile,
    )
    assert ref_len == 100.0

    results = [
        classify_broken(l, ref_len, reference_source=src, reference_status=status)["broken_label"]
        for l in all_lens
    ]
    assert results.count("broken") == 50
    assert results.count("whole") == 1


def test_case_j_all_broken_no_trusted_profile():
    """Test J: only broken + NO trusted profile -> undetermined, NOT whole!"""
    # 20 broken pieces around 55 px
    broken_lens = [50.0, 52.0, 55.0, 57.0, 58.0, 60.0, 54.0, 53.0, 56.0, 58.0]
    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=broken_lens,
        profile=None,
    )
    # Without trusted profile and without evidence of whole kernels, reference is unavailable
    assert ref_len is None or status == "undetermined" or src == "unavailable"

    result = classify_broken(
        length=55.0,
        whole_kernel_length=ref_len,
        reference_source=src,
        reference_status=status,
    )
    assert result["broken_label"] == "undetermined"


def test_case_k_single_grain_no_trusted_profile():
    """Test K: only one grain + NO trusted profile -> undetermined"""
    ref_len, src, status, meta = resolve_whole_kernel_reference(
        lengths=[65.0],
        profile=None,
    )
    assert ref_len is None
    assert status == "undetermined"

    result = classify_broken(
        length=65.0,
        whole_kernel_length=ref_len,
        reference_source=src,
        reference_status=status,
    )
    assert result["broken_label"] == "undetermined"


def test_case_l_stones_rice_gate():
    """Test L: stones only -> rice_gate rejects image"""
    from tests.test_rice_gate import create_stone_image
    pipeline = RiceQualityPipeline()
    res = pipeline.analyze(create_stone_image())
    assert res["success"] is True
    assert res["rice_detected"] is False
    assert res["grains"] == []


def test_case_m_summary_metrics_in_pipeline(trusted_profile):
    """Test M: verify summary exposes total_count, whole_count, broken_count, broken_percent"""
    pipeline = RiceQualityPipeline()

    # Create a synthetic image with 10 whole grains + 2 broken grains
    h, w = 400, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 10 whole grains: ellipse major axis ~100 (half-axis 50), minor axis ~30 (half-axis 15)
    for i in range(10):
        cx = 60 + (i % 5) * 110
        cy = 80 + (i // 5) * 120
        cv2.ellipse(img, (cx, cy), (50, 15), 15, 0, 360, (230, 230, 230), -1)

    # 2 broken grains: ellipse major axis ~55 (half-axis 28), minor axis ~28 (half-axis 14)
    cv2.ellipse(img, (150, 320), (28, 14), 45, 0, 360, (230, 230, 230), -1)
    cv2.ellipse(img, (350, 320), (28, 14), 75, 0, 360, (230, 230, 230), -1)

    enc_bytes = cv2.imencode(".png", img)[1].tobytes()
    res = pipeline.analyze(
        image_source=enc_bytes,
        profile=trusted_profile,
    )
    assert res["success"] is True
    assert res["rice_detected"] is True
    summary = res["summary"]

    assert "total_count" in summary
    assert "whole_count" in summary
    assert "broken_count" in summary
    assert "broken_percent" in summary

    assert summary["total_count"] == 12
    assert summary["whole_count"] == 10
    assert summary["broken_count"] == 2
    assert summary["broken_percent"] == 16.67
