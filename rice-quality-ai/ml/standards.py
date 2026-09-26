"""
Standards comparison engine.

Compares image-derived measurements against the official
India KMS 2026-27 raw rice standard.

Creates TWO separate outputs:
A. Image-based standard screening (parameter-by-parameter comparison)
B. Formal official grade status (honest about limitations)

Does NOT invent a formal Government of India grade from image features.
"""

import logging
from typing import Dict, List, Optional

from ml.config import load_standards

logger = logging.getLogger(__name__)


def compare_with_standards(
    sample_stats: Dict,
    total_count: int,
    calibration_mode: str = "none",
    grade: str = "grade_a",
) -> Dict:
    """
    Compare image-derived measurements with official standards.
    
    Args:
        sample_stats: Dict with defect percentages and counts
        total_count: Total analysed grain count
        calibration_mode: 'aruco', 'manual', or 'none'
        grade: 'grade_a' or 'common'
        
    Returns:
        Standards comparison result
    """
    try:
        standards = load_standards()
    except FileNotFoundError:
        return {
            "screening": {},
            "official_grade": {
                "status": "Standards file not found",
                "message": "Cannot perform standards comparison.",
            },
            "warnings": ["Standards JSON file not found. Supply it manually."],
        }
    
    grade_limits = standards.get(grade, standards.get("grade_a", {}))
    assessment_map = standards.get("image_assessment_mapping", {})
    
    # A. Image-based standard screening
    screening = {}
    warnings = []
    
    param_mappings = [
        ("broken", "broken_percent", "broken"),
        ("damaged_slightly_damaged", "damaged_percent", "damaged_slightly_damaged"),
        ("discoloured", "discoloured_percent", "discoloured"),
        ("chalky", "chalky_percent", "chalky"),
        ("red", "red_percent", "red"),
        ("dehusked", "dehusked_percent", "dehusked"),
        ("foreign_matter", "foreign_matter_percent", "foreign_matter"),
        ("admixture_of_lower_class", "admixture_percent", "admixture_of_lower_class"),
    ]
    
    for param_key, stat_key, assessment_key in param_mappings:
        limit_info = grade_limits.get(param_key, {})
        max_percent = limit_info.get("max_percent")
        basis = limit_info.get("basis", "weight")
        
        observed = sample_stats.get(stat_key)
        assess_info = assessment_map.get(assessment_key, {})
        image_assessable = assess_info.get("image_assessable", False)
        limitation = assess_info.get("limitation", "")
        
        if observed is None or not image_assessable:
            status = "NOT ASSESSABLE"
            difference = None
        elif max_percent is None:
            status = "NOT APPLICABLE"
            difference = None
        elif observed <= max_percent:
            status = "WITHIN REFERENCE LIMIT"
            difference = round(max_percent - observed, 2)
        else:
            status = "EXCEEDS REFERENCE LIMIT"
            difference = round(observed - max_percent, 2)
        
        screening[param_key] = {
            "observed_percent": round(observed, 2) if observed is not None else None,
            "reference_limit_percent": max_percent,
            "official_basis": basis,
            "image_basis": assess_info.get("image_basis", "count"),
            "status": status,
            "difference": difference,
            "limitation": limitation,
            "note": "Image-based screening; not an official laboratory test.",
        }
    
    # Moisture — explicitly NOT measurable
    screening["moisture"] = {
        "observed_percent": None,
        "reference_limit_percent": grade_limits.get("moisture", {}).get("max_percent", 14.0),
        "status": "NOT ASSESSABLE",
        "limitation": "Moisture requires an appropriate physical/laboratory measurement "
                     "and is outside the current image-only pipeline.",
        "note": "Not measurable from this image.",
    }
    
    # B. Formal official grade status
    assessable_params = [s for s in screening.values() 
                        if s["status"] in ("WITHIN REFERENCE LIMIT", "EXCEEDS REFERENCE LIMIT")]
    exceeding = [k for k, s in screening.items() if s["status"] == "EXCEEDS REFERENCE LIMIT"]
    
    reasons_cannot_grade = [
        "Laboratory/weight-based requirements cannot be verified from image alone.",
        "Moisture not measured (requires physical instrument).",
        "Official dehusked test requires chemical staining — not reproduced by image analysis.",
    ]
    
    if total_count < 30:
        reasons_cannot_grade.append(
            f"Small sample size ({total_count} grains) — not representative of a larger lot."
        )
    
    if calibration_mode == "none":
        reasons_cannot_grade.append(
            "No metric calibration — length/breadth measurements are in pixels."
        )
    
    if not exceeding and assessable_params:
        image_verdict = "Meets image-assessable reference limits"
    elif exceeding:
        image_verdict = "Exceeds one or more image-assessable reference limits"
    else:
        image_verdict = "Insufficient data for screening"
    
    official_grade = {
        "formal_grade_determined": False,
        "message": "Formal Grade A/Common determination is not established from this image alone.",
        "reasons": reasons_cannot_grade,
        "image_screening_verdict": image_verdict,
        "exceeding_parameters": exceeding,
        "note": "This is NOT a government certification. "
                "Image-based analysis provides screening estimates only.",
    }
    
    # Footnotes from standard
    footnotes = standards.get("footnotes", [])
    
    return {
        "screening": screening,
        "official_grade": official_grade,
        "standard_reference": {
            "source": standards.get("_metadata", {}).get("source_title", ""),
            "season": standards.get("_metadata", {}).get("season", ""),
            "grade_profile": grade,
        },
        "footnotes": footnotes,
        "warnings": warnings,
    }
