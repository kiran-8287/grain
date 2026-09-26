"""
Admixture of Lower Class detection module.

This is a SAMPLE-LEVEL parameter — NOT classified per individual grain.

Uses statistical outlier detection on grain geometry:
- Length, Breadth, L/B, Area, Shape descriptors
- Robust Mahalanobis distance
- Excludes broken grains and uncertain segmentation first

Clearly labelled as "Image-based statistical proxy".
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from ml.config import get_threshold

logger = logging.getLogger(__name__)


def detect_admixture(
    grain_geometries: List[Dict],
    broken_labels: List[str],
    segmentation_qualities: List[str],
) -> Dict:
    """
    Detect admixture of lower class grains in the sample.
    
    Sample-level parameter. Runs only when there are enough valid grains.
    First excludes broken grains and uncertain segmentation.
    
    Args:
        grain_geometries: List of grain geometry dicts
        broken_labels: List of broken status per grain ('broken', 'whole', 'undetermined')
        segmentation_qualities: List of segmentation quality per grain
        
    Returns:
        Admixture analysis result
    """
    min_grains = get_threshold("admixture", "min_grains_for_analysis", 10)
    mahalanobis_thresh = get_threshold("admixture", "mahalanobis_threshold", 3.0)
    
    # Filter to whole, well-segmented grains only
    valid_indices = []
    for i, (broken, seg_q) in enumerate(zip(broken_labels, segmentation_qualities)):
        if broken in ("whole", "undetermined") and seg_q in ("good", "limited"):
            valid_indices.append(i)
    
    n_valid = len(valid_indices)
    
    if n_valid < min_grains:
        return {
            "admixture_status": "not reliably estimable",
            "admixture_count": 0,
            "admixture_percentage": 0.0,
            "admixture_confidence": 0.0,
            "valid_grains_used": n_valid,
            "min_required": min_grains,
            "method": "mahalanobis_distance",
            "_source": "engineering_heuristic",
            "reason": f"Too few valid whole grains ({n_valid}) for reliable admixture analysis. "
                      f"Minimum required: {min_grains}.",
            "scope": "sample_level",
        }
    
    # Extract features for valid grains
    features = []
    for i in valid_indices:
        g = grain_geometries[i]
        features.append([
            g.get("length_pixels", 0),
            g.get("breadth_pixels", 0),
            g.get("lb_ratio", 0) or 0,
            g.get("area_pixels", 0),
            g.get("solidity", 0),
        ])
    
    features = np.array(features, dtype=float)
    
    if features.shape[0] < 3 or features.shape[1] < 2:
        return {
            "admixture_status": "not reliably estimable",
            "admixture_count": 0,
            "admixture_percentage": 0.0,
            "admixture_confidence": 0.0,
            "method": "mahalanobis_distance",
            "reason": "Insufficient feature dimensions",
            "scope": "sample_level",
        }
    
    # Compute robust Mahalanobis distance
    try:
        mean = np.mean(features, axis=0)
        cov = np.cov(features.T)
        
        # Regularize covariance
        cov += np.eye(cov.shape[0]) * 1e-6
        
        cov_inv = np.linalg.inv(cov)
        
        distances = []
        for f in features:
            diff = f - mean
            d = float(np.sqrt(diff @ cov_inv @ diff))
            distances.append(d)
        
        distances = np.array(distances)
        
        # Outliers = potential admixture
        outlier_mask = distances > mahalanobis_thresh
        outlier_count = int(np.sum(outlier_mask))
        outlier_percentage = (outlier_count / n_valid * 100) if n_valid > 0 else 0.0
        
        # Confidence based on sample size and separation
        confidence = min(1.0, n_valid / 50.0) * min(1.0, np.mean(distances[outlier_mask]) / mahalanobis_thresh if outlier_count > 0 else 0.5)
        
        return {
            "admixture_status": "detected" if outlier_count > 0 else "not_detected",
            "admixture_count": outlier_count,
            "admixture_percentage": round(outlier_percentage, 2),
            "admixture_confidence": round(float(confidence), 4),
            "valid_grains_used": n_valid,
            "total_grains": len(grain_geometries),
            "outlier_grain_indices": [valid_indices[i] for i in range(len(outlier_mask)) if outlier_mask[i]],
            "mahalanobis_threshold": mahalanobis_thresh,
            "method": "mahalanobis_distance",
            "_source": "engineering_heuristic",
            "basis": "Image-based statistical proxy",
            "scope": "sample_level",
        }
        
    except np.linalg.LinAlgError:
        return {
            "admixture_status": "not reliably estimable",
            "admixture_count": 0,
            "admixture_percentage": 0.0,
            "admixture_confidence": 0.0,
            "method": "mahalanobis_distance",
            "reason": "Covariance matrix is singular — insufficient variation in sample",
            "scope": "sample_level",
        }
