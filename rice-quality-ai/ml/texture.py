"""
Texture feature extraction for rice grain quality analysis.

Implements GLCM (Gray-Level Co-occurrence Matrix) features for chalkiness detection
and general texture analysis.

Features extracted:
- GLCM contrast, homogeneity, energy, correlation
- LAB brightness features (mean L*, std L*, bright-pixel fraction)
- Chalky pixel fraction
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ml.config import get_threshold

logger = logging.getLogger(__name__)


def compute_glcm(
    gray_image: np.ndarray,
    distances: List[int] = None,
    angles: List[float] = None,
    levels: int = 256,
) -> np.ndarray:
    """
    Compute GLCM manually (avoiding skimage dependency for basic version).
    Uses a simplified approach for the key features.
    
    For production, use skimage.feature.graycomatrix.
    """
    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Quantize to fewer levels for efficiency
    n_levels = min(levels, 64)
    if gray_image.max() > 0:
        quantized = (gray_image.astype(float) / gray_image.max() * (n_levels - 1)).astype(np.uint8)
    else:
        quantized = gray_image.astype(np.uint8)
    
    h, w = quantized.shape
    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)
    
    for d in distances:
        for angle in angles:
            dx = int(round(d * np.cos(angle)))
            dy = int(round(-d * np.sin(angle)))
            
            for i in range(max(0, -dy), min(h, h - dy)):
                for j in range(max(0, -dx), min(w, w - dx)):
                    row_val = quantized[i, j]
                    col_val = quantized[i + dy, j + dx]
                    glcm[row_val, col_val] += 1
    
    # Normalize
    total = glcm.sum()
    if total > 0:
        glcm = glcm / total
    
    return glcm


def glcm_features(glcm: np.ndarray) -> Dict[str, float]:
    """Extract GLCM properties: contrast, homogeneity, energy, correlation."""
    n = glcm.shape[0]
    i_indices, j_indices = np.meshgrid(range(n), range(n), indexing='ij')
    i_indices = i_indices.astype(float)
    j_indices = j_indices.astype(float)
    
    # Contrast
    contrast = float(np.sum(glcm * (i_indices - j_indices) ** 2))
    
    # Homogeneity (Inverse Difference Moment)
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(i_indices - j_indices))))
    
    # Energy (Angular Second Moment)
    energy = float(np.sum(glcm ** 2))
    
    # Correlation
    mu_i = float(np.sum(i_indices * glcm))
    mu_j = float(np.sum(j_indices * glcm))
    sigma_i = float(np.sqrt(np.sum(glcm * (i_indices - mu_i) ** 2)))
    sigma_j = float(np.sqrt(np.sum(glcm * (j_indices - mu_j) ** 2)))
    
    correlation = 0.0
    if sigma_i > 0 and sigma_j > 0:
        correlation = float(
            np.sum(glcm * (i_indices - mu_i) * (j_indices - mu_j)) / (sigma_i * sigma_j)
        )
    
    return {
        "glcm_contrast": round(contrast, 4),
        "glcm_homogeneity": round(homogeneity, 4),
        "glcm_energy": round(energy, 6),
        "glcm_correlation": round(correlation, 4),
    }


def extract_chalky_features(
    grain_rgb: np.ndarray,
    grain_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Extract features for chalkiness detection.
    
    Features:
    - LAB: mean L*, std L*, mean a*, mean b*
    - Brightness: bright-pixel fraction, local brightness stats
    - GLCM: contrast, homogeneity, energy, correlation
    - Mask: chalky pixel fraction, internal texture stats
    """
    from ml.colour import extract_grain_crop
    
    crop_rgb, crop_mask = extract_grain_crop(grain_rgb, grain_mask)
    
    if crop_mask.sum() == 0:
        return _empty_chalky_features()
    
    # LAB features
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    grain_lab = lab[crop_mask > 0].astype(float)
    
    l_star = grain_lab[:, 0]
    a_star = grain_lab[:, 1] - 128  # Center a*
    b_star = grain_lab[:, 2] - 128  # Center b*
    
    mean_l = float(np.mean(l_star))
    std_l = float(np.std(l_star))
    mean_a = float(np.mean(a_star))
    mean_b = float(np.mean(b_star))
    
    # Bright pixel fraction
    bright_threshold = get_threshold("chalky", "bright_pixel_l_threshold", 85)
    bright_fraction_thresh = get_threshold("chalky", "bright_pixel_fraction_threshold", 0.3)
    bright_pixels = np.sum(l_star > bright_threshold)
    total_pixels = len(l_star)
    bright_fraction = bright_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # GLCM features on grayscale of grain region
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    # Apply mask
    gray_masked = gray.copy()
    gray_masked[crop_mask == 0] = 0
    
    glcm = compute_glcm(gray_masked)
    glcm_feats = glcm_features(glcm)
    
    # Chalky pixel fraction estimate
    # Chalky regions appear white/opaque — high L*, low saturation
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    grain_hsv = hsv[crop_mask > 0]
    low_sat = grain_hsv[:, 1] < 40
    high_val = grain_hsv[:, 2] > 180
    chalky_pixels = np.sum(low_sat & high_val)
    chalky_fraction = chalky_pixels / total_pixels if total_pixels > 0 else 0.0
    
    features = {
        "mean_l_star": round(mean_l, 4),
        "std_l_star": round(std_l, 4),
        "mean_a_star": round(mean_a, 4),
        "mean_b_star": round(mean_b, 4),
        "bright_pixel_fraction": round(bright_fraction, 4),
        "chalky_pixel_fraction": round(float(chalky_fraction), 4),
        **glcm_feats,
    }
    
    return features


def _empty_chalky_features() -> Dict[str, float]:
    return {
        "mean_l_star": 0.0,
        "std_l_star": 0.0,
        "mean_a_star": 0.0,
        "mean_b_star": 0.0,
        "bright_pixel_fraction": 0.0,
        "chalky_pixel_fraction": 0.0,
        "glcm_contrast": 0.0,
        "glcm_homogeneity": 0.0,
        "glcm_energy": 0.0,
        "glcm_correlation": 0.0,
    }


def heuristic_chalky_classification(features: Dict[str, float]) -> Dict:
    """
    Heuristic fallback for chalkiness when ML model is unavailable.
    
    Uses bright-pixel fraction and chalky-pixel fraction as proxies.
    Clearly labelled as heuristic.
    """
    bright_frac = features.get("bright_pixel_fraction", 0)
    chalky_frac = features.get("chalky_pixel_fraction", 0)
    mean_l = features.get("mean_l_star", 0)
    
    score = 0.4 * bright_frac + 0.4 * chalky_frac + 0.2 * (mean_l / 255.0)
    
    threshold = get_threshold("chalky", "probability_threshold", 0.5)
    
    return {
        "chalky_label": "chalky" if score >= threshold else "not_chalky",
        "chalky_probability": round(score, 4),
        "chalky_model_version": "heuristic_fallback_v1",
        "method": "heuristic — LAB brightness + chalky pixel fraction",
        "_source": "engineering_heuristic",
        "limitation": "Heuristic fallback — ML model not available",
    }
