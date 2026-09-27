"""
scripts/generate_synthetic.py
==============================
Generate synthetic rice grain images with COCO instance segmentation annotations.

Sources single-grain images from data/raw/graindet_rice_v2/rice/0_NOR/ and composites
them onto blank or textured backgrounds with random augmentations (rotation, scale,
brightness, contrast, blur). Produces multiple density modes:

  * sparse        – 10-20 isolated grains per image
  * touching      – 2-5 pairs of touching grains
  * overlapping   – 2-3 pairs of overlapping grains
  * dense         – 50-100 grains
  * very_dense    – 100-250 grains
  * foreign_matter – grains + synthetic foreign debris

Outputs COCO-format JSON and composited PNG images under:
    datasets/processed/coco/images/synthetic/
    datasets/processed/coco/annotations/instances_synthetic.json

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --num-isolated 200 --seed 123
    python scripts/generate_synthetic.py --num-foreign 50
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
GRAINDET_V2_NOR_DIR = PROJECT_ROOT / "data" / "raw" / "graindet_rice_v2" / "rice" / "0_NOR"
COCO_OUT_DIR        = PROJECT_ROOT / "datasets" / "processed" / "coco"
SYNTH_IMAGES_DIR    = COCO_OUT_DIR / "images" / "synthetic"
SYNTH_ANN_PATH      = COCO_OUT_DIR / "annotations" / "instances_synthetic.json"

# ── Constants ──────────────────────────────────────────────────────────────────
CATEGORIES = [
    {"id": 1, "name": "rice_grain",     "supercategory": "rice"},
    {"id": 2, "name": "foreign_matter", "supercategory": "other"},
]

MIN_GRAIN_AREA    = 30
SIMPLIFY_EPSILON  = 1.5
DEFAULT_IMG_SIZE  = 640

MODE_CONFIG = {
    "sparse":       {"num_min": 10,  "num_max": 20,  "touch_pairs": 0, "overlap_pairs": 0},
    "touching":     {"num_min": 4,   "num_max": 10,  "touch_pairs": (2, 5), "overlap_pairs": 0},
    "overlapping":  {"num_min": 4,   "num_max": 10,  "touch_pairs": 0, "overlap_pairs": (2, 3)},
    "dense":        {"num_min": 50,  "num_max": 100, "touch_pairs": 0, "overlap_pairs": 0},
    "very_dense":   {"num_min": 100, "num_max": 250, "touch_pairs": 0, "overlap_pairs": 0},
    "foreign":      {"num_min": 10,  "num_max": 40,  "touch_pairs": 0, "overlap_pairs": 0},
}


# ── Helpers: mask ↔ polygon ────────────────────────────────────────────────────

def _mask_to_polygon(binary_mask: np.ndarray) -> Optional[List[float]]:
    """Convert a uint8 binary mask to a flat COCO polygon [x1,y1,x2,y2,...]."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_GRAIN_AREA:
        return None
    approx = cv2.approxPolyDP(contour, SIMPLIFY_EPSILON, True)
    if len(approx) < 3:
        return None
    return approx.flatten().tolist()


def _mask_to_bbox_area(binary_mask: np.ndarray) -> Tuple[List[int], int]:
    """Return bbox [x,y,w,h] and pixel area for a binary mask."""
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0], 0
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
    area = int(len(xs))
    return [x, y, w, h], area


# ── 1. Extract grain mask from source image ────────────────────────────────────

def extract_grain_mask(grain_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a per-grain binary mask and tightly-cropped RGB grain image.

    Uses Otsu thresholding on the luminance (V) channel of HSV after a small
    Gaussian blur to suppress sensor noise. The mask is then tightly cropped
    to the grain's bounding box so it can be pasted cleanly onto backgrounds.

    Parameters
    ----------
    grain_img : np.ndarray
        BGR or RGB source image depicting a single rice grain (with any background).

    Returns
    -------
    mask : np.ndarray
        uint8 binary mask (255 = grain, 0 = background) cropped to the grain.
        Shape is (H, W), single channel.
    cropped_rice : np.ndarray
        Cropped grain image with the same H, W as mask. Shape is (H, W, 3), BGR.
    """
    if grain_img.ndim == 2:
        grain_bgr = cv2.cvtColor(grain_img, cv2.COLOR_GRAY2BGR)
    elif grain_img.shape[2] == 4:
        grain_bgr = cv2.cvtColor(grain_img, cv2.COLOR_BGRA2BGR)
    else:
        grain_bgr = grain_img.copy()

    hsv = cv2.cvtColor(grain_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    _, binary = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        binary = 255 - binary
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    best_label, best_area = -1, -1
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area, best_label = area, lbl

    if best_label < 0 or best_area < MIN_GRAIN_AREA:
        h, w = grain_bgr.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return mask, grain_bgr

    clean_mask = (labels == best_label).astype(np.uint8) * 255
    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    cropped_mask = clean_mask[y:y + h, x:x + w]
    cropped_rice = grain_bgr[y:y + h, x:x + w]

    return cropped_mask, cropped_rice


# ── Fallback: synthetic grain shapes ───────────────────────────────────────────

def _generate_synthetic_grain(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic rice-grain-shaped image + mask when no real sources exist.

    Draws a rotated ellipse (elongated, realistic aspect ratio ~3.2:1) filled
    with a natural rice-colour gradient and mild per-pixel noise.
    """
    major = int(rng.integers(55, 90))
    minor = int(major // rng.uniform(2.8, 3.8))
    pad = 6
    h = major + pad * 2
    w = minor * 2 + pad * 2
    cx, cy = w // 2, h // 2
    angle = rng.uniform(-15, 15)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (minor, major // 2), angle, 0, 360, 255, -1)

    rice_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    b_val = int(rng.integers(175, 220))
    g_val = int(rng.integers(200, 235))
    r_val = int(rng.integers(215, 245))
    rice_bgr[:] = (b_val, g_val, r_val)

    grad_y = np.linspace(0.65, 1.0, h, dtype=np.float32)
    grad_x = np.linspace(0.7, 1.0, w, dtype=np.float32)
    grad = np.outer(grad_y, grad_x)
    rice_bgr = np.clip(rice_bgr.astype(np.float32) * grad[:, :, None], 0, 255).astype(np.uint8)

    noise = rng.normal(0, 8, rice_bgr.shape).astype(np.float32)
    rice_bgr = np.clip(rice_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    mask_3c = (mask > 0).astype(np.uint8)[:, :, None]
    rice_bgr = rice_bgr * mask_3c

    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        mask = mask[y0:y1, x0:x1]
        rice_bgr = rice_bgr[y0:y1, x0:x1]

    return mask, rice_bgr


def _generate_foreign_matter_piece(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single synthetic foreign-matter piece (stone / straw / husk splinter).

    Returns (mask_bgra, piece_bgr) — same conventions as source grains.
    """
    fm_type = rng.choice(["stone", "straw", "husk", "dark_speck"])
    if fm_type == "stone":
        r = int(rng.integers(6, 20))
        h, w = r * 2 + 6, r * 2 + 6
        cx, cy = w // 2, h // 2
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = []
        n_pts = int(rng.integers(6, 10))
        for i in range(n_pts):
            ang = (i / n_pts) * 2 * np.pi + rng.uniform(-0.3, 0.3)
            rr = r * rng.uniform(0.7, 1.2)
            pts.append([int(cx + np.cos(ang) * rr), int(cy + np.sin(ang) * rr)])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        tone = int(rng.integers(50, 120))
        bgr[:] = (tone, tone + int(rng.integers(-5, 15)), tone + int(rng.integers(5, 25)))
    elif fm_type == "straw":
        length = int(rng.integers(40, 90))
        thickness = int(rng.integers(3, 8))
        pad = 4
        h = length + pad * 2
        w = thickness * 2 + pad * 2
        cx, cy = w // 2, h // 2
        angle = rng.uniform(-40, 40)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (thickness // 2, length // 2), angle, 0, 360, 255, -1)
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[:] = (int(rng.integers(40, 90)), int(rng.integers(120, 170)), int(rng.integers(180, 220)))
    elif fm_type == "husk":
        length = int(rng.integers(25, 55))
        thickness = int(rng.integers(6, 12))
        pad = 4
        h = length + pad * 2
        w = thickness * 2 + pad * 2
        cx, cy = w // 2, h // 2
        angle = rng.uniform(-60, 60)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (thickness // 2, length // 2), angle, 0, 360, 255, -1)
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[:] = (int(rng.integers(25, 60)), int(rng.integers(60, 110)), int(rng.integers(90, 150)))
    else:
        r = int(rng.integers(3, 9))
        h, w = r * 2 + 4, r * 2 + 4
        cx, cy = w // 2, h // 2
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        tone = int(rng.integers(10, 45))
        bgr[:] = (tone, tone, tone)

    noise = rng.normal(0, 7, bgr.shape).astype(np.float32)
    bgr = np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    mask_3c = (mask > 0).astype(np.uint8)[:, :, None]
    bgr = bgr * mask_3c

    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        mask = mask[y0:y1, x0:x1]
        bgr = bgr[y0:y1, x0:x1]

    return mask, bgr


# ── Source grain loading ───────────────────────────────────────────────────────

def _load_source_grains(rng: np.random.Generator,
                        fallback_count: int = 60) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load real single-grain images from 0_NOR folder; fall back to synthetic shapes.

    Each tuple in the returned list is (mask_uint8, grain_bgr_uint8) cropped tightly.
    """
    source_grains: List[Tuple[np.ndarray, np.ndarray]] = []
    if GRAINDET_V2_NOR_DIR.exists():
        files = sorted(list(GRAINDET_V2_NOR_DIR.glob("*.png")) +
                       list(GRAINDET_V2_NOR_DIR.glob("*.jpg")))
        logger.info(f"Found {len(files)} source grain images in {GRAINDET_V2_NOR_DIR}")
        for fp in files:
            try:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                mask, cropped = extract_grain_mask(img)
                area = int(np.sum(mask > 0))
                if area >= MIN_GRAIN_AREA:
                    source_grains.append((mask, cropped))
            except Exception:
                continue

    if len(source_grains) < 10:
        logger.info(
            f"Only {len(source_grains)} usable real grains found; "
            f"generating {fallback_count} synthetic grain shapes as fallback."
        )
        for _ in range(fallback_count):
            source_grains.append(_generate_synthetic_grain(rng))

    logger.info(f"Total source grains available: {len(source_grains)}")
    return source_grains


# ── Augmentation helpers ───────────────────────────────────────────────────────

def _rotate_image(img: np.ndarray, mask: np.ndarray,
                  angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate both image and mask around their centre, expanding canvas to fit."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a) + 2
    new_h = int(h * cos_a + w * sin_a) + 2
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    img_r = cv2.warpAffine(img, M, (new_w, new_h),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_r = cv2.warpAffine(mask, M, (new_w, new_h),
                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    ys, xs = np.where(mask_r > 0)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return mask_r[y0:y1, x0:x1], img_r[y0:y1, x0:x1]
    return mask_r, img_r


def _apply_photometric(grain_bgr: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                       brightness: float, contrast: float,
                       blur_sigma: float) -> np.ndarray:
    """Apply brightness/contrast multiply + Gaussian blur within mask region only."""
    out = grain_bgr.astype(np.float32)
    out = out * contrast + (brightness - 128.0) * (contrast * 0.5 + 0.5)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if blur_sigma > 0.05:
        k = max(3, int(round(blur_sigma * 4)) | 1)
        blurred = cv2.GaussianBlur(out, (k, k), blur_sigma)
        m = (mask > 0).astype(np.uint8)[:, :, None]
        out = out * (1 - m) + blurred * m
        out = out.astype(np.uint8)
    return out


def _build_background(rng: np.random.Generator, size: int, textured: bool) -> np.ndarray:
    """Create a synthetic background canvas (blank or textured paper-like)."""
    if textured:
        base_val = int(rng.integers(220, 250))
        bg = np.full((size, size, 3), base_val, dtype=np.uint8)
        noise = rng.normal(0, 10, bg.shape).astype(np.float32)
        bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        for _ in range(3):
            k = int(rng.integers(size // 10, size // 4))
            bg = cv2.GaussianBlur(bg, (k | 1, k | 1), 0)
        vignette = np.zeros((size, size), dtype=np.float32)
        cv2.circle(vignette, (size // 2, size // 2), int(size * 0.55), 1.0, -1)
        vignette = cv2.GaussianBlur(vignette, (size // 6 | 1, size // 6 | 1), 0)
        vignette = vignette[:, :, None] * 0.18 + 0.82
        bg = np.clip(bg.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
    else:
        base_val = int(rng.integers(235, 252))
        bg = np.full((size, size, 3), base_val, dtype=np.uint8)
    return bg


# ── 2. Synthesize a single image ───────────────────────────────────────────────

def synthesize_image(source_grains: List[Tuple[np.ndarray, np.ndarray]],
                     num_grains: int,
                     mode: str,
                     img_size: int = 640,
                     rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, List[Dict]]:
    """
    Create a single synthetic rice-grain image with per-grain COCO annotations.

    Packs the requested number of grains into an ``img_size × img_size`` canvas
    while respecting the density mode (sparse, touching, overlapping, dense,
    very_dense, foreign).  Applies random rotation (0–360°), scale (±20%),
    brightness (±30%), contrast (±20%), and Gaussian blur (σ 0–1.0) to each grain.

    Parameters
    ----------
    source_grains : List of (mask, grain_bgr) tuples from :func:`_load_source_grains`
                    (or :func:`_generate_synthetic_grain`).
    num_grains    : Base number of grains to place. Actual count may be higher for
                    ``touching`` / ``overlapping`` modes to accommodate the paired
                    grains on top of isolated ones.
    mode          : One of "sparse", "touching", "overlapping", "dense", "very_dense",
                    or "foreign".
    img_size      : Output canvas size in pixels (square). Defaults to 640.
    rng           : Optional ``numpy.random.Generator`` instance for reproducibility.

    Returns
    -------
    canvas : np.ndarray
        The composited BGR image, shape ``(img_size, img_size, 3)``, uint8.
    annotations : List of Dict
        Per-instance annotation dictionaries, each with keys:
        ``segmentation`` (simplified polygon), ``bbox`` [x,y,w,h], ``area``,
        ``category_id`` (1 for rice, 2 for foreign matter), ``iscrowd`` (0).
    """
    if rng is None:
        rng = np.random.default_rng()

    cfg = MODE_CONFIG[mode]
    annotations: List[Dict] = []
    placed_masks: List[Tuple[int, int, np.ndarray]] = []

    textured_bg = rng.random() < 0.55
    canvas = _build_background(rng, img_size, textured_bg)
    occ_mask = np.zeros((img_size, img_size), dtype=np.uint8)

    total_grains = num_grains
    touch_pairs = 0
    overlap_pairs = 0
    if mode == "touching":
        tmin, tmax = cfg["touch_pairs"]
        touch_pairs = int(rng.integers(tmin, tmax + 1))
        total_grains += touch_pairs
    elif mode == "overlapping":
        omin, omax = cfg["overlap_pairs"]
        overlap_pairs = int(rng.integers(omin, omax + 1))
        total_grains += overlap_pairs

    max_attempts = total_grains * 80
    placed = 0
    attempts = 0
    paired_placements: List[int] = []

    while placed < total_grains and attempts < max_attempts:
        attempts += 1

        if mode == "foreign" and placed < max(1, num_grains // 4):
            category_id = 2
            mask_src, grain_src = _generate_foreign_matter_piece(rng)
        else:
            category_id = 1
            src_idx = int(rng.integers(0, len(source_grains)))
            mask_src, grain_src = source_grains[src_idx]

        angle = float(rng.uniform(0.0, 360.0))
        scale = float(rng.uniform(0.8, 1.2))
        brightness = float(rng.uniform(-30.0, 30.0))
        contrast = float(rng.uniform(0.8, 1.2))
        blur_sigma = float(rng.uniform(0.0, 1.0))

        mask_t, grain_t = _rotate_image(grain_src, mask_src, angle)
        if scale != 1.0:
            nh = max(8, int(mask_t.shape[0] * scale))
            nw = max(8, int(mask_t.shape[1] * scale))
            grain_t = cv2.resize(grain_t, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask_t = cv2.resize(mask_t, (nw, nh), interpolation=cv2.INTER_NEAREST)

        grain_t = _apply_photometric(grain_t, mask_t, rng, brightness, contrast, blur_sigma)

        gh, gw = mask_t.shape[:2]
        margin = 6

        if placed in paired_placements or (
            (touch_pairs > 0 or overlap_pairs > 0) and
            len(paired_placements) < (touch_pairs + overlap_pairs) * 2 and
            len(placed_masks) > 0
        ):
            if placed in paired_placements:
                pass
            else:
                partner_idx = int(rng.integers(0, len(placed_masks)))
                px, py, pmask = placed_masks[partner_idx]
                ph, pw = pmask.shape[:2]
                offset_mode = "touching" if touch_pairs > len(
                    [p for p in paired_placements if p < placed]) else "overlap"
                if offset_mode == "touching":
                    dx = int(rng.integers(max(-pw // 2, -gw // 2),
                                          min(pw // 2, gw // 2)))
                    dy = int(rng.integers(max(-ph // 2, -gh // 2),
                                          min(ph // 2, gh // 2)))
                else:
                    dx = int(rng.integers(-gw // 3, gw // 3))
                    dy = int(rng.integers(-gh // 3, gh // 3))
                cx = px + pw // 2 + dx
                cy = py + ph // 2 + dy
                x = cx - gw // 2
                y = cy - gh // 2
                paired_placements.append(partner_idx)
                paired_placements.append(placed)
        else:
            x = int(rng.integers(margin, max(margin + 1, img_size - gw - margin)))
            y = int(rng.integers(margin, max(margin + 1, img_size - gh - margin)))

        if x < 0 or y < 0 or x + gw > img_size or y + gh > img_size:
            continue

        local_mask = mask_t > 0
        region = occ_mask[y:y + gh, x:x + gw]
        overlap_amt = float(np.sum(np.logical_and(local_mask, region > 0)))
        local_area = float(np.sum(local_mask))
        if local_area < MIN_GRAIN_AREA:
            continue

        overlap_ratio = overlap_amt / local_area if local_area > 0 else 1.0
        max_overlap = 0.05 if mode == "sparse" else (
            0.15 if mode == "touching" else (
                0.55 if mode == "overlapping" else (
                    0.35 if mode in ("dense", "foreign") else 0.55
                )
            )
        )

        if placed in paired_placements:
            pass
        elif overlap_ratio > max_overlap:
            continue

        dest = canvas[y:y + gh, x:x + gw].astype(np.float32)
        m3 = local_mask.astype(np.float32)[:, :, None]
        canvas[y:y + gh, x:x + gw] = np.clip(
            dest * (1 - m3) + grain_t.astype(np.float32) * m3, 0, 255
        ).astype(np.uint8)

        occ_mask[y:y + gh, x:x + gw] = np.where(
            local_mask, 255, occ_mask[y:y + gh, x:x + gw]
        )

        full_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        full_mask[y:y + gh, x:x + gw] = np.where(local_mask, 255, 0).astype(np.uint8)

        polygon = _mask_to_polygon(full_mask)
        if polygon is None:
            continue
        bbox, area = _mask_to_bbox_area(full_mask)
        if area < MIN_GRAIN_AREA:
            continue

        annotations.append({
            "category_id": category_id,
            "segmentation": [polygon],
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })

        placed_masks.append((x, y, mask_t))
        placed += 1

    return canvas, annotations


# ── 3. Dataset generator ───────────────────────────────────────────────────────

def generate_dataset(output_dir: Path, config: Dict) -> Dict:
    """
    Generate the full synthetic dataset and save COCO annotations + images.

    Parameters
    ----------
    output_dir : Path
        Root of the COCO output directory (datasets/processed/coco). Images are
        written into ``output_dir/images/synthetic/`` and JSON into
        ``output_dir/annotations/instances_synthetic.json``.
    config : Dict
        Generation configuration containing:
        ``num_isolated``, ``num_touching``, ``num_overlapping``, ``num_dense``,
        ``num_very_dense``, ``num_foreign`` (all int counts of images per mode),
        plus ``seed`` (int) and optional ``img_size`` (int, default 640).

    Returns
    -------
    stats : Dict
        Summary statistics: total images, total annotations, per-mode counts,
        per-category instance counts, image paths used.
    """
    seed = int(config.get("seed", 42))
    img_size = int(config.get("img_size", DEFAULT_IMG_SIZE))
    rng = np.random.default_rng(seed)
    cv2.setRNGSeed(seed & 0xFFFFFFFF)

    images_dir = output_dir / "images" / "synthetic"
    ann_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    source_grains = _load_source_grains(rng)

    mode_plan: List[Tuple[str, int]] = []
    mode_plan.extend([("sparse",      int(config.get("num_isolated", 500)))])
    mode_plan.extend([("touching",    int(config.get("num_touching", 500)))])
    mode_plan.extend([("overlapping", int(config.get("num_overlapping", 500)))])
    mode_plan.extend([("dense",       int(config.get("num_dense", 200)))])
    mode_plan.extend([("very_dense",  int(config.get("num_very_dense", 100)))])
    mode_plan.extend([("foreign",     int(config.get("num_foreign", 100)))])

    coco_dict = {
        "info": {
            "description": "Synthetic rice grain instance segmentation dataset",
            "version": "1.0",
            "year": 2026,
            "contributor": "rice-quality-ai generate_synthetic.py",
            "date_created": "2026-09-28",
            "seed": seed,
        },
        "licenses": [{"id": 1, "name": "Proprietary", "url": ""}],
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_id = 1
    ann_id = 1
    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "per_mode_images": Counter(),
        "per_category_instances": Counter(),
        "per_mode_instances": {},
    }
    per_mode_instances: Dict[str, int] = {}

    for mode_name, count in mode_plan:
        if count <= 0:
            continue
        per_mode_instances[mode_name] = 0
        logger.info(f"Generating mode='{mode_name}' — {count} images")
        mc = MODE_CONFIG[mode_name]

        for i in range(count):
            if (i + 1) % 100 == 0:
                logger.info(f"  {mode_name}: {i + 1}/{count}")

            if mode_name in ("sparse", "foreign"):
                ng = int(rng.integers(mc["num_min"], mc["num_max"] + 1))
            elif mode_name == "touching":
                tmin, tmax = mc["touch_pairs"]
                ng = int(rng.integers(mc["num_min"], mc["num_max"] + 1))
                _ = (tmin, tmax)
            elif mode_name == "overlapping":
                omin, omax = mc["overlap_pairs"]
                ng = int(rng.integers(mc["num_min"], mc["num_max"] + 1))
                _ = (omin, omax)
            else:
                ng = int(rng.integers(mc["num_min"], mc["num_max"] + 1))

            canvas, anns = synthesize_image(
                source_grains=source_grains,
                num_grains=ng,
                mode=mode_name,
                img_size=img_size,
                rng=rng,
            )

            fname = f"synthetic_{mode_name}_{image_id:06d}.png"
            fpath = images_dir / fname
            ok = cv2.imwrite(str(fpath), canvas)
            if not ok:
                logger.warning(f"Failed to write {fpath}, skipping")
                continue

            coco_dict["images"].append({
                "id": image_id,
                "file_name": fname,
                "width": int(canvas.shape[1]),
                "height": int(canvas.shape[0]),
            })

            for a in anns:
                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    **a,
                })
                per_mode_instances[mode_name] += 1
                stats["per_category_instances"][
                    CATEGORIES[a["category_id"] - 1]["name"]
                ] += 1
                ann_id += 1

            stats["per_mode_images"][mode_name] += 1
            stats["total_images"] += 1
            stats["total_annotations"] += len(anns)
            image_id += 1

    stats["per_mode_instances"] = per_mode_instances

    with open(ann_dir / "instances_synthetic.json", "w") as f:
        json.dump(coco_dict, f)

    json_size = (ann_dir / "instances_synthetic.json").stat().st_size / 1024 / 1024
    logger.info(
        f"Saved {stats['total_images']} images, {stats['total_annotations']} "
        f"annotations → {images_dir} | JSON {json_size:.2f} MB"
    )
    return stats


# ── 4. CLI entry point ─────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and dispatch dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic rice grain images with COCO segmentation annotations."
    )
    parser.add_argument("--num-isolated", type=int, default=500,
                        help="Number of sparse (isolated 10-20 grains) images. Default 500.")
    parser.add_argument("--num-touching", type=int, default=500,
                        help="Number of images with 2-5 touching grain pairs. Default 500.")
    parser.add_argument("--num-overlapping", type=int, default=500,
                        help="Number of images with 2-3 overlapping grain pairs. Default 500.")
    parser.add_argument("--num-dense", type=int, default=200,
                        help="Number of dense (50-100 grains) images. Default 200.")
    parser.add_argument("--num-very-dense", type=int, default=100,
                        help="Number of very dense (100-250 grains) images. Default 100.")
    parser.add_argument("--num-foreign", type=int, default=100,
                        help="Number of foreign-matter composite images. Default 100.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default 42.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE,
                        help=f"Output image size (square). Default {DEFAULT_IMG_SIZE}.")
    parser.add_argument("--output-dir", type=str, default=str(COCO_OUT_DIR),
                        help="Root COCO output directory.")
    args = parser.parse_args()

    config = {
        "num_isolated":    args.num_isolated,
        "num_touching":    args.num_touching,
        "num_overlapping": args.num_overlapping,
        "num_dense":       args.num_dense,
        "num_very_dense":  args.num_very_dense,
        "num_foreign":     args.num_foreign,
        "seed":            args.seed,
        "img_size":        args.img_size,
    }

    logger.info(
        "Generation config: "
        f"isolated={config['num_isolated']}, "
        f"touching={config['num_touching']}, "
        f"overlapping={config['num_overlapping']}, "
        f"dense={config['num_dense']}, "
        f"very_dense={config['num_very_dense']}, "
        f"foreign={config['num_foreign']}, "
        f"seed={config['seed']}, "
        f"img_size={config['img_size']}"
    )

    stats = generate_dataset(Path(args.output_dir), config)

    logger.info("===== Generation complete =====")
    for mode, im_count in stats["per_mode_images"].items():
        inst = stats["per_mode_instances"].get(mode, 0)
        logger.info(f"  {mode:<14s}: {im_count:4d} images | {inst:6d} instances")
    logger.info(f"  Total images      : {stats['total_images']}")
    logger.info(f"  Total annotations : {stats['total_annotations']}")
    logger.info(
        "  Per-category      : "
        + ", ".join(f"{k}={v}" for k, v in stats["per_category_instances"].items())
    )


if __name__ == "__main__":
    main()
