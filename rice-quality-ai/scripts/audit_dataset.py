"""
scripts/audit_dataset.py
========================
Audit the GrainSet v3 dataset (and optionally other datasets) before any training.

What it does:
  - Scans GrainSet v3 train/ and mask/ folders
  - For each mask: counts unique pixel values → determines instance vs semantic
  - Reports grains-per-image distribution
  - Detects touching grains (dilation + intersection check)
  - Flags corrupt/blank/missing files
  - Detects visual duplicates (perceptual hash via average hash)
  - Generates visual preview grid: original + mask overlay + grain IDs
  - Exports datasets/inspection/audit_report.json

Usage:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --dataset grainset_v3 --preview-count 20
    python scripts/audit_dataset.py --sample 500   # audit random 500 pairs
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Dataset paths ──────────────────────────────────────────────────────────────
GRAINSET_V3_BASE = PROJECT_ROOT / "data" / "raw" / "grainset_rice_v3" / "rice (1)" / "rice"
GRAINDET_V2_BASE = PROJECT_ROOT / "data" / "raw" / "graindet_rice_v2" / "rice"
INSPECTION_DIR   = PROJECT_ROOT / "datasets" / "inspection"


# ── Core analysis functions ────────────────────────────────────────────────────

def analyze_mask_type(mask_path: Path) -> str:
    """
    Inspect a mask file and determine its encoding type.

    Returns
    -------
    'instance'  – each grain has a unique pixel value (>2 unique values excl. 0)
    'semantic'  – all grains merged into one binary blob (values 0 and 255 only)
    'empty'     – mask has zero foreground pixels
    'corrupt'   – file could not be read or decoded
    """
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return "corrupt"
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        unique_vals = np.unique(mask)
        # Remove background (0)
        fg_vals = unique_vals[unique_vals > 0]
        if len(fg_vals) == 0:
            return "empty"
        if len(fg_vals) == 1:
            return "semantic"   # only one foreground value (binary mask)
        # If values are exactly {255} or {0,255} → binary semantic
        if set(fg_vals.tolist()) <= {255}:
            return "semantic"
        # Multiple distinct non-zero, non-255 values → instance
        return "instance"
    except Exception as e:
        logger.debug(f"Error reading {mask_path}: {e}")
        return "corrupt"


def count_instances(mask: np.ndarray, mask_type: str) -> int:
    """
    Count grain instances in a mask.

    For 'instance' masks: count unique non-zero pixel values.
    For 'semantic' masks: run connected components on binary foreground.
    """
    if mask_type == "instance":
        unique_vals = np.unique(mask)
        return int(np.sum(unique_vals > 0))
    elif mask_type == "semantic":
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary = (mask > 0).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # Count components with area > 50px (ignore noise)
        return int(sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 50))
    return 0


def detect_touching_grains(mask: np.ndarray, mask_type: str) -> bool:
    """
    Return True if any two grain instances are touching (share a boundary pixel).

    For semantic masks: check if any connected component touches another after dilation.
    For instance masks: dilate each instance and check intersection with others.
    """
    try:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        if mask_type == "semantic":
            binary = (mask > 0).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels <= 2:
                return False
            # Check if any two component regions are adjacent (dilate one, intersect with another)
            for i in range(1, min(num_labels, 30)):  # cap at 30 for speed
                comp_i = (labels == i).astype(np.uint8)
                dilated_i = cv2.dilate(comp_i, kernel, iterations=1)
                for j in range(i + 1, min(num_labels, 30)):
                    comp_j = (labels == j).astype(np.uint8)
                    if np.any(dilated_i & comp_j):
                        return True
            return False

        elif mask_type == "instance":
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]
            if len(instance_ids) <= 1:
                return False
            for i, id_i in enumerate(instance_ids[:20]):  # cap at 20
                comp_i = (mask == id_i).astype(np.uint8)
                dilated_i = cv2.dilate(comp_i, kernel, iterations=1)
                for id_j in instance_ids[i + 1:20]:
                    comp_j = (mask == id_j).astype(np.uint8)
                    if np.any(dilated_i & comp_j):
                        return True
            return False

    except Exception:
        return False

    return False


def average_hash(image: np.ndarray, hash_size: int = 8) -> int:
    """Compute a simple average perceptual hash for duplicate detection."""
    resized = cv2.resize(image, (hash_size, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    avg = gray.mean()
    bits = (gray > avg).flatten()
    return int(sum(b << i for i, b in enumerate(bits)))


def generate_preview(
    img_path: Path,
    mask_path: Path,
    out_path: Path,
    mask_type: str = "semantic",
) -> bool:
    """
    Generate a 4-panel preview: original | mask | overlay | grain ID labels.

    Returns True on success.
    """
    try:
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            return False

        h, w = img.shape[:2]
        if mask.ndim == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask.copy()

        # Panel 1: original
        p1 = img.copy()

        # Panel 2: mask visualised
        if mask_type == "instance":
            # Colour-code each instance
            unique_ids = np.unique(mask_gray)
            unique_ids = unique_ids[unique_ids > 0]
            p2 = np.zeros((h, w, 3), dtype=np.uint8)
            for uid in unique_ids:
                hue = int((uid * 37) % 180)
                color = cv2.cvtColor(
                    np.array([[[hue, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
                )[0, 0].tolist()
                p2[mask_gray == uid] = color
        else:
            p2_gray = (mask_gray > 0).astype(np.uint8) * 255
            p2 = cv2.cvtColor(p2_gray, cv2.COLOR_GRAY2BGR)

        # Panel 3: overlay
        p3 = img.copy()
        alpha_mask = (mask_gray > 0)
        p3[alpha_mask] = (p3[alpha_mask] * 0.5 + p2[alpha_mask] * 0.5).astype(np.uint8)

        # Panel 4: grain ID labels
        p4 = img.copy()
        num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
            (mask_gray > 0).astype(np.uint8), connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                cv2.putText(p4, str(i), (cx - 6, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Combine 4 panels in a 2×2 grid
        row1 = np.hstack([p1, p2])
        row2 = np.hstack([p3, p4])
        grid = np.vstack([row1, row2])

        # Resize to max 1200px wide for viewing
        max_w = 1200
        if grid.shape[1] > max_w:
            scale = max_w / grid.shape[1]
            grid = cv2.resize(grid, (max_w, int(grid.shape[0] * scale)))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid)
        return True

    except Exception as e:
        logger.debug(f"Preview error {img_path.name}: {e}")
        return False


# ── Main audit function ────────────────────────────────────────────────────────

def audit_grainset_v3(
    sample: Optional[int] = None,
    preview_count: int = 20,
) -> Dict:
    """
    Audit the GrainSet v3 dataset.

    Parameters
    ----------
    sample        : If set, randomly sample this many image/mask pairs.
    preview_count : Number of preview images to generate.

    Returns
    -------
    dict: Full audit report.
    """
    train_dir = GRAINSET_V3_BASE / "train"
    mask_dir  = GRAINSET_V3_BASE / "mask"
    test_dir  = GRAINSET_V3_BASE / "test"

    if not train_dir.exists():
        logger.error(f"Train directory not found: {train_dir}")
        return {"error": f"Train directory not found: {train_dir}"}

    if not mask_dir.exists():
        logger.error(f"Mask directory not found: {mask_dir}")
        return {"error": f"Mask directory not found: {mask_dir}"}

    logger.info(f"Scanning training images: {train_dir}")
    all_train_images = sorted(
        list(train_dir.glob("*.png")) +
        list(train_dir.glob("*.jpg")) +
        list(train_dir.glob("*.jpeg"))
    )
    logger.info(f"Found {len(all_train_images)} training images")

    if sample and sample < len(all_train_images):
        logger.info(f"Sampling {sample} images for audit")
        all_train_images = random.sample(all_train_images, sample)

    # Audit report structure
    report = {
        "dataset": "GrainSet Rice v3",
        "paths": {
            "train": str(train_dir),
            "mask": str(mask_dir),
            "test": str(test_dir),
        },
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_train_images": len(all_train_images),
        "total_masks_scanned": 0,
        "mask_type": "unknown",
        "mask_type_votes": {"instance": 0, "semantic": 0, "empty": 0, "corrupt": 0},
        "grain_count_distribution": {},
        "touching_grain_count": 0,
        "missing_masks": [],
        "corrupt_files": [],
        "empty_masks": [],
        "duplicate_pairs": [],
        "preview_images_generated": 0,
        "per_image_stats": [],
        "summary": {},
    }

    # ── File pair discovery ────────────────────────────────────────────────────
    paired = []
    missing_masks = []
    for img_path in all_train_images:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            # Try common alternate extensions
            for ext in [".png", ".jpg", ".jpeg"]:
                alt = mask_dir / (img_path.stem + ext)
                if alt.exists():
                    mask_path = alt
                    break
        if not mask_path.exists():
            missing_masks.append(img_path.name)
        else:
            paired.append((img_path, mask_path))

    report["missing_masks"] = missing_masks
    logger.info(f"Paired image/mask pairs: {len(paired)} | Missing masks: {len(missing_masks)}")

    # ── Per-file analysis ──────────────────────────────────────────────────────
    hash_map: Dict[int, str] = {}
    grain_counts = []
    touching_count = 0
    previews_generated = 0
    preview_out_dir = INSPECTION_DIR / "grainset_v3_previews"

    for idx, (img_path, mask_path) in enumerate(paired):
        if idx % 100 == 0:
            logger.info(f"  Auditing {idx}/{len(paired)}...")

        mask_type = analyze_mask_type(mask_path)
        report["mask_type_votes"][mask_type] += 1

        if mask_type in ("corrupt", "empty"):
            if mask_type == "corrupt":
                report["corrupt_files"].append(mask_path.name)
            elif mask_type == "empty":
                report["empty_masks"].append(mask_path.name)
            continue

        report["total_masks_scanned"] += 1

        # Load mask for analysis
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            report["corrupt_files"].append(mask_path.name)
            continue
        if mask.ndim == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        n_instances = count_instances(mask_gray, mask_type)
        grain_counts.append(n_instances)
        has_touching = detect_touching_grains(mask_gray, mask_type)
        if has_touching:
            touching_count += 1

        # Duplicate detection
        img = cv2.imread(str(img_path))
        if img is not None:
            h = average_hash(img)
            if h in hash_map:
                report["duplicate_pairs"].append((img_path.name, hash_map[h]))
            else:
                hash_map[h] = img_path.name

        # Per-image entry
        report["per_image_stats"].append({
            "image": img_path.name,
            "mask": mask_path.name,
            "mask_type": mask_type,
            "grain_count": n_instances,
            "has_touching": bool(has_touching),
        })

        # Generate preview for first N
        if previews_generated < preview_count:
            out_path = preview_out_dir / f"preview_{idx:04d}_{img_path.stem}.jpg"
            if generate_preview(img_path, mask_path, out_path, mask_type):
                previews_generated += 1

    # ── Determine dominant mask type ───────────────────────────────────────────
    votes = report["mask_type_votes"]
    dominant = max(votes, key=lambda k: votes[k])
    report["mask_type"] = dominant
    report["touching_grain_count"] = touching_count
    report["preview_images_generated"] = previews_generated

    # ── Grain count distribution ───────────────────────────────────────────────
    if grain_counts:
        counter = Counter()
        for cnt in grain_counts:
            bucket = (cnt // 5) * 5  # bucket by 5s
            counter[bucket] += 1
        report["grain_count_distribution"] = {str(k): v for k, v in sorted(counter.items())}

    # ── Summary ────────────────────────────────────────────────────────────────
    report["summary"] = {
        "mask_type_verdict": dominant,
        "total_pairs_analyzed": len(paired),
        "corrupt_count": len(report["corrupt_files"]),
        "empty_count": len(report["empty_masks"]),
        "missing_mask_count": len(missing_masks),
        "duplicate_pair_count": len(report["duplicate_pairs"]),
        "touching_image_count": touching_count,
        "touching_image_fraction": round(touching_count / max(len(paired), 1), 3),
        "grain_count_stats": {
            "min": int(min(grain_counts)) if grain_counts else 0,
            "max": int(max(grain_counts)) if grain_counts else 0,
            "mean": round(float(np.mean(grain_counts)), 2) if grain_counts else 0,
            "median": round(float(np.median(grain_counts)), 2) if grain_counts else 0,
        },
        "recommendation": (
            "Use instance masks directly as COCO annotations."
            if dominant == "instance"
            else "Run connected components on semantic masks to extract per-grain instances."
        ),
    }

    # ── Save report ────────────────────────────────────────────────────────────
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = INSPECTION_DIR / "audit_report.json"

    # Truncate per_image_stats if very large (keep first 1000)
    if len(report["per_image_stats"]) > 1000:
        report["per_image_stats"] = report["per_image_stats"][:1000]
        report["per_image_stats_note"] = "Truncated to first 1000 entries in JSON output."

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Audit report saved to: {report_path}")
    return report


def audit_graindet_v2() -> Dict:
    """Quick audit of GrainDet v2 classification dataset."""
    if not GRAINDET_V2_BASE.exists():
        return {"error": f"GrainDet v2 not found at {GRAINDET_V2_BASE}"}

    classes = {}
    for class_dir in sorted(GRAINDET_V2_BASE.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            classes[class_dir.name] = len(images)

    total = sum(classes.values())
    report = {
        "dataset": "GrainDet Rice v2",
        "path": str(GRAINDET_V2_BASE),
        "total_images": total,
        "class_distribution": classes,
        "note": "Classification dataset only — useful for synthetic data generation.",
    }

    report_path = INSPECTION_DIR / "graindet_v2_audit.json"
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"GrainDet v2 audit: {total} images across {len(classes)} classes")
    return report


def print_summary(report: Dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("DATASET AUDIT SUMMARY")
    print("=" * 60)
    s = report.get("summary", {})
    print(f"Dataset:          {report.get('dataset', '?')}")
    print(f"Pairs analyzed:   {s.get('total_pairs_analyzed', '?')}")
    print(f"Mask type:        {s.get('mask_type_verdict', '?').upper()}")
    print(f"Corrupt files:    {s.get('corrupt_count', 0)}")
    print(f"Empty masks:      {s.get('empty_count', 0)}")
    print(f"Missing masks:    {s.get('missing_mask_count', 0)}")
    print(f"Duplicate pairs:  {s.get('duplicate_pair_count', 0)}")
    print(f"Images w/ touching grains: {s.get('touching_image_count', 0)} "
          f"({s.get('touching_image_fraction', 0)*100:.1f}%)")
    stats = s.get("grain_count_stats", {})
    print(f"Grain count:      min={stats.get('min',0)}  "
          f"max={stats.get('max',0)}  "
          f"mean={stats.get('mean',0)}  "
          f"median={stats.get('median',0)}")
    print(f"\nRecommendation: {s.get('recommendation', '?')}")
    print("=" * 60)
    print(f"Full report:      {INSPECTION_DIR / 'audit_report.json'}")
    print(f"Previews:         {INSPECTION_DIR / 'grainset_v3_previews'}/")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit GrainSet v3 and GrainDet v2 datasets.")
    parser.add_argument("--dataset", choices=["grainset_v3", "graindet_v2", "all"],
                        default="all", help="Which dataset to audit")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N image/mask pairs (default: all)")
    parser.add_argument("--preview-count", type=int, default=20,
                        help="Number of preview images to generate (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.dataset in ("grainset_v3", "all"):
        logger.info("Starting GrainSet v3 audit...")
        t0 = time.time()
        report = audit_grainset_v3(
            sample=args.sample,
            preview_count=args.preview_count,
        )
        if "error" not in report:
            print_summary(report)
            logger.info(f"GrainSet v3 audit completed in {time.time()-t0:.1f}s")

    if args.dataset in ("graindet_v2", "all"):
        logger.info("Starting GrainDet v2 audit...")
        graindet_report = audit_graindet_v2()
        if "error" not in graindet_report:
            print("\nGrainDet v2:")
            for cls, count in graindet_report.get("class_distribution", {}).items():
                print(f"  {cls}: {count} images")
            print(f"  Total: {graindet_report.get('total_images', 0)}")


if __name__ == "__main__":
    main()
