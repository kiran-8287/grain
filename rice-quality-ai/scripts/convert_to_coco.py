"""
scripts/convert_to_coco.py
==========================
Convert GrainSet v3 image/mask pairs to COCO instance segmentation format.

Handles both mask types:
  - SEMANTIC masks: run connected components per binary mask to extract individual
    grain regions, then convert each region to a polygon annotation.
  - INSTANCE masks: each unique pixel value is a separate grain — extract polygon
    directly from each unique-value region.

Output structure:
  datasets/processed/coco/
    annotations/
      instances_train.json
      instances_val.json
      instances_test.json
    images/
      train/   (symlinked or copied)
      val/
      test/

Usage:
    python scripts/convert_to_coco.py
    python scripts/convert_to_coco.py --mask-type semantic
    python scripts/convert_to_coco.py --mask-type instance
    python scripts/convert_to_coco.py --split-file datasets/splits/train.txt --split train
"""

import argparse
import json
import logging
import shutil
import sys
import time
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
GRAINSET_V3_BASE = PROJECT_ROOT / "data" / "raw" / "grainset_rice_v3" / "rice (1)" / "rice"
COCO_OUT_DIR     = PROJECT_ROOT / "datasets" / "processed" / "coco"
SPLITS_DIR       = PROJECT_ROOT / "datasets" / "splits"
INSPECTION_DIR   = PROJECT_ROOT / "datasets" / "inspection"

CATEGORIES = [
    {"id": 1, "name": "rice_grain",     "supercategory": "rice"},
    {"id": 2, "name": "foreign_matter", "supercategory": "other"},
]

MIN_GRAIN_AREA = 50      # px² — smaller regions are noise
SIMPLIFY_EPSILON = 2.0   # Douglas-Peucker epsilon for polygon simplification


# ── Polygon extraction ─────────────────────────────────────────────────────────

def mask_region_to_polygon(binary_mask: np.ndarray) -> Optional[List[float]]:
    """
    Convert a binary mask region to a COCO polygon (flat list of xy coords).

    Returns None if the region is too small or no contour is found.
    """
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_GRAIN_AREA:
        return None

    # Simplify contour to reduce JSON size while preserving shape
    epsilon = SIMPLIFY_EPSILON
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Need at least 3 points for a valid polygon
    if len(approx) < 3:
        return None

    # Flatten to [x1, y1, x2, y2, ...]
    polygon = approx.flatten().tolist()
    return polygon


def extract_annotations_from_semantic_mask(
    mask: np.ndarray,
    image_id: int,
    start_ann_id: int,
    category_id: int = 1,
) -> Tuple[List[Dict], int]:
    """
    Extract COCO annotations from a semantic (binary) mask using connected components.

    Returns (list_of_annotations, next_ann_id)
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    annotations = []
    ann_id = start_ann_id

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_GRAIN_AREA:
            continue

        region_mask = (labels == label).astype(np.uint8)
        polygon = mask_region_to_polygon(region_mask)
        if polygon is None:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [polygon],
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0,
        })
        ann_id += 1

    return annotations, ann_id


def extract_annotations_from_instance_mask(
    mask: np.ndarray,
    image_id: int,
    start_ann_id: int,
    category_id: int = 1,
) -> Tuple[List[Dict], int]:
    """
    Extract COCO annotations from an instance mask (unique pixel value per grain).

    Returns (list_of_annotations, next_ann_id)
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    annotations = []
    ann_id = start_ann_id

    for uid in unique_ids:
        region_mask = (mask == uid).astype(np.uint8)
        area = int(np.sum(region_mask))
        if area < MIN_GRAIN_AREA:
            continue

        polygon = mask_region_to_polygon(region_mask)
        if polygon is None:
            continue

        ys, xs = np.where(region_mask > 0)
        x, y = int(xs.min()), int(ys.min())
        w = int(xs.max()) - x + 1
        h = int(ys.max()) - y + 1

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [polygon],
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0,
        })
        ann_id += 1

    return annotations, ann_id


# ── COCO JSON builder ──────────────────────────────────────────────────────────

def build_coco_dataset(
    image_mask_pairs: List[Tuple[Path, Path]],
    mask_type: str,
    out_dir: Path,
    split_name: str,
    copy_images: bool = True,
) -> Dict:
    """
    Build a COCO JSON annotation file from image/mask pairs.

    Parameters
    ----------
    image_mask_pairs : List of (image_path, mask_path) tuples
    mask_type        : 'semantic' or 'instance'
    out_dir          : Output base directory (datasets/processed/coco)
    split_name       : 'train', 'val', or 'test'
    copy_images      : If True, copy images to coco/images/<split_name>/

    Returns
    -------
    coco_dict: The full COCO JSON structure
    """
    images_out_dir = out_dir / "images" / split_name
    images_out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    coco_dict = {
        "info": {
            "description": f"Rice Grain Instance Segmentation — GrainSet v3 {split_name}",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    skipped = 0

    for img_id, (img_path, mask_path) in enumerate(image_mask_pairs, start=1):
        if img_id % 500 == 0:
            logger.info(f"  Processing {img_id}/{len(image_mask_pairs)}...")

        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot read image: {img_path.name}")
            skipped += 1
            continue

        h, w = img.shape[:2]

        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            logger.warning(f"Cannot read mask: {mask_path.name}")
            skipped += 1
            continue

        # Register image entry
        coco_dict["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        # Extract annotations
        if mask_type == "instance":
            anns, ann_id = extract_annotations_from_instance_mask(mask, img_id, ann_id)
        else:
            anns, ann_id = extract_annotations_from_semantic_mask(mask, img_id, ann_id)

        coco_dict["annotations"].extend(anns)

        # Copy image
        if copy_images:
            dest = images_out_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

    logger.info(
        f"Split '{split_name}': {len(coco_dict['images'])} images, "
        f"{len(coco_dict['annotations'])} annotations, "
        f"{skipped} skipped"
    )

    # Save JSON
    ann_path = ann_dir / f"instances_{split_name}.json"
    with open(ann_path, "w") as f:
        json.dump(coco_dict, f)  # No indent for size
    logger.info(f"Saved: {ann_path} ({ann_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return coco_dict


# ── Mask type detection ────────────────────────────────────────────────────────

def detect_mask_type_from_report() -> Optional[str]:
    """Read mask type from audit report if available."""
    report_path = INSPECTION_DIR / "audit_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        mask_type = report.get("mask_type", "unknown")
        if mask_type in ("semantic", "instance"):
            logger.info(f"Using mask type from audit report: {mask_type}")
            return mask_type
    return None


def detect_mask_type_from_sample(mask_dir: Path, n: int = 20) -> str:
    """Sample N masks and determine type by majority vote."""
    masks = list(mask_dir.glob("*.png"))[:n]
    votes = {"semantic": 0, "instance": 0, "empty": 0, "corrupt": 0}
    for mp in masks:
        mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if mask is None:
            votes["corrupt"] += 1
            continue
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        unique = np.unique(mask)
        fg = unique[unique > 0]
        if len(fg) == 0:
            votes["empty"] += 1
        elif set(fg.tolist()) <= {255}:
            votes["semantic"] += 1
        else:
            votes["instance"] += 1
    dominant = max(votes, key=lambda k: votes[k])
    logger.info(f"Auto-detected mask type: {dominant} (votes: {votes})")
    return dominant


# ── Split loading ──────────────────────────────────────────────────────────────

def load_split_file(split_file: Path) -> List[str]:
    """Load filenames from a split .txt file."""
    if not split_file.exists():
        return []
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert GrainSet v3 to COCO format.")
    parser.add_argument("--mask-type", choices=["semantic", "instance", "auto"],
                        default="auto", help="Mask encoding type")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to split .txt file (overrides auto-split)")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None,
                        help="Which split to process (use with --split-file)")
    parser.add_argument("--no-copy-images", action="store_true",
                        help="Skip copying images to output directory")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit total images (for testing)")
    args = parser.parse_args()

    train_dir = GRAINSET_V3_BASE / "train"
    mask_dir  = GRAINSET_V3_BASE / "mask"
    test_dir  = GRAINSET_V3_BASE / "test"

    if not train_dir.exists():
        logger.error(f"Train directory not found: {train_dir}")
        sys.exit(1)

    # ── Determine mask type ────────────────────────────────────────────────────
    if args.mask_type == "auto":
        mask_type = detect_mask_type_from_report()
        if mask_type is None:
            mask_type = detect_mask_type_from_sample(mask_dir)
    else:
        mask_type = args.mask_type

    logger.info(f"Using mask type: {mask_type}")

    # ── Collect all image/mask pairs ───────────────────────────────────────────
    all_images = sorted(
        list(train_dir.glob("*.png")) +
        list(train_dir.glob("*.jpg"))
    )
    if args.max_images:
        all_images = all_images[:args.max_images]

    paired = []
    for img_path in all_images:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            paired.append((img_path, mask_path))

    logger.info(f"Total valid pairs: {len(paired)}")

    # ── Load splits if available ───────────────────────────────────────────────
    if args.split_file and args.split:
        split_names = load_split_file(Path(args.split_file))
        split_set = set(split_names)
        split_pairs = [(ip, mp) for ip, mp in paired if ip.name in split_set]
        logger.info(f"Processing '{args.split}' split: {len(split_pairs)} pairs")
        build_coco_dataset(split_pairs, mask_type, COCO_OUT_DIR, args.split,
                           copy_images=not args.no_copy_images)
    else:
        # Auto-split: 70/15/15
        n = len(paired)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)

        import random
        random.seed(42)
        shuffled = paired[:]
        random.shuffle(shuffled)

        train_pairs = shuffled[:n_train]
        val_pairs   = shuffled[n_train:n_train + n_val]
        test_pairs  = shuffled[n_train + n_val:]

        logger.info(f"Auto-split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

        t0 = time.time()
        build_coco_dataset(train_pairs, mask_type, COCO_OUT_DIR, "train",
                           copy_images=not args.no_copy_images)
        build_coco_dataset(val_pairs,   mask_type, COCO_OUT_DIR, "val",
                           copy_images=not args.no_copy_images)
        build_coco_dataset(test_pairs,  mask_type, COCO_OUT_DIR, "test",
                           copy_images=not args.no_copy_images)

        # Save split file lists
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
            with open(SPLITS_DIR / f"{split_name}.txt", "w") as f:
                f.write("\n".join(ip.name for ip, _ in pairs))

        logger.info(f"Conversion completed in {time.time()-t0:.1f}s")
        logger.info(f"COCO annotations: {COCO_OUT_DIR / 'annotations'}/")


if __name__ == "__main__":
    main()
