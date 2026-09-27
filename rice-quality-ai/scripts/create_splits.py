"""
scripts/create_splits.py
========================
Create stratified train/val/test splits (70%/15%/15%) for the rice grain dataset.

Splitting strategy:
  - Stratified by estimated grain density: sparse (<20), medium (20-50), dense (>50)
  - Split at image level — no image appears in multiple splits
  - Synthetic images from the same source grain are grouped into the same split
  - Seed=42 for reproducibility

Outputs:
  datasets/splits/
    train.txt, val.txt, test.txt          — image filenames, one per line
    split_stats.json                      — detailed split statistics
  datasets/processed/coco/
    images/{train,val,test}/              — images (copied or symlinked)
    annotations/instances_{train,val,test}.json — filtered COCO annotations
  tests/datasets/
    {touching,overlapping,dense,foreign_matter,no_rice,single_grain}/
                                          — representative test category images

Usage:
    python scripts/create_splits.py
    python scripts/create_splits.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
    python scripts/create_splits.py --seed 42 --copy-images
    python scripts/create_splits.py --no-copy-images
"""

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
GRAINSET_V3_BASE = PROJECT_ROOT / "data" / "raw" / "grainset_rice_v3" / "rice (1)" / "rice"
COCO_DIR         = PROJECT_ROOT / "datasets" / "processed" / "coco"
SPLITS_DIR       = PROJECT_ROOT / "datasets" / "splits"
SYNTHETIC_DIR    = PROJECT_ROOT / "data" / "raw" / "synthetic"
TEST_DATA_DIR    = PROJECT_ROOT / "tests" / "datasets"
TEST_IMAGES_SRC  = Path(r"a:\grain\27 tests")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# ── Density estimation ─────────────────────────────────────────────────────────

def estimate_density(grain_count: int) -> str:
    """
    Classify grain count into density bucket.

    Returns
    -------
    'sparse'  – < 20 grains
    'medium'  – 20 to 50 grains (inclusive)
    'dense'   – > 50 grains
    """
    if grain_count < 20:
        return "sparse"
    elif grain_count <= 50:
        return "medium"
    else:
        return "dense"


# ── Synthetic source grouping ──────────────────────────────────────────────────

def extract_source_hash(filename: str) -> Optional[str]:
    """
    Extract a source grain hash from a synthetic image filename.

    Synthetic images are expected to encode their source in the filename, e.g.:
      synth_src_abc123_aug_001.png  → hash = abc123
      synthetic_<hash>_rot45.jpg    → hash = <hash>

    If the name doesn't match a synthetic pattern, returns None (real image).
    """
    stem = Path(filename).stem.lower()

    markers = ["synth_src_", "synthetic_", "src_", "aug_", "syn_"]
    for marker in markers:
        idx = stem.find(marker)
        if idx != -1:
            remainder = stem[idx + len(marker):]
            parts = remainder.split("_")
            if parts and parts[0]:
                return parts[0][:16]

    if "synth" in stem or "synthetic" in stem or "aug" in stem:
        return hashlib.md5(stem.encode()).hexdigest()[:12]

    return None


def is_synthetic_image(filename: str) -> bool:
    """Return True if filename suggests a synthetic/augmented image."""
    stem = Path(filename).stem.lower()
    synthetic_keywords = {"synth", "synthetic", "aug", "augment", "gen_", "generated"}
    return any(kw in stem for kw in synthetic_keywords)


# ── Grain counting helpers ─────────────────────────────────────────────────────

def count_grains_from_mask(mask_path: Path) -> int:
    """Count grain instances in a mask file (handles semantic or instance)."""
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return 0
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        unique_vals = np.unique(mask)
        fg_vals = unique_vals[unique_vals > 0]
        if len(fg_vals) == 0:
            return 0
        if set(fg_vals.tolist()) <= {255}:
            binary = (mask > 0).astype(np.uint8)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            return sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 50)
        else:
            count = 0
            for uid in fg_vals:
                area = int(np.sum((mask == uid).astype(np.uint8)))
                if area >= 50:
                    count += 1
            return count
    except Exception:
        return 0


def count_grains_from_coco(coco_annotations: Dict, image_id: int) -> int:
    """Count annotations (grains) for a given image_id in the COCO dict."""
    return sum(1 for ann in coco_annotations.get("annotations", [])
               if ann.get("image_id") == image_id)


# ── Metadata loading ───────────────────────────────────────────────────────────

def load_image_metadata(
    images_dir: Path,
    masks_dir: Optional[Path] = None,
    coco_annotations: Optional[Dict] = None,
) -> List[Dict]:
    """
    Build a list of image metadata dicts from disk and/or COCO annotations.

    Each metadata entry:
        {
            "file_name": str,
            "image_path": Path,
            "mask_path": Optional[Path],
            "grain_count": int,
            "density": "sparse" | "medium" | "dense",
            "is_synthetic": bool,
            "source_hash": Optional[str],
        }
    """
    metadata: List[Dict] = []

    coco_img_id_to_anns: Dict[int, int] = {}
    coco_filename_to_id: Dict[str, int] = {}
    if coco_annotations:
        for img in coco_annotations.get("images", []):
            coco_filename_to_id[img["file_name"]] = img["id"]
        for ann in coco_annotations.get("annotations", []):
            iid = ann["image_id"]
            coco_img_id_to_anns[iid] = coco_img_id_to_anns.get(iid, 0) + 1

    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return metadata

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info(f"Scanning {len(image_files)} images in {images_dir}")

    for img_path in image_files:
        fname = img_path.name
        mask_path = None
        grain_count = 0

        if masks_dir and masks_dir.exists():
            mask_path = masks_dir / fname
            if not mask_path.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt = masks_dir / (img_path.stem + ext)
                    if alt.exists():
                        mask_path = alt
                        break
            if not mask_path.exists():
                mask_path = None

        if fname in coco_filename_to_id:
            iid = coco_filename_to_id[fname]
            grain_count = coco_img_id_to_anns.get(iid, 0)
        elif mask_path and mask_path.exists():
            grain_count = count_grains_from_mask(mask_path)

        source_hash = extract_source_hash(fname)
        is_synth = is_synthetic_image(fname) or source_hash is not None

        metadata.append({
            "file_name": fname,
            "image_path": img_path,
            "mask_path": mask_path,
            "grain_count": grain_count,
            "density": estimate_density(grain_count),
            "is_synthetic": is_synth,
            "source_hash": source_hash,
        })

    return metadata


def collect_all_metadata() -> List[Dict]:
    """
    Collect metadata from all sources:
      1. Primary: GrainSet v3 train/ + mask/
      2. COCO annotations (if combined instances_train.json exists)
      3. Synthetic data directory (if present)
    """
    all_meta: List[Dict] = []
    seen_filenames: Set[str] = set()

    combined_coco_path = COCO_DIR / "annotations" / "instances_train.json"
    combined_coco: Optional[Dict] = None
    if combined_coco_path.exists():
        logger.info(f"Loading combined COCO annotations from: {combined_coco_path}")
        with open(combined_coco_path) as f:
            combined_coco = json.load(f)

    train_dir = GRAINSET_V3_BASE / "train"
    mask_dir = GRAINSET_V3_BASE / "mask"
    if train_dir.exists():
        logger.info("Loading metadata from GrainSet v3 train/mask directories")
        grainset_meta = load_image_metadata(train_dir, mask_dir, combined_coco)
        for m in grainset_meta:
            if m["file_name"] not in seen_filenames:
                seen_filenames.add(m["file_name"])
                all_meta.append(m)
    else:
        logger.warning(f"GrainSet v3 train directory not found: {train_dir}")

    if SYNTHETIC_DIR.exists():
        logger.info(f"Scanning synthetic data directory: {SYNTHETIC_DIR}")
        for subdir in SYNTHETIC_DIR.iterdir():
            if subdir.is_dir():
                syn_images = subdir / "images"
                syn_masks = subdir / "masks"
                if not syn_images.exists():
                    syn_images = subdir
                    syn_masks = None
                syn_meta = load_image_metadata(syn_images, syn_masks, None)
                for m in syn_meta:
                    if m["file_name"] not in seen_filenames:
                        m["is_synthetic"] = True
                        if not m["source_hash"]:
                            m["source_hash"] = hashlib.md5(subdir.name.encode()).hexdigest()[:12]
                        seen_filenames.add(m["file_name"])
                        all_meta.append(m)

    logger.info(f"Total image metadata entries collected: {len(all_meta)}")
    return all_meta


# ── Stratified splitting ───────────────────────────────────────────────────────

def stratified_split(
    image_metadata: List[Dict],
    ratios: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform a stratified train/val/test split.

    Stratification keys:
      - density bucket (sparse / medium / dense)
      - synthetic vs real

    Additionally, images sharing a source_hash (synthetic variants from the same
    source grain) are grouped into the same split.

    Parameters
    ----------
    image_metadata : List of per-image metadata dicts
    ratios         : (train_ratio, val_ratio, test_ratio) — must sum to ~1.0
    seed           : Random seed for reproducibility

    Returns
    -------
    (train_meta, val_meta, test_meta)
    """
    rng = random.Random(seed)
    train_ratio, val_ratio, test_ratio = ratios

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    hash_groups: Dict[str, List[Dict]] = defaultdict(list)
    ungrouped: List[Dict] = []

    for meta in image_metadata:
        sh = meta.get("source_hash")
        if sh and meta["is_synthetic"]:
            hash_groups[sh].append(meta)
        else:
            ungrouped.append(meta)

    items: List[Tuple[str, List[Dict]]] = []
    for m in ungrouped:
        items.append((m["file_name"], [m]))
    for sh, group in hash_groups.items():
        items.append((f"group_{sh}", group))

    def strat_key(item: Tuple[str, List[Dict]]) -> str:
        rep = item[1][0]
        return f"{rep['density']}|{'synth' if rep['is_synthetic'] else 'real'}"

    strata: Dict[str, List[Tuple[str, List[Dict]]]] = defaultdict(list)
    for it in items:
        strata[strat_key(it)].append(it)

    train_items: List[Tuple[str, List[Dict]]] = []
    val_items:   List[Tuple[str, List[Dict]]] = []
    test_items:  List[Tuple[str, List[Dict]]] = []

    for key, stratum_items in strata.items():
        rng.shuffle(stratum_items)
        n = len(stratum_items)
        if n == 0:
            continue
        n_train = max(1, int(round(n * train_ratio))) if n >= 3 else (1 if n >= 1 else 0)
        n_val   = max(1, int(round(n * val_ratio)))   if n - n_train >= 2 else (1 if n - n_train >= 1 else 0)
        n_train = min(n_train, n - n_val)
        n_val   = min(n_val, n - n_train)
        n_test  = n - n_train - n_val

        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        elif n == 3:
            n_train, n_val, n_test = 2, 1, 0

        train_items.extend(stratum_items[:n_train])
        val_items.extend(stratum_items[n_train:n_train + n_val])
        test_items.extend(stratum_items[n_train + n_val:])

    def flatten(items_list: List[Tuple[str, List[Dict]]]) -> List[Dict]:
        result = []
        for _, group in items_list:
            result.extend(group)
        return result

    train_meta = flatten(train_items)
    val_meta   = flatten(val_items)
    test_meta  = flatten(test_items)

    total = len(train_meta) + len(val_meta) + len(test_meta)
    logger.info(
        f"Split complete (seed={seed}): "
        f"train={len(train_meta)} ({len(train_meta)/max(total,1)*100:.1f}%), "
        f"val={len(val_meta)} ({len(val_meta)/max(total,1)*100:.1f}%), "
        f"test={len(test_meta)} ({len(test_meta)/max(total,1)*100:.1f}%)"
    )

    return train_meta, val_meta, test_meta


# ── Existing split loading ─────────────────────────────────────────────────────

def load_existing_splits(
    image_metadata: List[Dict],
) -> Optional[Tuple[List[Dict], List[Dict], List[Dict]]]:
    """
    If datasets/splits/{train,val,test}.txt exist, use them to partition metadata.

    Returns None if any split file is missing.
    """
    split_files = {
        "train": SPLITS_DIR / "train.txt",
        "val":   SPLITS_DIR / "val.txt",
        "test":  SPLITS_DIR / "test.txt",
    }
    if not all(p.exists() for p in split_files.values()):
        return None

    logger.info("Found existing split files — using them instead of re-splitting")

    meta_by_name = {m["file_name"]: m for m in image_metadata}
    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for split_name, path in split_files.items():
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
        for name in names:
            if name in meta_by_name:
                splits[split_name].append(meta_by_name[name])
            else:
                logger.debug(f"  {split_name}: file '{name}' listed in split but not in metadata")

    return splits["train"], splits["val"], splits["test"]


# ── COCO split annotations ─────────────────────────────────────────────────────

def create_split_annotations(
    combined_coco: Dict,
    split_filenames: Set[str],
) -> Dict:
    """
    Filter a combined COCO annotation dict to only include images (and their
    annotations) whose file_name is in split_filenames.

    Image and annotation ids are re-mapped to be contiguous starting at 1.
    """
    split_coco = {
        "info": dict(combined_coco.get("info", {})),
        "licenses": list(combined_coco.get("licenses", [])),
        "categories": list(combined_coco.get("categories", [])),
        "images": [],
        "annotations": [],
    }

    filename_set = set(split_filenames)
    old_img_id_to_new: Dict[int, int] = {}

    new_img_id = 1
    for img in combined_coco.get("images", []):
        if img["file_name"] in filename_set:
            old_img_id_to_new[img["id"]] = new_img_id
            new_img = dict(img)
            new_img["id"] = new_img_id
            split_coco["images"].append(new_img)
            new_img_id += 1

    new_ann_id = 1
    for ann in combined_coco.get("annotations", []):
        old_iid = ann["image_id"]
        if old_iid in old_img_id_to_new:
            new_ann = dict(ann)
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = old_img_id_to_new[old_iid]
            split_coco["annotations"].append(new_ann)
            new_ann_id += 1

    return split_coco


# ── Split statistics ───────────────────────────────────────────────────────────

def grain_count_stats(grain_counts: List[int]) -> Dict:
    """Compute min/max/mean/median for a list of grain counts."""
    if not grain_counts:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    arr = np.array(grain_counts, dtype=float)
    return {
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
    }


def compute_split_stats(
    train_meta: List[Dict],
    val_meta: List[Dict],
    test_meta: List[Dict],
) -> Dict:
    """Compute detailed statistics for the split."""
    all_meta = train_meta + val_meta + test_meta
    total_images = len(all_meta)

    stats: Dict = {
        "total_images": total_images,
        "train_count": len(train_meta),
        "val_count": len(val_meta),
        "test_count": len(test_meta),
        "per_split": {},
    }

    for split_name, meta_list in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        grain_counts = [m["grain_count"] for m in meta_list]
        densities = [m["density"] for m in meta_list]
        n_synth = sum(1 for m in meta_list if m["is_synthetic"])
        n_real  = len(meta_list) - n_synth

        stats["per_split"][split_name] = {
            "count": len(meta_list),
            "grain_count_stats": grain_count_stats(grain_counts),
            "density_buckets": dict(Counter(densities)),
            "synthetic_vs_real": {
                "synthetic": n_synth,
                "real": n_real,
                "ratio_synthetic": round(n_synth / max(len(meta_list), 1), 3),
            },
        }

    return stats


# ── Image deployment (copy / symlink) ──────────────────────────────────────────

def deploy_split_images(
    split_meta: List[Dict],
    split_name: str,
    copy_images: bool = True,
) -> None:
    """Copy or symlink images into datasets/processed/coco/images/<split_name>/."""
    target_dir = COCO_DIR / "images" / split_name
    target_dir.mkdir(parents=True, exist_ok=True)

    n_deployed = 0
    for meta in split_meta:
        src = meta["image_path"]
        dst = target_dir / meta["file_name"]
        if dst.exists():
            continue
        try:
            if copy_images:
                shutil.copy2(src, dst)
            else:
                if os.name == "nt":
                    import _winapi
                    try:
                        os.symlink(str(src), str(dst), target_is_directory=False)
                    except (OSError, AttributeError):
                        shutil.copy2(src, dst)
                else:
                    os.symlink(src.resolve(), dst)
            n_deployed += 1
        except Exception as e:
            logger.warning(f"Failed to deploy {meta['file_name']} to {split_name}: {e}")

    action = "Copied" if copy_images else "Linked"
    logger.info(f"{action} {n_deployed}/{len(split_meta)} images to {target_dir}")


def write_split_files(
    train_meta: List[Dict],
    val_meta: List[Dict],
    test_meta: List[Dict],
    stats: Dict,
) -> None:
    """Write datasets/splits/*.txt and split_stats.json."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, meta_list in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        path = SPLITS_DIR / f"{split_name}.txt"
        with open(path, "w") as f:
            f.write("\n".join(m["file_name"] for m in meta_list))
        logger.info(f"Wrote split file: {path} ({len(meta_list)} entries)")

    stats_path = SPLITS_DIR / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote split statistics: {stats_path}")


def write_split_coco_annotations(
    train_meta: List[Dict],
    val_meta: List[Dict],
    test_meta: List[Dict],
) -> None:
    """
    Write instances_{train,val,test}.json by filtering the combined COCO file.
    If no combined file exists, try to build minimal COCO annotations from metadata.
    """
    ann_dir = COCO_DIR / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    combined_path = COCO_DIR / "annotations" / "instances_train.json"
    combined_coco: Optional[Dict] = None
    if combined_path.exists():
        with open(combined_path) as f:
            combined_coco = json.load(f)

    for split_name, meta_list in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        filenames = {m["file_name"] for m in meta_list}
        out_path = ann_dir / f"instances_{split_name}.json"

        if combined_coco and combined_coco.get("images"):
            split_coco = create_split_annotations(combined_coco, filenames)
            if not split_coco["images"]:
                logger.warning(
                    f"Combined COCO yielded 0 images for '{split_name}'. "
                    f"Falling back to metadata-only annotations."
                )
                split_coco = build_minimal_coco_from_metadata(meta_list, split_name)
        else:
            split_coco = build_minimal_coco_from_metadata(meta_list, split_name)

        with open(out_path, "w") as f:
            json.dump(split_coco, f)
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info(
            f"Wrote COCO annotations: {out_path} "
            f"({len(split_coco['images'])} images, "
            f"{len(split_coco['annotations'])} annotations, {size_mb:.2f} MB)"
        )


def build_minimal_coco_from_metadata(meta_list: List[Dict], split_name: str) -> Dict:
    """Build a minimal COCO structure when no combined annotation file is available."""
    coco = {
        "info": {
            "description": f"Rice Grain Dataset — {split_name} (minimal, metadata only)",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "rice_grain", "supercategory": "rice"},
            {"id": 2, "name": "foreign_matter", "supercategory": "other"},
        ],
        "images": [],
        "annotations": [],
    }
    for i, m in enumerate(meta_list, start=1):
        entry = {
            "id": i,
            "file_name": m["file_name"],
            "width": 0,
            "height": 0,
        }
        try:
            img = cv2.imread(str(m["image_path"]))
            if img is not None:
                entry["height"], entry["width"] = img.shape[:2]
        except Exception:
            pass
        coco["images"].append(entry)
    return coco


# ── Test category subdirectories ───────────────────────────────────────────────

TEST_CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("touching",       ["23.", "touching"]),
    ("overlapping",    ["24.", "overlap"]),
    ("dense",          ["25.", "dense", "6 ", "7 ", "many"]),
    ("foreign_matter", ["17.", "18.", "19.", "22.", "foreign", "stone", "1 stones"]),
    ("no_rice",        ["2 No", "no rice", "wheat"]),
    ("single_grain",   ["3 One", "one clean", "single"]),
]

def create_test_category_subdirs() -> None:
    """
    Create tests/datasets/{category}/ subdirectories and copy representative
    images from "a:\\grain\\27 tests\\" into the matching categories.
    """
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_IMAGES_SRC.exists():
        logger.warning(f"Test images source not found: {TEST_IMAGES_SRC}")
        return

    src_files = sorted(
        p for p in TEST_IMAGES_SRC.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not src_files:
        logger.warning(f"No test images found in: {TEST_IMAGES_SRC}")
        return

    copied: Dict[str, int] = defaultdict(int)

    for category, keywords in TEST_CATEGORY_RULES:
        cat_dir = TEST_DATA_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for src_path in src_files:
            name_lower = src_path.name.lower()
            if any(kw.lower() in name_lower for kw in keywords):
                dst = cat_dir / src_path.name
                if not dst.exists():
                    try:
                        shutil.copy2(src_path, dst)
                        copied[category] += 1
                    except Exception as e:
                        logger.warning(f"Failed to copy {src_path.name} → {category}: {e}")

    uncategorized_dir = TEST_DATA_DIR / "uncategorized"
    uncategorized_dir.mkdir(parents=True, exist_ok=True)
    assigned: Set[str] = set()
    for category, _ in TEST_CATEGORY_RULES:
        for p in (TEST_DATA_DIR / category).iterdir():
            assigned.add(p.name)
    for src_path in src_files:
        if src_path.name not in assigned:
            dst = uncategorized_dir / src_path.name
            if not dst.exists():
                try:
                    shutil.copy2(src_path, dst)
                    copied["uncategorized"] += 1
                except Exception:
                    pass

    for cat, count in copied.items():
        if count:
            logger.info(f"  tests/datasets/{cat}/: {count} images")
    logger.info(f"Test category setup complete. Total copied: {sum(copied.values())}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits for the rice grain dataset."
    )
    parser.add_argument("--train-ratio", type=float, default=0.70,
                        help="Fraction of images for training (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of images for validation (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Fraction of images for testing (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--copy-images", dest="copy_images", action="store_true",
                        default=True,
                        help="Copy images to processed dir (default: True)")
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false",
                        help="Symlink images instead of copying")
    parser.add_argument("--skip-test-categories", action="store_true",
                        help="Skip creating tests/datasets/ category subdirs")
    parser.add_argument("--force-resplit", action="store_true",
                        help="Ignore existing datasets/splits/*.txt and re-split")
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        logger.error(f"Ratios must sum to 1.0, got {ratios} = {sum(ratios)}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("RICE QUALITY AI — DATASET SPLIT CREATOR")
    logger.info("=" * 60)
    logger.info(f"Ratios: train={ratios[0]}, val={ratios[1]}, test={ratios[2]}")
    logger.info(f"Seed:   {args.seed}")
    logger.info(f"Copy:   {args.copy_images}")

    # 1. Collect image metadata from all sources
    logger.info("\n[1/5] Collecting image metadata...")
    all_metadata = collect_all_metadata()
    if not all_metadata:
        logger.error("No image metadata could be collected. Check paths.")
        sys.exit(1)

    density_counts = Counter(m["density"] for m in all_metadata)
    synth_count = sum(1 for m in all_metadata if m["is_synthetic"])
    logger.info(
        f"  Total: {len(all_metadata)} images  |  "
        f"Density: {dict(density_counts)}  |  "
        f"Synthetic: {synth_count}, Real: {len(all_metadata) - synth_count}"
    )

    # 2. Load existing splits or create new stratified split
    logger.info("\n[2/5] Splitting dataset...")
    if args.force_resplit:
        logger.info("  --force-resplit: ignoring existing split files")
        train_meta, val_meta, test_meta = stratified_split(all_metadata, ratios, args.seed)
    else:
        existing = load_existing_splits(all_metadata)
        if existing is not None:
            train_meta, val_meta, test_meta = existing
        else:
            train_meta, val_meta, test_meta = stratified_split(all_metadata, ratios, args.seed)

    # 3. Compute and write split statistics + .txt files
    logger.info("\n[3/5] Writing split files and statistics...")
    stats = compute_split_stats(train_meta, val_meta, test_meta)
    write_split_files(train_meta, val_meta, test_meta, stats)

    for split_name in ("train", "val", "test"):
        s = stats["per_split"][split_name]
        gc = s["grain_count_stats"]
        db = s["density_buckets"]
        svr = s["synthetic_vs_real"]
        logger.info(
            f"  {split_name:5s}: count={s['count']:<5d}  "
            f"grains[min={gc['min']},max={gc['max']},mean={gc['mean']}]  "
            f"density={db}  "
            f"synth/real={svr['synthetic']}/{svr['real']}"
        )

    # 4. Deploy images and write per-split COCO JSONs
    logger.info("\n[4/5] Deploying images and COCO annotations...")
    for split_name, meta_list in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        deploy_split_images(meta_list, split_name, copy_images=args.copy_images)
    write_split_coco_annotations(train_meta, val_meta, test_meta)

    # 5. Create tests/datasets category subdirectories
    logger.info("\n[5/5] Setting up tests/datasets/ category subdirectories...")
    if not args.skip_test_categories:
        create_test_category_subdirs()
    else:
        logger.info("  --skip-test-categories: skipped")

    logger.info("\n" + "=" * 60)
    logger.info("SPLIT CREATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Split files:      {SPLITS_DIR}/")
    logger.info(f"COCO images:      {COCO_DIR / 'images'}/")
    logger.info(f"COCO annotations: {COCO_DIR / 'annotations'}/")
    logger.info(f"Test categories:  {TEST_DATA_DIR}/")
    logger.info("")


if __name__ == "__main__":
    main()
