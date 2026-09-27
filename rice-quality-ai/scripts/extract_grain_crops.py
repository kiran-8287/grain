"""
scripts/extract_grain_crops.py
==============================
Extract per-grain crops, masks, and visual overlays from segmentation analysis.

Supports input modes:
  1. Single image file + optional pre-computed analysis.json
  2. Results directory (with per-image analysis.json files)
  3. Test images directory (runs inference per image)
  4. Direct inference via ml.inference.analyze_image

Output structure per input image {image_name}:
  results/{image_name}/
      original.png
      overlay.png
      masks/
          grain_0001.png   (full-image binary mask, white=grain on black)
          ...
      crops/
          grain_0001.png   (cropped + masked, RGBA with transparent bg)
          ...
      foreign_matter/
          fm_001.png       (RGB crop of FM bbox + 5px padding)
      analysis.json

Usage:
    python scripts/extract_grain_crops.py --input path/to/image.png
    python scripts/extract_grain_crops.py --input path/to/test_images/ --run-inference
    python scripts/extract_grain_crops.py --input path/to/results_dir/
    python scripts/extract_grain_crops.py --input img.png --analysis-json analysis.json
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from ml.inference import analyze_image


def _polygon_to_mask(polygon_xy: List[List[float]], h: int, w: int) -> Optional[np.ndarray]:
    if not polygon_xy or len(polygon_xy) < 3:
        return None
    pts = np.array(polygon_xy, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask


def _reconstruct_mask(grain_entry: Dict[str, Any], img_h: int, img_w: int) -> np.ndarray:
    mask = grain_entry.get("mask")
    if isinstance(mask, np.ndarray):
        m = mask.astype(np.uint8)
        if m.shape[:2] == (img_h, img_w):
            return (m > 0).astype(np.uint8) * 255
    polygon = grain_entry.get("mask_polygon")
    if polygon:
        m = _polygon_to_mask(polygon, img_h, img_w)
        if m is not None:
            return m
    bbox = grain_entry.get("bbox", [0, 0, 0, 0])
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, w, h = [int(v) for v in bbox]
        m = np.zeros((img_h, img_w), dtype=np.uint8)
        if w > 0 and h > 0:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_w, x + w)
            y2 = min(img_h, y + h)
            m[y1:y2, x1:x2] = 255
        return m
    return np.zeros((img_h, img_w), dtype=np.uint8)


def extract_single_grain_crop(
    image_bgr: np.ndarray,
    mask_binary: np.ndarray,
    bbox_xywh: Tuple[int, int, int, int],
    padding: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a single grain crop with transparency, plus a full-image binary mask.

    Parameters
    ----------
    image_bgr : np.ndarray
        HxWx3 BGR source image.
    mask_binary : np.ndarray
        HxW uint8 binary mask (255=grain, 0=bg), same size as image.
    bbox_xywh : tuple
        (x, y, w, h) grain bounding box in full-image pixel coords.
    padding : int
        Pixels of padding to add around the bbox (clamped to image edges).

    Returns
    -------
    cropped_rgba : np.ndarray
        Cropped region with alpha channel = mask, background transparent.
    full_size_mask : np.ndarray
        Same as input mask_binary (normalised to 0/255 uint8).
    """
    img_h, img_w = image_bgr.shape[:2]
    x, y, w, h = [int(v) for v in bbox_xywh]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    full_size_mask = (mask_binary > 0).astype(np.uint8) * 255

    crop_bgr = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = full_size_mask[y1:y2, x1:x2]

    if crop_bgr.size == 0:
        crop_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        crop_mask = np.zeros((1, 1), dtype=np.uint8)

    rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0:3] = crop_bgr
    rgba[:, :, 3] = crop_mask

    return rgba, full_size_mask


def generate_overlay(
    image_bgr: np.ndarray,
    grains_list: List[Dict[str, Any]],
    fm_list: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Generate a BGR overlay visualisation:
      - Each grain mask is tinted with a distinct hue-cycled colour.
      - Grain ID numbers drawn at each centroid.
      - Foreign-matter boxes drawn in red with confidence labels.

    Parameters
    ----------
    image_bgr : np.ndarray
        HxWx3 BGR source image.
    grains_list : list[dict]
        Per-grain dicts as returned by analyze_image (id, mask_polygon,
        bbox, centroid, confidence, ...).
    fm_list : list[dict]
        Per-FM dicts (id, bbox, confidence, class, ...).

    Returns
    -------
    np.ndarray
        HxWx3 BGR overlay image (uint8).
    """
    img_h, img_w = image_bgr.shape[:2]
    out = image_bgr.copy()
    overlay_layer = out.copy()

    n_grains = len(grains_list)
    for i, grain in enumerate(grains_list):
        hue = int((i / max(n_grains, 1)) * 170)
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
        color_tuple = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        mask = _reconstruct_mask(grain, img_h, img_w)
        mask_bool = mask > 0
        if mask_bool.any():
            for c in range(3):
                overlay_layer[:, :, c] = np.where(
                    mask_bool, color_tuple[c], overlay_layer[:, :, c]
                )

    cv2.addWeighted(overlay_layer, 0.45, out, 0.55, 0, out)

    for i, grain in enumerate(grains_list):
        mask = _reconstruct_mask(grain, img_h, img_w)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

        centroid = grain.get("centroid")
        gid = grain.get("id", i + 1)
        if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
            cx, cy = int(round(float(centroid[0]))), int(round(float(centroid[1])))
        else:
            bbox = grain.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bx, by, bw, bh = [int(v) for v in bbox]
                cx, cy = bx + bw // 2, by + bh // 2
            else:
                cx, cy = 0, 0

        label = str(gid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (cx - tw // 2 - 3, cy - th // 2 - 3),
                      (cx + tw // 2 + 3, cy + th // 2 + 3), (0, 0, 0), -1)
        cv2.putText(out, label, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for fm in fm_list:
        bbox = fm.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            fx, fy, fw, fh = [int(v) for v in bbox]
            conf = float(fm.get("confidence", 0.0))
            cls_name = str(fm.get("class", "FM"))
            cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
            lbl = f"{cls_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (fx, max(0, fy - lh - 6)),
                          (fx + lw + 6, max(0, fy)), (0, 0, 255), -1)
            cv2.putText(out, lbl, (fx + 3, max(0, fy - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def process_image(
    image_path: Path,
    output_root: Path,
    analysis_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a single image: run/load analysis, save crops, masks, overlay.

    Parameters
    ----------
    image_path : Path
        Input image file path.
    output_root : Path
        Top-level results directory; per-image subdir created here.
    analysis_result : dict, optional
        Pre-computed analysis dict from analyze_image. If None and
        analysis JSON next to the image is missing, runs inference.

    Returns
    -------
    dict
        Per-image stats: {image_name, n_grains, n_fm, saved_files: list[str], error: str|None}
    """
    image_path = Path(image_path)
    output_root = Path(output_root)

    stats: Dict[str, Any] = {
        "image_name": image_path.stem,
        "n_grains": 0,
        "n_fm": 0,
        "saved_files": [],
        "error": None,
    }

    try:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            stats["error"] = f"Could not read image: {image_path}"
            return stats
    except Exception as exc:
        stats["error"] = f"Exception reading {image_path}: {exc}"
        return stats

    img_h, img_w = image_bgr.shape[:2]

    result = analysis_result
    if result is None:
        json_candidate = image_path.with_suffix(".json")
        if json_candidate.exists():
            try:
                with open(json_candidate, "r", encoding="utf-8") as f:
                    result = json.load(f)
                logger.info(f"Loaded pre-computed analysis from {json_candidate}")
            except Exception as exc:
                logger.warning(f"Failed to parse {json_candidate}: {exc}. Running inference.")
        if result is None:
            logger.info(f"Running inference on {image_path.name}")
            result = analyze_image(image_bgr)

    grains: List[Dict[str, Any]] = result.get("grains", [])
    foreign_matter: List[Dict[str, Any]] = result.get("foreign_matter", [])

    per_image_dir = output_root / image_path.stem
    masks_dir = per_image_dir / "masks"
    crops_dir = per_image_dir / "crops"
    fm_dir = per_image_dir / "foreign_matter"
    for d in (per_image_dir, masks_dir, crops_dir, fm_dir):
        d.mkdir(parents=True, exist_ok=True)

    original_out = per_image_dir / "original.png"
    if not original_out.exists():
        shutil.copyfile(image_path, original_out)
    else:
        cv2.imwrite(str(original_out), image_bgr)
    stats["saved_files"].append(str(original_out))

    overlay_bgr = generate_overlay(image_bgr, grains, foreign_matter)
    overlay_out = per_image_dir / "overlay.png"
    cv2.imwrite(str(overlay_out), overlay_bgr)
    stats["saved_files"].append(str(overlay_out))

    for idx, grain in enumerate(grains):
        gid = grain.get("id", idx + 1)
        bbox = grain.get("bbox", [0, 0, 0, 0])
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        mask = _reconstruct_mask(grain, img_h, img_w)

        crop_rgba, full_mask = extract_single_grain_crop(
            image_bgr, mask, tuple(bbox), padding=10
        )

        mask_path = masks_dir / f"grain_{int(gid):04d}.png"
        cv2.imwrite(str(mask_path), full_mask)
        stats["saved_files"].append(str(mask_path))

        crop_path = crops_dir / f"grain_{int(gid):04d}.png"
        cv2.imwrite(str(crop_path), crop_rgba)
        stats["saved_files"].append(str(crop_path))

        stats["n_grains"] += 1

    for idx, fm in enumerate(foreign_matter):
        fid = fm.get("id", idx + 1)
        bbox = fm.get("bbox", [0, 0, 0, 0])
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        fx, fy, fw, fh = [int(v) for v in bbox]
        pad = 5
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(img_w, fx + fw + pad)
        y2 = min(img_h, fy + fh + pad)
        fm_crop = image_bgr[y1:y2, x1:x2].copy()
        if fm_crop.size == 0:
            fm_crop = np.zeros((1, 1, 3), dtype=np.uint8)
        fm_out = fm_dir / f"fm_{int(fid):03d}.png"
        cv2.imwrite(str(fm_out), fm_crop)
        stats["saved_files"].append(str(fm_out))
        stats["n_fm"] += 1

    analysis_out = per_image_dir / "analysis.json"
    serialisable = {}
    for k, v in result.items():
        try:
            json.dumps({k: v}, default=float)
            serialisable[k] = v
        except (TypeError, ValueError):
            if isinstance(v, np.ndarray):
                serialisable[k] = v.shape
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.ndarray):
                serialisable[k] = len(v)
            else:
                serialisable[k] = str(type(v).__name__)
    with open(analysis_out, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=float)
    stats["saved_files"].append(str(analysis_out))

    logger.info(
        f"{image_path.name}: grains={stats['n_grains']}, fm={stats['n_fm']}, "
        f"files={len(stats['saved_files'])}"
    )
    return stats


def process_directory(
    input_dir: Path,
    output_root: Path,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> Dict[str, Any]:
    """
    Batch-process a directory of images.

    If input_dir looks like a previous results root (contains subdirs with
    analysis.json files), those are reused instead of running inference.

    Parameters
    ----------
    input_dir : Path
        Directory with images *or* a results root directory.
    output_root : Path
        Where new results are written.
    extensions : tuple
        Accepted image file suffixes.

    Returns
    -------
    dict
        Aggregate: {total_images, total_grains, total_fm, per_image: list[dict]}
    """
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    agg: Dict[str, Any] = {
        "total_images": 0,
        "total_grains": 0,
        "total_fm": 0,
        "per_image": [],
    }

    possible_results_subdirs = [
        p for p in input_dir.iterdir()
        if p.is_dir() and (p / "analysis.json").exists()
    ]
    results_mode = len(possible_results_subdirs) > 0

    if results_mode:
        logger.info(f"Detected results directory mode: {len(possible_results_subdirs)} subdirs")
        for sub in sorted(possible_results_subdirs):
            analysis_json = sub / "analysis.json"
            try:
                with open(analysis_json, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except Exception as exc:
                logger.warning(f"Skipping {sub}: failed to load analysis.json: {exc}")
                continue

            original_p = sub / "original.png"
            original_candidates = [original_p]
            for ext in extensions:
                original_candidates.append(input_dir / f"{sub.name}{ext}")
            image_path = None
            for cand in original_candidates:
                if cand.exists():
                    image_path = cand
                    break
            if image_path is None:
                logger.warning(f"No original image found for {sub.name}; skipping")
                continue

            stats = process_image(image_path, output_root, analysis_result=result)
            agg["total_images"] += 1
            agg["total_grains"] += stats["n_grains"]
            agg["total_fm"] += stats["n_fm"]
            agg["per_image"].append(stats)
    else:
        image_files = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        for img_p in image_files:
            stats = process_image(img_p, output_root)
            agg["total_images"] += 1
            agg["total_grains"] += stats["n_grains"]
            agg["total_fm"] += stats["n_fm"]
            agg["per_image"].append(stats)

    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-grain crops, masks, and visualisations."
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input: image file, directory of images, or results directory.")
    parser.add_argument("--output", "-o", type=str, default=str(PROJECT_ROOT / "results"),
                        help=f"Output root directory. Default: {PROJECT_ROOT / 'results'}")
    parser.add_argument("--run-inference", action="store_true",
                        help="Force running ml.inference.analyze_image even if cached JSON exists.")
    parser.add_argument("--analysis-json", type=str, default=None,
                        help="Path to a pre-computed analysis.json (only valid for single-image --input).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    preloaded: Optional[Dict[str, Any]] = None
    if args.analysis_json:
        if not Path(args.analysis_json).exists():
            logger.error(f"--analysis-json file not found: {args.analysis_json}")
            sys.exit(1)
        try:
            with open(args.analysis_json, "r", encoding="utf-8") as f:
                preloaded = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to parse analysis JSON: {exc}")
            sys.exit(1)

    if input_path.is_file():
        if preloaded is not None:
            logger.info(f"Using pre-computed analysis from {args.analysis_json}")
            stats = process_image(input_path, output_root, analysis_result=preloaded)
        elif args.run_inference:
            img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"Cannot read image: {input_path}")
                sys.exit(1)
            result = analyze_image(img)
            stats = process_image(input_path, output_root, analysis_result=result)
        else:
            stats = process_image(input_path, output_root)
        print(json.dumps(stats, indent=2, default=str))
    elif input_path.is_dir():
        if args.analysis_json:
            logger.error("--analysis-json is only valid for single-image input.")
            sys.exit(1)
        agg = process_directory(input_path, output_root)
        summary = {
            "total_images": agg["total_images"],
            "total_grains": agg["total_grains"],
            "total_fm": agg["total_fm"],
            "output_root": str(output_root.resolve()),
        }
        print(json.dumps(summary, indent=2))
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
