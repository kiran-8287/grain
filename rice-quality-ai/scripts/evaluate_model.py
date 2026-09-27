"""
scripts/evaluate_model.py
==========================
Evaluate trained instance segmentation models for rice quality analysis.

Supports:
- YOLOv8: via ultralytics YOLO class (loads .pt from models/yolo_seg/*/weights/best.pt)
- Mask R-CNN: via torchvision maskrcnn_resnet50_fpn_v2 loaded from models/maskrcnn/best/model.pth
- Classical CV: ml/segmentation.segment_grains fallback as baseline comparison

Outputs:
- results/evaluation/eval_report.json      (overall + per-category breakdown)
- results/evaluation/per_image_results.json
- results/evaluation/summary.txt           (human readable table)
- results/evaluation/failures/{category}/{image_name}.png  (4-panel failure images)

Usage:
    python scripts/evaluate_model.py --model-type auto
    python scripts/evaluate_model.py --model-type yolo --weights models/yolo_seg/run1_baseline/weights/best.pt
    python scripts/evaluate_model.py --model-type maskrcnn --weights models/maskrcnn/best/model.pth
    python scripts/evaluate_model.py --model-type classical --generate-failure-images
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Optional Imports ──────────────────────────────────────────────────────────
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logger.warning("pycocotools not available; using custom metric approximation.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    MASKRCNN_AVAILABLE = True
except ImportError:
    MASKRCNN_AVAILABLE = False

from ml.segmentation import segment_grains

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES_9 = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
CATEGORY_NAMES = {
    "A": "single_isolated",
    "B": "few_separated",
    "C": "touching",
    "D": "overlapping",
    "E": "dense",
    "F": "rice_plus_foreign",
    "G": "no_rice",
    "H": "foreign_only",
    "I": "mixed_difficult",
}

FAILURE_TYPES = [
    "MISSED_GRAIN",
    "MERGED_GRAINS",
    "SPLIT_GRAIN",
    "FALSE_RICE",
    "FALSE_FOREIGN_MATTER",
    "DUPLICATE",
    "BAD_MASK",
    "LOW_CONFIDENCE",
]

DEFAULT_COCO_ANN = PROJECT_ROOT / "datasets" / "processed" / "coco" / "annotations" / "instances_test.json"
DEFAULT_COCO_IMG_DIR = PROJECT_ROOT / "datasets" / "processed" / "coco" / "images" / "test"
DEFAULT_TEST_DATASETS_DIR = PROJECT_ROOT / "tests" / "datasets"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo_seg" / "run1_baseline" / "weights" / "best.pt"
DEFAULT_MASKRCNN_WEIGHTS = PROJECT_ROOT / "models" / "maskrcnn" / "best" / "model.pth"

IOU_THRESHOLDS_AP50_95 = [round(0.5 + i * 0.05, 2) for i in range(10)]


# ── Category Assignment ───────────────────────────────────────────────────────
def categorize_image(gt_count: int, has_touching: bool, has_overlap: bool,
                     has_foreign: bool, is_no_rice: bool) -> str:
    """
    Categorize a test image into one of 9 categories (A-I) based on its
    ground-truth properties.

    Parameters
    ----------
    gt_count : int
        Number of rice grain ground-truth instances.
    has_touching : bool
        Whether at least one pair of GT grains is touching.
    has_overlap : bool
        Whether at least one pair of GT grains overlaps.
    has_foreign : bool
        Whether at least one foreign-matter GT instance exists.
    is_no_rice : bool
        Whether the image contains zero rice grains.

    Returns
    -------
    str
        Single-letter category code A–I.
    """
    if is_no_rice or gt_count == 0:
        return "H" if has_foreign else "G"

    if has_foreign and gt_count > 0:
        return "F"

    if has_overlap:
        return "D"

    if has_touching:
        return "C"

    if gt_count >= 50:
        return "E"

    if gt_count == 1:
        return "A"

    if 2 <= gt_count <= 10:
        return "B"

    return "I"


# ── COCO Helpers ──────────────────────────────────────────────────────────────
def _polygon_to_mask(polygon_xy: List[float], h: int, w: int) -> Optional[np.ndarray]:
    """Convert a flat COCO polygon [x1,y1,...] to a uint8 binary mask."""
    if len(polygon_xy) < 6:
        return None
    pts = np.array(polygon_xy, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask


def _bbox_xywh_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def _box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU of two boxes in xyxy format."""
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    ix1 = max(x1a, x1b)
    iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b)
    iy2 = min(y2a, y2b)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area1 = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
    area2 = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two uint8 binary masks."""
    a = mask_a > 0
    b = mask_b > 0
    inter = float(np.sum(np.logical_and(a, b)))
    union = float(np.sum(np.logical_or(a, b)))
    if union <= 0:
        return 0.0
    return inter / union


def _annotations_to_dicts(coco_anns: List[Dict], img_h: int, img_w: int) -> List[Dict]:
    """Convert raw COCO annotation records to normalised dicts with boxes and masks."""
    result = []
    for a in coco_anns:
        cat_id = int(a.get("category_id", 1))
        bbox_xywh = a.get("bbox", [0, 0, 0, 0])
        box_xyxy = _bbox_xywh_to_xyxy(bbox_xywh)

        mask = None
        seg = a.get("segmentation")
        if isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list):
                for poly in seg:
                    m = _polygon_to_mask(poly, img_h, img_w)
                    if m is not None:
                        mask = m if mask is None else cv2.bitwise_or(mask, m)
            else:
                mask = _polygon_to_mask(seg, img_h, img_w)

        result.append({
            "id": a.get("id"),
            "category_id": cat_id,
            "is_rice": cat_id == 1,
            "bbox_xyxy": box_xyxy,
            "bbox_xywh": list(bbox_xywh),
            "area": float(a.get("area", 0)),
            "mask": mask,
        })
    return result


def _detect_touching_overlap(gt_anns: List[Dict]) -> Tuple[bool, bool]:
    """Scan GT pairs and detect whether any are touching or overlapping (mask-level)."""
    has_touching = False
    has_overlap = False

    rice_masks = [a["mask"] for a in gt_anns if a["is_rice"] and a["mask"] is not None]
    n = len(rice_masks)
    for i in range(n):
        for j in range(i + 1, n):
            iou = _mask_iou(rice_masks[i], rice_masks[j])
            if iou > 0.0:
                if iou >= 0.15:
                    has_overlap = True
                else:
                    has_touching = True
            else:
                a_i = rice_masks[i] > 0
                a_j = rice_masks[j] > 0
                dil_i = cv2.dilate(a_i.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                if float(np.sum(np.logical_and(dil_i > 0, a_j))) > 0:
                    has_touching = True

    return has_touching, has_overlap


# ── Prediction Matching ───────────────────────────────────────────────────────
def match_predictions_to_gt(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    pred_masks: Optional[List[np.ndarray]] = None,
    gt_masks: Optional[List[np.ndarray]] = None,
    iou_threshold: float = 0.5,
    use_mask: bool = False,
) -> Tuple[List[Optional[int]], Dict[str, int]]:
    """
    Greedily match prediction boxes/masks to GT, highest IoU first.

    Parameters
    ----------
    pred_boxes : list of xyxy
    gt_boxes : list of xyxy
    pred_masks, gt_masks : optional list of uint8 masks
    iou_threshold : float
    use_mask : bool
        If True, use mask IoU for matching (where masks available).

    Returns
    -------
    matches : list of length len(pred_boxes)
        Element i = matched GT index (int) or None for FP.
    tp_fp_fn_dict : dict
        TP / FP / FN counts.
    """
    n_p = len(pred_boxes)
    n_g = len(gt_boxes)

    iou_matrix = np.zeros((n_p, n_g), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_g):
            if use_mask and pred_masks is not None and gt_masks is not None \
                    and pred_masks[i] is not None and gt_masks[j] is not None:
                iou_matrix[i, j] = _mask_iou(pred_masks[i], gt_masks[j])
            else:
                iou_matrix[i, j] = _box_iou(pred_boxes[i], gt_boxes[j])

    matches: List[Optional[int]] = [None] * n_p
    matched_gt: set = set()

    order = sorted(range(n_p), key=lambda i: (iou_matrix[i].max() if n_g else 0), reverse=True)
    for i in order:
        best_j = -1
        best_iou = 0.0
        for j in range(n_g):
            if j in matched_gt:
                continue
            if iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= iou_threshold:
                best_iou = iou_matrix[i, j]
                best_j = j
        if best_j >= 0:
            matches[i] = best_j
            matched_gt.add(best_j)

    tp = len(matched_gt)
    fp = n_p - tp
    fn = n_g - tp

    return matches, {"TP": tp, "FP": fp, "FN": fn}


# ── Merged / Split detection ──────────────────────────────────────────────────
def count_merged_split(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    matches: List[Optional[int]],
    iou_threshold: float = 0.3,
) -> Dict[str, int]:
    """
    Count merged and split detections from mask overlap analysis.

    - **MERGED**: 1 pred mask overlaps >=2 GT masks significantly (IoU >= threshold).
    - **SPLIT**:  1 GT mask overlaps >=2 pred masks significantly.
    """
    n_p = len(pred_masks)
    n_g = len(gt_masks)

    p_to_g_overlaps: List[List[int]] = [[] for _ in range(n_p)]
    g_to_p_overlaps: List[List[int]] = [[] for _ in range(n_g)]

    for i in range(n_p):
        for j in range(n_g):
            if pred_masks[i] is not None and gt_masks[j] is not None:
                iou = _mask_iou(pred_masks[i], gt_masks[j])
                if iou >= iou_threshold:
                    p_to_g_overlaps[i].append(j)
                    g_to_p_overlaps[j].append(i)

    merged_count = sum(1 for row in p_to_g_overlaps if len(row) >= 2)
    split_count = sum(1 for row in g_to_p_overlaps if len(row) >= 2)

    duplicate_count = 0
    matched_preds = [i for i, m in enumerate(matches) if m is not None]
    gt_to_preds: Dict[int, List[int]] = defaultdict(list)
    for i in matched_preds:
        gt_to_preds[matches[i]].append(i)
    for _, preds in gt_to_preds.items():
        if len(preds) >= 2:
            duplicate_count += len(preds) - 1

    return {
        "merged_count": merged_count,
        "split_count": split_count,
        "duplicate_count": duplicate_count,
    }


# ── Per-Image Metrics ─────────────────────────────────────────────────────────
def compute_metrics_for_image(
    gt_anns: List[Dict],
    pred_anns: List[Dict],
    img_h: int,
    img_w: int,
) -> Dict[str, Any]:
    """
    Compute all per-image metrics (AP, count error, rates, failures).

    Parameters
    ----------
    gt_anns : list of normalised GT dicts (from _annotations_to_dicts)
    pred_anns : list of prediction dicts with keys:
        bbox_xyxy, mask, confidence, is_rice
    img_h, img_w : image dimensions

    Returns
    -------
    dict with all metric fields.
    """
    gt_rice = [a for a in gt_anns if a["is_rice"]]
    gt_foreign = [a for a in gt_anns if not a["is_rice"]]
    pred_rice = [a for a in pred_anns if a.get("is_rice", True)]
    pred_foreign = [a for a in pred_anns if not a.get("is_rice", True)]

    gt_boxes_r = [a["bbox_xyxy"] for a in gt_rice]
    pred_boxes_r = [a["bbox_xyxy"] for a in pred_rice]
    gt_masks_r = [a["mask"] for a in gt_rice]
    pred_masks_r = [a.get("mask") for a in pred_rice]

    gt_count = len(gt_rice)
    pred_count = len(pred_rice)
    count_error_pct = 100.0 * abs(pred_count - gt_count) / max(gt_count, 1)

    matches_50, tp_fp_fn_50 = match_predictions_to_gt(
        pred_boxes_r, gt_boxes_r, pred_masks_r, gt_masks_r, iou_threshold=0.5, use_mask=False
    )
    matches_mask_50, mask_tp_fp_fn_50 = match_predictions_to_gt(
        pred_boxes_r, gt_boxes_r, pred_masks_r, gt_masks_r, iou_threshold=0.5, use_mask=True
    )

    ap_ious = IOU_THRESHOLDS_AP50_95
    recalls = []
    precisions = []
    mask_recalls = []
    for t in ap_ious:
        _, m = match_predictions_to_gt(pred_boxes_r, gt_boxes_r, pred_masks_r, gt_masks_r, t, use_mask=False)
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls.append(rec)
        precisions.append(prec)

        _, mm = match_predictions_to_gt(pred_boxes_r, gt_boxes_r, pred_masks_r, gt_masks_r, t, use_mask=True)
        mtp, mfn = mm["TP"], mm["FN"]
        mask_recalls.append(mtp / (mtp + mfn) if (mtp + mfn) > 0 else 0.0)

    ap50_box = recalls[0]
    ap50_95_box = float(np.mean(recalls))
    ap50_mask = mask_recalls[0]
    ap50_95_mask = float(np.mean(mask_recalls))

    missed_rate = tp_fp_fn_50["FN"] / max(gt_count, 1)

    ms = count_merged_split(pred_masks_r, gt_masks_r, matches_50)
    merged_rate = ms["merged_count"] / max(gt_count, 1)
    split_rate = ms["split_count"] / max(gt_count, 1)
    duplicate_rate = ms["duplicate_count"] / max(gt_count, 1)

    false_foreign_rate = 0.0
    if len(gt_rice) > 0 and len(pred_foreign) > 0:
        f_boxes = [a["bbox_xyxy"] for a in pred_foreign]
        f_masks = [a.get("mask") for a in pred_foreign]
        false_foreign_hits = 0
        for i, fb in enumerate(f_boxes):
            for j, gb in enumerate(gt_boxes_r):
                fm = f_masks[i] if i < len(f_masks) else None
                gm = gt_masks_r[j]
                iou_val = _mask_iou(fm, gm) if (fm is not None and gm is not None) else _box_iou(fb, gb)
                if iou_val >= 0.5:
                    false_foreign_hits += 1
                    break
        false_foreign_rate = false_foreign_hits / max(gt_count, 1)

    failures: Dict[str, int] = {k: 0 for k in FAILURE_TYPES}
    failures["MISSED_GRAIN"] = tp_fp_fn_50["FN"]
    failures["MERGED_GRAINS"] = ms["merged_count"]
    failures["SPLIT_GRAIN"] = ms["split_count"]
    failures["DUPLICATE"] = ms["duplicate_count"]

    failures["FALSE_FOREIGN_MATTER"] = sum(1 for i, m in enumerate(matches_50) if m is None
                                           and not pred_anns[i].get("is_rice", True))
    failures["FALSE_RICE"] = tp_fp_fn_50["FP"] - failures["FALSE_FOREIGN_MATTER"]

    bad_mask_count = 0
    low_conf_count = 0
    for i, gi in enumerate(matches_50):
        if gi is None:
            continue
        pm = pred_masks_r[i]
        gm = gt_masks_r[gi]
        if pm is not None and gm is not None:
            if _mask_iou(pm, gm) < 0.5:
                bad_mask_count += 1
        if pred_anns[i].get("confidence", 1.0) < 0.5:
            low_conf_count += 1
    failures["BAD_MASK"] = bad_mask_count
    failures["LOW_CONFIDENCE"] = low_conf_count

    return {
        "gt_count": gt_count,
        "pred_count": pred_count,
        "count_error_pct": count_error_pct,
        "ap50_box": ap50_box,
        "ap50_95_box": ap50_95_box,
        "ap50_mask": ap50_mask,
        "ap50_95_mask": ap50_95_mask,
        "missed_rate": missed_rate,
        "merged_rate": merged_rate,
        "split_rate": split_rate,
        "duplicate_rate": duplicate_rate,
        "false_foreign_rate": false_foreign_rate,
        "gt_foreign_count": len(gt_foreign),
        "pred_foreign_count": len(pred_foreign),
        "tp_fp_fn": tp_fp_fn_50,
        "mask_tp_fp_fn": mask_tp_fp_fn_50,
        "failures": failures,
    }


# ── Model Predictors ──────────────────────────────────────────────────────────
class YOLOPredictor:
    def __init__(self, weights_path: Path, device: str = "", conf: float = 0.25):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed; cannot use YOLOPredictor.")
        self.weights_path = weights_path
        self.model = YOLO(str(weights_path))
        self.device = device
        self.conf = conf

    def predict(self, img_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        t0 = time.perf_counter()
        results = self.model.predict(img_bgr, conf=self.conf, verbose=False, device=self.device or None)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pred_anns: List[Dict] = []
        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            masks = r.masks

            h, w = img_bgr.shape[:2]

            if boxes is not None:
                for idx in range(len(boxes)):
                    xyxy = boxes.xyxy[idx].cpu().numpy().tolist()
                    conf = float(boxes.conf[idx].cpu().numpy()) if boxes.conf is not None else 0.5
                    cls_id = int(boxes.cls[idx].cpu().numpy()) if boxes.cls is not None else 0

                    mask = None
                    if masks is not None and idx < len(masks.data):
                        m_data = masks.data[idx].cpu().numpy().astype(np.uint8) * 255
                        if m_data.ndim == 2 and (m_data.shape[0] != h or m_data.shape[1] != w):
                            mask = cv2.resize(m_data, (w, h), interpolation=cv2.INTER_NEAREST)
                        elif m_data.ndim == 2:
                            mask = m_data

                    is_rice = cls_id in (0, 1)
                    pred_anns.append({
                        "bbox_xyxy": xyxy,
                        "confidence": conf,
                        "mask": mask,
                        "is_rice": is_rice,
                        "category_id": 1 if is_rice else 2,
                    })

        return pred_anns, elapsed_ms


class MaskRCNNPredictor:
    def __init__(self, weights_path: Path, device: str = "", conf: float = 0.5):
        if not MASKRCNN_AVAILABLE:
            raise RuntimeError("torch/torchvision not installed; cannot use MaskRCNNPredictor.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights=None, num_classes=3)
        state = torch.load(str(weights_path), map_location=self.device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        t0 = time.perf_counter()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        out = outputs[0]
        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        masks = out["masks"].cpu().numpy()

        h, w = img_bgr.shape[:2]
        pred_anns: List[Dict] = []

        for i in range(len(boxes)):
            if float(scores[i]) < self.conf:
                continue
            xyxy = boxes[i].tolist()
            cat_id = int(labels[i])
            conf = float(scores[i])

            mask = None
            if i < len(masks):
                m_thresh = (masks[i, 0] >= 0.5).astype(np.uint8) * 255
                if m_thresh.shape[0] != h or m_thresh.shape[1] != w:
                    mask = cv2.resize(m_thresh, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = m_thresh

            is_rice = cat_id == 1
            pred_anns.append({
                "bbox_xyxy": xyxy,
                "confidence": conf,
                "mask": mask,
                "is_rice": is_rice,
                "category_id": cat_id,
            })

        return pred_anns, elapsed_ms


class ClassicalCVPredictor:
    def __init__(self):
        pass

    def predict(self, img_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        result = segment_grains(img_rgb, method="classical_cv")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pred_anns: List[Dict] = []
        for g in result.grains:
            x, y, bw, bh = g.bbox
            pred_anns.append({
                "bbox_xyxy": [x, y, x + bw, y + bh],
                "confidence": float(g.confidence),
                "mask": g.mask,
                "is_rice": True,
                "category_id": 1,
            })

        return pred_anns, elapsed_ms


def build_predictor(model_type: str, weights_path: Optional[Path],
                    device: str = "", conf: float = 0.25):
    if model_type == "yolo":
        if weights_path is None or not weights_path.exists():
            weights_path = DEFAULT_YOLO_WEIGHTS
        return YOLOPredictor(weights_path, device=device, conf=conf)
    elif model_type == "maskrcnn":
        if weights_path is None or not weights_path.exists():
            weights_path = DEFAULT_MASKRCNN_WEIGHTS
        return MaskRCNNPredictor(weights_path, device=device, conf=conf)
    elif model_type == "classical":
        return ClassicalCVPredictor()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def auto_detect_model_type(weights_path: Optional[Path]) -> str:
    if weights_path is not None:
        s = str(weights_path).lower()
        if "yolo" in s or s.endswith(".pt"):
            return "yolo"
        if "maskrcnn" in s or "mask_rcnn" in s:
            return "maskrcnn"
    if DEFAULT_YOLO_WEIGHTS.exists():
        return "yolo"
    if DEFAULT_MASKRCNN_WEIGHTS.exists():
        return "maskrcnn"
    return "classical"


# ── Failure Visualisation ─────────────────────────────────────────────────────
def _overlay_masks(img_bgr: np.ndarray, masks: List[np.ndarray],
                   color: Tuple[int, int, int], alpha: float = 0.4) -> np.ndarray:
    out = img_bgr.copy()
    overlay = out.copy()
    for m in masks:
        if m is None:
            continue
        m3 = (m > 0).astype(np.uint8)[:, :, None]
        color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
        colored = np.full_like(overlay, color_arr)
        overlay = np.where(m3, colored, overlay)
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
    for m in masks:
        if m is None:
            continue
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)
    return out


def generate_4_panel_failure(img_bgr: np.ndarray, gt_masks: List[np.ndarray],
                             pred_masks: List[np.ndarray], out_path: Path) -> bool:
    """
    Generate a 4-panel diagnostic image:
        [input | GT overlay | prediction overlay | diff mask]
    """
    try:
        h, w = img_bgr.shape[:2]

        input_panel = img_bgr.copy()
        gt_panel = _overlay_masks(img_bgr, gt_masks, (0, 255, 0), alpha=0.35)
        pred_panel = _overlay_masks(img_bgr, pred_masks, (0, 165, 255), alpha=0.35)

        gt_union = np.zeros((h, w), dtype=np.uint8)
        pred_union = np.zeros((h, w), dtype=np.uint8)
        for m in gt_masks:
            if m is not None:
                gt_union = cv2.bitwise_or(gt_union, m.astype(np.uint8))
        for m in pred_masks:
            if m is not None:
                pred_union = cv2.bitwise_or(pred_union, m.astype(np.uint8))

        diff_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        only_gt = np.logical_and(gt_union > 0, pred_union == 0)
        only_pred = np.logical_and(pred_union > 0, gt_union == 0)
        both = np.logical_and(gt_union > 0, pred_union > 0)
        diff_bgr[only_gt] = (0, 255, 0)
        diff_bgr[only_pred] = (0, 0, 255)
        diff_bgr[both] = (255, 255, 255)

        def _label(img, title):
            cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            return img

        input_panel = _label(input_panel, "1. Input")
        gt_panel = _label(gt_panel, "2. GT (green)")
        pred_panel = _label(pred_panel, "3. Prediction (orange)")
        diff_bgr = _label(diff_bgr, "4. Diff: GT(grn)=miss, Pred(red)=FP")

        top = np.hstack([input_panel, gt_panel])
        bot = np.hstack([pred_panel, diff_bgr])
        composite = np.vstack([top, bot])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(out_path), composite))
    except Exception as e:
        logger.warning(f"Failed to generate 4-panel failure image {out_path}: {e}")
        return False


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_coco_split(ann_path: Path, img_dir: Path, split: str = "test",
                    max_images: Optional[int] = None) -> List[Dict]:
    """
    Load image records + GT annotations from a COCO JSON.

    Returns list of dicts: {image_id, file_name, img_path, h, w, gt_anns, category}
    """
    if not ann_path.exists():
        logger.warning(f"COCO annotations not found at {ann_path}; skipping COCO split.")
        return []

    with open(ann_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    records = []
    img_id_to_anns: Dict[int, List[Dict]] = defaultdict(list)
    for a in coco_data.get("annotations", []):
        img_id_to_anns[a["image_id"]].append(a)

    cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    _ = cat_map

    for i, img_entry in enumerate(coco_data.get("images", [])):
        if max_images is not None and i >= max_images:
            break
        img_id = img_entry["id"]
        fname = img_entry["file_name"]
        h = int(img_entry.get("height", 0))
        w = int(img_entry.get("width", 0))

        candidates = [
            img_dir / fname,
            img_dir / split / fname,
            DEFAULT_COCO_IMG_DIR / fname,
            PROJECT_ROOT / "datasets" / "processed" / "coco" / "images" / split / fname,
        ]
        img_path = None
        for cand in candidates:
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            logger.warning(f"Image not found for {fname}; skipping.")
            continue

        raw_anns = img_id_to_anns.get(img_id, [])
        if h == 0 or w == 0:
            tmp = cv2.imread(str(img_path))
            if tmp is not None:
                h, w = tmp.shape[:2]
        gt_anns = _annotations_to_dicts(raw_anns, h, w)

        rice_count = sum(1 for a in gt_anns if a["is_rice"])
        has_fm = any(not a["is_rice"] for a in gt_anns)
        no_rice = rice_count == 0
        touching, overlap = _detect_touching_overlap(gt_anns)
        cat = categorize_image(rice_count, touching, overlap, has_fm, no_rice)

        records.append({
            "image_id": img_id,
            "file_name": fname,
            "img_path": img_path,
            "h": h,
            "w": w,
            "gt_anns": gt_anns,
            "category": cat,
            "source": "coco",
        })

    logger.info(f"Loaded {len(records)} images from COCO split {ann_path}.")
    return records


def load_tests_datasets(tests_datasets_dir: Path,
                        max_images_per_category: Optional[int] = None) -> List[Dict]:
    """
    Load image records from tests/datasets/{category_code_A..I}/ subdirectories.

    Each subfolder name maps directly to a category code (A-I) or to a category
    name (e.g. "single_isolated" -> "A").  Both are accepted.
    """
    if not tests_datasets_dir.exists():
        logger.info(f"tests/datasets/ directory not present at {tests_datasets_dir}; skipping.")
        return []

    name_to_code = {v: k for k, v in CATEGORY_NAMES.items()}
    records: List[Dict] = []

    for sub in sorted(tests_datasets_dir.iterdir()):
        if not sub.is_dir():
            continue
        key = sub.name.strip()
        cat = None
        if key in CATEGORIES_9:
            cat = key
        elif key.lower() in name_to_code:
            cat = name_to_code[key.lower()]
        elif key.lower() in {n.lower() for n in CATEGORY_NAMES.values()}:
            for nc, name in CATEGORY_NAMES.items():
                if name.lower() == key.lower():
                    cat = nc
                    break
        if cat is None:
            logger.info(f"Skipping tests/datasets/{sub.name} — no known category mapping.")
            continue

        img_files = sorted([p for p in sub.iterdir()
                            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])
        loaded = 0
        for ip in img_files:
            if max_images_per_category is not None and loaded >= max_images_per_category:
                break
            img = cv2.imread(str(ip))
            if img is None:
                continue
            h, w = img.shape[:2]

            gt_anns: List[Dict] = []
            ann_json = ip.with_suffix(".json")
            if ann_json.exists():
                try:
                    with open(ann_json, "r", encoding="utf-8") as f:
                        per_img_anns = json.load(f)
                    raw_list = per_img_anns if isinstance(per_img_anns, list) else per_img_anns.get("annotations", [])
                    gt_anns = _annotations_to_dicts(raw_list, h, w)
                except Exception as e:
                    logger.warning(f"Failed to parse GT JSON {ann_json}: {e}")

            rice_count = sum(1 for a in gt_anns if a["is_rice"])
            has_fm = any(not a["is_rice"] for a in gt_anns)
            no_rice = rice_count == 0
            touching, overlap = _detect_touching_overlap(gt_anns)
            if len(gt_anns) == 0:
                inferred = cat
            else:
                inferred = categorize_image(rice_count, touching, overlap, has_fm, no_rice)
            final_cat = cat if cat else inferred

            records.append({
                "image_id": f"tds_{sub.name}_{loaded}",
                "file_name": ip.name,
                "img_path": ip,
                "h": h,
                "w": w,
                "gt_anns": gt_anns,
                "category": final_cat,
                "source": f"tests/datasets/{sub.name}",
            })
            loaded += 1

    logger.info(f"Loaded {len(records)} images from tests/datasets/ structured folder.")
    return records


# ── Aggregation ───────────────────────────────────────────────────────────────
def _aggregate_category_metrics(per_image: List[Dict]) -> Dict[str, Any]:
    if not per_image:
        empty_rates = {k: 0.0 for k in FAILURE_TYPES}
        return {
            "num_images": 0,
            "total_gt_instances": 0,
            "total_pred_instances": 0,
            "mean_ap50_box": 0.0,
            "mean_ap50_95_box": 0.0,
            "mean_ap50_mask": 0.0,
            "mean_ap50_95_mask": 0.0,
            "mean_count_error_pct": 0.0,
            "mean_missed_rate": 0.0,
            "mean_merged_rate": 0.0,
            "mean_split_rate": 0.0,
            "mean_duplicate_rate": 0.0,
            "mean_false_foreign_rate": 0.0,
            "mean_inference_ms": 0.0,
            "failure_counts": {k: 0 for k in FAILURE_TYPES},
            "failure_rates_total_gt": empty_rates,
        }

    def _mean(key):
        vals = [r[key] for r in per_image if key in r]
        return float(np.mean(vals)) if vals else 0.0

    total_gt = sum(r["gt_count"] + r["gt_foreign_count"] for r in per_image)
    total_gt_rice = sum(r["gt_count"] for r in per_image)

    failures_total: Dict[str, int] = {k: 0 for k in FAILURE_TYPES}
    for r in per_image:
        f = r.get("failures", {})
        for k in FAILURE_TYPES:
            failures_total[k] += int(f.get(k, 0))

    failure_rates = {k: failures_total[k] / max(total_gt_rice, 1) for k in FAILURE_TYPES}

    return {
        "num_images": len(per_image),
        "total_gt_instances": total_gt,
        "total_gt_rice": total_gt_rice,
        "total_pred_instances": sum(r["pred_count"] + r["pred_foreign_count"] for r in per_image),
        "mean_ap50_box": _mean("ap50_box"),
        "mean_ap50_95_box": _mean("ap50_95_box"),
        "mean_ap50_mask": _mean("ap50_mask"),
        "mean_ap50_95_mask": _mean("ap50_95_mask"),
        "mean_count_error_pct": _mean("count_error_pct"),
        "mean_missed_rate": _mean("missed_rate"),
        "mean_merged_rate": _mean("merged_rate"),
        "mean_split_rate": _mean("split_rate"),
        "mean_duplicate_rate": _mean("duplicate_rate"),
        "mean_false_foreign_rate": _mean("false_foreign_rate"),
        "mean_inference_ms": _mean("inference_ms"),
        "failure_counts": failures_total,
        "failure_rates_total_gt": failure_rates,
    }


def format_summary_txt(overall: Dict, per_cat: Dict[str, Dict],
                       model_type: str, weights_path: Optional[Path]) -> str:
    lines = []
    lines.append("=" * 98)
    lines.append("RICE QUALITY AI — SEGMENTATION MODEL EVALUATION REPORT")
    lines.append("=" * 98)
    lines.append(f"Model Type     : {model_type}")
    lines.append(f"Weights Path   : {weights_path if weights_path else 'N/A (classical CV)'}")
    lines.append("")

    def _fmt_pct(x): return f"{x*100:6.2f}%" if x is not None else "  N/A  "
    def _fmt_val(x): return f"{x:7.4f}" if x is not None else "  N/A  "

    header = (
        f"{'Cat':<3s} {'Name':<18s} {'N':>4s} "
        f"{'AP50_b':>7s} {'mAP_b':>7s} {'AP50_m':>7s} {'mAP_m':>7s} "
        f"{'CntE%':>6s} {'Miss%':>6s} {'Mrg%':>6s} {'Spl%':>6s} {'Dup%':>5s} {'FF%':>5s} "
        f"{'ms':>6s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for cat in CATEGORIES_9:
        m = per_cat.get(cat)
        if m is None or m["num_images"] == 0:
            continue
        n = m["num_images"]
        name = CATEGORY_NAMES.get(cat, cat)
        lines.append(
            f"{cat:<3s} {name:<18s} {n:>4d} "
            f"{_fmt_val(m['mean_ap50_box'])} {_fmt_val(m['mean_ap50_95_box'])} "
            f"{_fmt_val(m['mean_ap50_mask'])} {_fmt_val(m['mean_ap50_95_mask'])} "
            f"{m['mean_count_error_pct']:6.2f} {m['mean_missed_rate']*100:6.2f} "
            f"{m['mean_merged_rate']*100:6.2f} {m['mean_split_rate']*100:6.2f} "
            f"{m['mean_duplicate_rate']*100:5.2f} {m['mean_false_foreign_rate']*100:5.2f} "
            f"{m['mean_inference_ms']:6.1f}"
        )

    lines.append("-" * len(header))
    o = overall
    lines.append(
        f"{'ALL':<3s} {'OVERALL':<18s} {o['num_images']:>4d} "
        f"{_fmt_val(o['mean_ap50_box'])} {_fmt_val(o['mean_ap50_95_box'])} "
        f"{_fmt_val(o['mean_ap50_mask'])} {_fmt_val(o['mean_ap50_95_mask'])} "
        f"{o['mean_count_error_pct']:6.2f} {o['mean_missed_rate']*100:6.2f} "
        f"{o['mean_merged_rate']*100:6.2f} {o['mean_split_rate']*100:6.2f} "
        f"{o['mean_duplicate_rate']*100:5.2f} {o['mean_false_foreign_rate']*100:5.2f} "
        f"{o['mean_inference_ms']:6.1f}"
    )
    lines.append("")
    lines.append("Failure breakdown (counts across all categories):")
    fc = o["failure_counts"]
    for fk in FAILURE_TYPES:
        lines.append(f"  {fk:<24s}: {fc[fk]:>5d}  ({fc[fk] / max(o.get('total_gt_rice',1),1)*100:5.2f}% of GT rice)")
    lines.append("")
    lines.append("Legend: AP50_b = box AP@0.50, mAP_b = box AP@0.50:0.95")
    lines.append("        AP50_m = mask AP@0.50, mAP_m = mask AP@0.50:0.95")
    lines.append("        CntE% = count error %, Miss% = missed grain %, Mrg% = merged %, Spl% = split %")
    lines.append("        Dup% = duplicate %, FF% = false foreign %, ms = inference time per image (ms)")
    lines.append("=" * 98)
    return "\n".join(lines)


# ── Main Evaluation Loop ──────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate instance segmentation models on the rice-quality-ai test splits."
    )
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (.pt for YOLO, .pth for MaskRCNN).")
    parser.add_argument("--model-type", type=str, default="auto",
                        choices=["auto", "yolo", "maskrcnn", "classical"],
                        help="Model inference engine. 'auto' picks by weights path or availability.")
    parser.add_argument("--split", type=str, default="test",
                        help="COCO split name used to locate image subfolder. Default 'test'.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap total images evaluated from COCO split.")
    parser.add_argument("--max-images-per-category", type=int, default=None,
                        help="Cap images per category from tests/datasets/ folders.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size hint (currently single-image inference).")
    parser.add_argument("--generate-failure-images", action="store_true",
                        help="Write 4-panel diagnostic images for all evaluated examples.")
    parser.add_argument("--failures-only", action="store_true",
                        help="Only save failure images for images with any FP/FN (not perfect).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions. Default 0.25.")
    parser.add_argument("--device", type=str, default="",
                        help="Device ('' auto, 'cpu', 'cuda:0'). YOLO/MaskRCNN only.")
    parser.add_argument("--coco-ann", type=str, default=str(DEFAULT_COCO_ANN),
                        help=f"Path to COCO instances JSON. Default: {DEFAULT_COCO_ANN}")
    parser.add_argument("--coco-img-dir", type=str, default=str(DEFAULT_COCO_IMG_DIR.parent),
                        help="Parent dir of COCO images (contains {train,val,test} subdirs).")
    parser.add_argument("--tests-datasets-dir", type=str, default=str(DEFAULT_TEST_DATASETS_DIR),
                        help=f"Path to tests/datasets/ folder. Default: {DEFAULT_TEST_DATASETS_DIR}")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help=f"Output directory for results. Default: {DEFAULT_RESULTS_DIR}")
    args = parser.parse_args()

    weights_path = Path(args.weights) if args.weights else None
    if args.model_type == "auto":
        model_type = auto_detect_model_type(weights_path)
        logger.info(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = out_dir / "failures"
    if args.generate_failure_images:
        failures_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(model_type, weights_path, device=args.device, conf=args.conf)
    logger.info(f"Instantiated predictor: {model_type}")

    coco_images = load_coco_split(Path(args.coco_ann), Path(args.coco_img_dir),
                                  split=args.split, max_images=args.max_images)
    tds_images = load_tests_datasets(Path(args.tests_datasets_dir),
                                     max_images_per_category=args.max_images_per_category)
    all_records = coco_images + tds_images
    logger.info(f"Total evaluation records: {len(all_records)}")

    if not all_records:
        logger.error("No evaluation records could be loaded. Check paths and data.")
        sys.exit(1)

    per_image_results: List[Dict] = []
    per_category_images: Dict[str, List[Dict]] = {c: [] for c in CATEGORIES_9}

    for idx, rec in enumerate(all_records):
        img_path: Path = rec["img_path"]
        cat: str = rec["category"]
        h, w = rec["h"], rec["w"]
        fname = rec["file_name"]
        gt_anns: List[Dict] = rec["gt_anns"]

        logger.info(f"[{idx+1}/{len(all_records)}] cat={cat} {fname}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Failed to read {img_path}; skipping.")
            continue
        if img_bgr.shape[:2] != (h, w):
            h, w = img_bgr.shape[:2]
            gt_anns = _annotations_to_dicts(rec["gt_anns"], h, w) if rec["source"] == "coco" else gt_anns

        try:
            pred_anns, inf_ms = predictor.predict(img_bgr)
        except Exception as e:
            logger.error(f"Predictor failed on {fname}: {e}")
            pred_anns, inf_ms = [], 0.0

        metrics = compute_metrics_for_image(gt_anns, pred_anns, h, w)
        metrics["inference_ms"] = inf_ms

        record = {
            "image_id": rec["image_id"],
            "file_name": fname,
            "source": rec["source"],
            "category": cat,
            "img_path": str(img_path),
            **metrics,
        }
        per_image_results.append(record)
        per_category_images[cat].append(record)

        if args.generate_failure_images:
            has_issue = (
                metrics["tp_fp_fn"]["FP"] > 0
                or metrics["tp_fp_fn"]["FN"] > 0
                or metrics["count_error_pct"] > 0
                or any(metrics["failures"].get(k, 0) > 0 for k in FAILURE_TYPES)
            )
            if (not args.failures_only) or has_issue:
                out_img_path = failures_dir / cat / Path(fname).with_suffix(".png").name
                gt_masks = [a["mask"] for a in gt_anns if a["mask"] is not None]
                pred_masks = [a.get("mask") for a in pred_anns if a.get("mask") is not None]
                generate_4_panel_failure(img_bgr, gt_masks, pred_masks, out_img_path)

    per_category_summary: Dict[str, Dict] = {}
    for cat in CATEGORIES_9:
        per_category_summary[cat] = _aggregate_category_metrics(per_category_images[cat])
        per_category_summary[cat]["category_name"] = CATEGORY_NAMES.get(cat, cat)

    overall_images = [r for r in per_image_results]
    overall = _aggregate_category_metrics(overall_images)
    overall["num_categories_with_data"] = sum(
        1 for c in CATEGORIES_9 if per_category_summary[c]["num_images"] > 0
    )

    eval_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_type": model_type,
        "weights_path": str(weights_path) if weights_path else None,
        "split": args.split,
        "pycocotools_used": PYCOCOTOOLS_AVAILABLE,
        "yolo_available": YOLO_AVAILABLE,
        "maskrcnn_available": MASKRCNN_AVAILABLE,
        "overall": overall,
        "per_category": per_category_summary,
    }

    with open(out_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, default=float)

    with open(out_dir / "per_image_results.json", "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, indent=2, default=float)

    summary_txt = format_summary_txt(overall, per_category_summary, model_type, weights_path)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)

    logger.info("===== Evaluation Complete =====")
    logger.info(f"Results written to {out_dir}")
    logger.info(f"  eval_report.json         — structured JSON report")
    logger.info(f"  per_image_results.json   — per-image metrics")
    logger.info(f"  summary.txt              — human-readable table")
    if args.generate_failure_images:
        logger.info(f"  failures/<cat>/*.png     — 4-panel diagnostic images")
    print("\n" + summary_txt)


if __name__ == "__main__":
    main()
