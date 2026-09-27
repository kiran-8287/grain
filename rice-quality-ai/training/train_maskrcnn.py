"""
Mask R-CNN ResNet50-FPN-v2 Instance Segmentation Training Pipeline for Rice Quality AI.

Baseline comparison model for YOLOv8 segmentation. Uses torchvision Mask R-CNN with
custom COCO dataset loader, custom training loop, and simple mAP evaluation.

Supports:
- COCO pretrained initialization (DEFAULT weights)
- Fine-tune from existing checkpoint via --resume
- CLI argument overrides
- Automatic hardware detection (CPU/GPU)
- Gradient clipping and early stopping
- Structured run logging to training/run_log.md
"""

import argparse
import json
import logging
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2,
        MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import box_iou
except ImportError as e:
    logger.error(f"Required package not installed: {e}")
    logger.error("Install PyTorch and torchvision:")
    logger.error("  pip install torch torchvision")
    sys.exit(1)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logger.warning("pycocotools not installed — using built-in simplified mAP evaluation")


NUM_CLASSES = 2
CLASS_NAMES = {0: "background", 1: "rice_grain"}


def collate_fn(batch: List[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any]]:
    """Collate function for variable-length targets in Mask R-CNN batches."""
    return tuple(zip(*batch))


def get_transforms(train: bool) -> transforms.Compose:
    """
    Get data transforms.

    For training: random horizontal flip + ToTensor + ImageNet normalization.
    For validation: ToTensor + ImageNet normalization.

    Max 800px resize is handled in the model (min_size, max_size params).
    """
    transform_list = []
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return transforms.Compose(transform_list)


class CocoRiceDataset(Dataset):
    """
    Custom COCO-format Dataset for rice grain instance segmentation.

    Loads images from images/{split}/ and annotations from annotations/instances_{split}.json
    under the data_root directory.

    Returns tuples of (image_tensor, target_dict) where target_dict has:
        - boxes: FloatTensor[N, 4] in x1y1x2y2 format
        - labels: Int64Tensor[N] (class ids, 1-indexed: 1=rice_grain)
        - masks: UInt8Tensor[N, H, W] binary instance masks
        - image_id: Int64Tensor[1]
        - area: FloatTensor[N]
        - iscrowd: Int64Tensor[N]
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        transforms: Optional[transforms.Compose] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms

        self.img_dir = self.data_root / "images" / split
        ann_path = self.data_root / "annotations" / f"instances_{split}.json"

        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        with open(ann_path, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.category_id_to_label = {}
        for cat in self.coco_data.get("categories", []):
            cid = cat["id"]
            name = cat["name"]
            if name == "rice_grain":
                self.category_id_to_label[cid] = 1
            elif name == "foreign_matter":
                self.category_id_to_label[cid] = 2

        self.image_ids = [img["id"] for img in self.coco_data["images"]]
        self.id_to_image = {img["id"]: img for img in self.coco_data["images"]}

        self.img_anns = defaultdict(list)
        for ann in self.coco_data.get("annotations", []):
            self.img_anns[ann["image_id"]].append(ann)

        logger.info(
            f"Dataset split='{split}': {len(self.image_ids)} images, "
            f"{sum(len(v) for v in self.img_anns.values())} annotations"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _polygon_to_mask(
        self, polygon: List[float], width: int, height: int
    ) -> np.ndarray:
        """Convert COCO polygon (flat [x1,y1,x2,y2,...]) to a binary uint8 mask."""
        try:
            import cv2
        except ImportError:
            return self._polygon_to_mask_simple(polygon, width, height)

        pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        pts_int = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts_int], 1)
        return mask

    def _polygon_to_mask_simple(
        self, polygon: List[float], width: int, height: int
    ) -> np.ndarray:
        """Fallback polygon rasterizer when cv2 is unavailable."""
        from PIL import Image, ImageDraw

        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
        return np.array(mask_img, dtype=np.uint8)

    def _rle_to_mask(self, rle: Dict, width: int, height: int) -> np.ndarray:
        """Decode COCO RLE segmentation to binary mask."""
        mask = np.zeros(height * width, dtype=np.uint8)
        counts = rle.get("counts", [])
        if isinstance(counts, str):
            counts = self._decode_rle_string(counts)

        pos = 0
        val = 0
        for cnt in counts:
            if pos + cnt <= len(mask):
                mask[pos : pos + cnt] = val
            pos += cnt
            val = 1 - val

        return mask.reshape((height, width), order="F")

    def _decode_rle_string(self, s: str) -> List[int]:
        """Simple MS COCO RLE string decoder."""
        counts = []
        p = 0
        while p < len(s):
            x = 0
            k = 0
            more = True
            while more:
                c = ord(s[p]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(counts) > 2:
                x += counts[-2]
            counts.append(x)
        return counts

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_info = self.id_to_image[img_id]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        img_path = self.img_dir / file_name
        img = Image.open(img_path).convert("RGB")

        annotations = self.img_anns.get(img_id, [])

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in annotations:
            label = self.category_id_to_label.get(ann.get("category_id"))
            if label is None:
                continue

            seg = ann.get("segmentation")
            bbox = ann.get("bbox")
            area = float(ann.get("area", 0.0))
            crowd = int(ann.get("iscrowd", 0))

            mask = None
            if isinstance(seg, list) and len(seg) > 0:
                for poly in seg:
                    if len(poly) >= 6:
                        poly_mask = self._polygon_to_mask(poly, width, height)
                        if mask is None:
                            mask = poly_mask
                        else:
                            mask = np.maximum(mask, poly_mask)
            elif isinstance(seg, dict) and "counts" in seg:
                mask = self._rle_to_mask(seg, width, height)

            if mask is None or mask.sum() == 0:
                if bbox and len(bbox) == 4:
                    bx, by, bw, bh = [int(v) for v in bbox]
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[by : by + bh, bx : bx + bw] = 1
                else:
                    continue

            if bbox and len(bbox) == 4:
                bx, by, bw, bh = bbox
                x1 = float(bx)
                y1 = float(by)
                x2 = float(bx + bw)
                y2 = float(by + bh)
            else:
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                x1 = float(xs.min())
                y1 = float(ys.min())
                x2 = float(xs.max())
                y2 = float(ys.max())

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            masks.append(mask)
            if area <= 0:
                area = float(np.sum(mask))
            areas.append(area)
            iscrowd.append(crowd)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.as_tensor([img_id], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        img_np = np.array(img)
        if self.transforms is not None:
            img_tensor = self.transforms(img)
        else:
            img_tensor = torch.as_tensor(
                img_np.transpose((2, 0, 1)) / 255.0, dtype=torch.float32
            )

        return img_tensor, target


def compute_simple_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Simplified AP@iou_threshold calculation for boxes and masks.

    This is a count-based approximation when pycocotools is unavailable.
    For each image/class, computes precision-recall by sorting detections by
    score, then approximates AP as the mean precision at 11 recall points.

    Returns dict with:
        - overall/mAP: box AP across classes
        - mask/overall/mAP: mask AP across classes
        - per_class: dict of class_id -> {box_AP, mask_AP}
        - class_names: mapping dict
    """
    per_class_box = defaultdict(list)
    per_class_mask = defaultdict(list)

    for pred, tgt in zip(predictions, targets):
        if "boxes" not in pred or len(pred["boxes"]) == 0:
            continue

        pred_boxes = pred["boxes"].cpu()
        pred_scores = pred["scores"].cpu()
        pred_labels = pred["labels"].cpu()
        pred_masks = pred.get("masks")
        if pred_masks is not None:
            pred_masks = pred_masks.cpu()
            if pred_masks.dim() == 4:
                pred_masks = pred_masks[:, 0]
            pred_masks = (pred_masks > 0.5).to(torch.uint8)

        tgt_boxes = tgt["boxes"].cpu()
        tgt_labels = tgt["labels"].cpu()
        tgt_masks = tgt["masks"].cpu()

        all_classes = torch.unique(torch.cat([pred_labels, tgt_labels])).tolist()

        for cls in all_classes:
            cls_mask_p = pred_labels == cls
            cls_mask_t = tgt_labels == cls

            p_boxes = pred_boxes[cls_mask_p]
            p_scores = pred_scores[cls_mask_p]
            p_masks = pred_masks[cls_mask_p] if pred_masks is not None else None

            t_boxes = tgt_boxes[cls_mask_t]
            t_masks = tgt_masks[cls_mask_t]

            n_gt = len(t_boxes)
            n_pred = len(p_boxes)

            if n_gt == 0 and n_pred == 0:
                continue
            if n_gt == 0:
                per_class_box[cls].append(0.0)
                per_class_mask[cls].append(0.0)
                continue
            if n_pred == 0:
                per_class_box[cls].append(0.0)
                per_class_mask[cls].append(0.0)
                continue

            sort_idx = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[sort_idx]
            if p_masks is not None:
                p_masks = p_masks[sort_idx]

            box_iou_mat = box_iou(p_boxes, t_boxes) if len(p_boxes) > 0 and len(t_boxes) > 0 else torch.zeros(n_pred, n_gt)

            mask_ious = torch.zeros(n_pred, n_gt)
            if p_masks is not None and t_masks is not None:
                for i in range(min(n_pred, 50)):
                    pm = p_masks[i].flatten()
                    for j in range(min(n_gt, 50)):
                        tm = t_masks[j].flatten()
                        inter = (pm & tm).sum().item()
                        union = (pm | tm).sum().item()
                        if union > 0:
                            mask_ious[i, j] = inter / union

            tp_box = torch.zeros(n_pred)
            tp_mask = torch.zeros(n_pred)
            matched_t = set()

            for i in range(n_pred):
                best_box_iou = 0.0
                best_mask_iou = 0.0
                best_j = -1
                for j in range(n_gt):
                    if j in matched_t:
                        continue
                    bi = box_iou_mat[i, j].item()
                    mi = mask_ious[i, j].item()
                    if bi > best_box_iou:
                        best_box_iou = bi
                        best_mask_iou = mi
                        best_j = j

                if best_box_iou >= iou_threshold:
                    tp_box[i] = 1
                    matched_t.add(best_j)
                if best_mask_iou >= iou_threshold:
                    tp_mask[i] = 1

            cumsum_tp_box = torch.cumsum(tp_box, dim=0)
            cumsum_fp_box = torch.cumsum(1 - tp_box, dim=0)
            cumsum_tp_mask = torch.cumsum(tp_mask, dim=0)
            cumsum_fp_mask = torch.cumsum(1 - tp_mask, dim=0)

            precision_box = cumsum_tp_box / (cumsum_tp_box + cumsum_fp_box + 1e-10)
            recall_box = cumsum_tp_box / (n_gt + 1e-10)
            precision_mask = cumsum_tp_mask / (cumsum_tp_mask + cumsum_fp_mask + 1e-10)
            recall_mask = cumsum_tp_mask / (n_gt + 1e-10)

            recalls_pts = torch.linspace(0, 1, 11)
            ap_box = 0.0
            ap_mask = 0.0
            for r in recalls_pts:
                precs_box = precision_box[recall_box >= r]
                precs_mask = precision_mask[recall_mask >= r]
                ap_box += precs_box.max().item() if len(precs_box) > 0 else 0.0
                ap_mask += precs_mask.max().item() if len(precs_mask) > 0 else 0.0
            ap_box /= 11.0
            ap_mask /= 11.0

            per_class_box[cls].append(ap_box)
            per_class_mask[cls].append(ap_mask)

    result: Dict[str, Any] = {
        "overall": {},
        "mask": {"overall": {}},
        "per_class": {},
        "class_names": CLASS_NAMES,
    }

    all_cls = set(per_class_box.keys()) | set(per_class_mask.keys())
    box_aps = []
    mask_aps = []
    for cls in sorted(all_cls):
        box_vals = per_class_box.get(cls, [])
        mask_vals = per_class_mask.get(cls, [])
        box_ap = float(np.mean(box_vals)) if box_vals else 0.0
        mask_ap = float(np.mean(mask_vals)) if mask_vals else 0.0
        result["per_class"][int(cls)] = {"box_AP": box_ap, "mask_AP": mask_ap}
        if int(cls) != 0:
            box_aps.append(box_ap)
            mask_aps.append(mask_ap)

    result["overall"]["mAP"] = float(np.mean(box_aps)) if box_aps else 0.0
    result["mask"]["overall"]["mAP"] = float(np.mean(mask_aps)) if mask_aps else 0.0

    return result


def get_maskrcnn_model(num_classes: int) -> nn.Module:
    """
    Create Mask R-CNN ResNet50-FPN-v2 with custom detection and mask heads.

    Parameters
    ----------
    num_classes : int
        Total classes including background (e.g., 2 = bg + rice_grain).
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, max_size=800, min_size=800)

    in_features_roi = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_roi, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[nn.Module, Optional[optim.Optimizer], Optional[Any], int, float]:
    """
    Load training checkpoint from disk.

    Returns (model, optimizer, scheduler, start_epoch, best_map).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    best_map = ckpt.get("best_map", 0.0)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    logger.info(
        f"Checkpoint loaded: epoch={start_epoch}, best_mask_map={best_map:.4f} — {path}"
    )
    return model, optimizer, scheduler, start_epoch, best_map


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    best_map: float,
) -> None:
    """Save training checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_map": best_map,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def get_existing_run_index(log_path: Path) -> int:
    """Scan existing run entries in run_log.md and return the next run number N."""
    if not log_path.exists():
        return 1
    try:
        content = log_path.read_text(encoding="utf-8")
        run_nums = []
        for line in content.splitlines():
            if line.startswith("## Run "):
                try:
                    num_part = line.split(" - ")[0].replace("## Run ", "").strip()
                    run_nums.append(int(num_part))
                except (ValueError, IndexError):
                    continue
        return max(run_nums) + 1 if run_nums else 1
    except Exception:
        return 1


def append_run_log(log_path: Path, run_data: Dict) -> None:
    """Append a structured Mask R-CNN run entry to run_log.md."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# Rice Segmentation Training Run Log\n\n"
        "Automatically generated by training scripts (train_yolo_seg.py / train_maskrcnn.py).\n\n"
    )

    early_stop_note = ""
    if run_data.get("early_stop_epoch"):
        early_stop_note = f" (early stop at epoch {run_data['early_stop_epoch']})"

    entry = (
        f"## Run {run_data['run_number']} - {run_data['date']}\n\n"
        f"- **Model:** maskrcnn_resnet50_fpn_v2\n"
        f"- **Dataset:** GrainSet v3 + synthetic\n"
        f"- **Epochs:** {run_data['total_epochs']}{early_stop_note}\n"
        f"- **Best val box mAP@50:** {run_data['best_box_map50']}\n"
        f"- **Best val mask mAP@50:** {run_data['best_mask_map50']}\n"
        f"- **Duration:** {run_data['duration_hours']}h {run_data['duration_min']}m\n"
        f"- **Hardware:** {run_data['hardware']}\n\n"
    )

    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(entry)
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    logger.info(f"Run log appended to {log_path}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip_norm: float = 5.0,
) -> float:
    """Run one training epoch; returns average total loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            loss_str = ", ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            logger.info(
                f"  [Train E{epoch} B{batch_idx}/{len(data_loader)}] "
                f"total={losses.item():.4f}, {loss_str}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Returns dict with:
        - box_mAP_50, mask_mAP_50 (from simple or pycocotools evaluation)
    """
    model.eval()
    all_preds: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    for batch_idx, (images, targets) in enumerate(data_loader):
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)

        for out, tgt in zip(outputs, targets):
            pred_cpu = {k: v.detach().cpu() for k, v in out.items()}
            tgt_cpu = {k: v.detach().cpu() for k, v in tgt.items()}
            all_preds.append(pred_cpu)
            all_targets.append(tgt_cpu)

        if batch_idx % 10 == 0:
            logger.info(f"  [Val B{batch_idx}/{len(data_loader)}] done")

    metrics = compute_simple_map(all_preds, all_targets, iou_threshold=0.5)
    box_map50 = metrics["overall"]["mAP"]
    mask_map50 = metrics["mask"]["overall"]["mAP"]

    logger.info(f"  Validation box_mAP@50  = {box_map50:.4f}")
    logger.info(f"  Validation mask_mAP@50 = {mask_map50:.4f}")

    for cls_id, cls_metrics in metrics.get("per_class", {}).items():
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        logger.info(
            f"    Class {cls_id} ({cls_name}): box_AP={cls_metrics['box_AP']:.4f}, "
            f"mask_AP={cls_metrics['mask_AP']:.4f}"
        )

    return {
        "box_mAP_50": box_map50,
        "mask_mAP_50": mask_map50,
    }


def detect_hardware() -> Dict:
    """Detect available hardware and return configuration recommendations."""
    info = {
        "gpu_available": False,
        "device_name": "CPU",
        "num_workers": 0 if platform.system() == "Windows" else 4,
    }
    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["device_name"] = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if gpu_mem < 8 and platform.system() != "Windows":
            info["num_workers"] = 2
    else:
        info["device_name"] = platform.processor() or "CPU"
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN ResNet50-FPN-v2 for rice grain segmentation"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/processed/coco",
        help="COCO dataset root (relative to PROJECT_ROOT or absolute)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path (.pth), e.g. models/maskrcnn/last/model.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cpu' or 'cuda' (auto if omitted)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Dataloader workers (0 on Windows by default)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience on val mask mAP@50",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Gradient clipping max norm",
    )
    args = parser.parse_args()

    hw = detect_hardware()
    logger.info(
        f"Hardware detected: GPU={hw['gpu_available']}, device={hw['device_name']}, "
        f"workers={hw['num_workers']}"
    )

    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if hw["gpu_available"] else "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    num_workers = args.num_workers if args.num_workers is not None else hw["num_workers"]

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    logger.info(f"Data root: {data_root}")

    if not data_root.exists():
        logger.error(f"Data root directory does not exist: {data_root}")
        sys.exit(1)

    available_splits = [
        p.name for p in (data_root / "images").iterdir()
        if (data_root / "images" / p).is_dir()
    ]
    logger.info(f"Available image splits: {available_splits}")

    train_split = "train" if "train" in available_splits else available_splits[0] if available_splits else None
    val_split = "val" if "val" in available_splits else None

    if train_split is None:
        logger.error("No training split found in data_root/images/")
        sys.exit(1)

    logger.info(f"Train split: {train_split}")
    logger.info(f"Val split:   {val_split if val_split else '(none found, will skip val)'}")

    train_dataset = CocoRiceDataset(data_root, train_split, transforms=get_transforms(train=True))
    val_dataset = None
    if val_split:
        try:
            val_dataset = CocoRiceDataset(data_root, val_split, transforms=get_transforms(train=False))
        except FileNotFoundError as e:
            logger.warning(f"Validation dataset unavailable: {e}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device_str == "cuda"),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(device_str == "cuda"),
        )

    logger.info("Building Mask R-CNN ResNet50-FPN-v2 model...")
    model = get_maskrcnn_model(NUM_CLASSES)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    start_epoch = 0
    best_mask_map = 0.0
    best_box_map = 0.0
    best_epoch = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        if resume_path.exists():
            model, optimizer, scheduler, start_epoch, best_mask_map = load_checkpoint(
                resume_path, model, optimizer, scheduler
            )
        else:
            logger.error(f"Resume checkpoint not found: {resume_path}")
            sys.exit(1)

    best_dir = PROJECT_ROOT / "models" / "maskrcnn" / "best"
    last_dir = PROJECT_ROOT / "models" / "maskrcnn" / "last"
    best_dir.mkdir(parents=True, exist_ok=True)
    last_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Epochs:         {args.epochs}")
    logger.info(f"Batch size:     {args.batch_size}")
    logger.info(f"LR:             {args.lr}")
    logger.info(f"Optimizer:      SGD (momentum=0.9, weight_decay=5e-4)")
    logger.info(f"Scheduler:      StepLR (step=20, gamma=0.1)")
    logger.info(f"Grad clip:      {args.grad_clip}")
    logger.info(f"Early stop pat: {args.patience}")
    logger.info(f"Num workers:    {num_workers}")
    logger.info(f"Train samples:  {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val samples:    {len(val_dataset)}")
    logger.info("=" * 60)

    start_time = time.time()
    patience_counter = 0
    early_stop_epoch = None

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")

        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.grad_clip
        )
        logger.info(f"Epoch {epoch} train avg loss: {avg_train_loss:.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Current learning rate: {current_lr:.6f}")

        val_box_map = 0.0
        val_mask_map = 0.0
        if val_loader is not None:
            logger.info("Running validation...")
            val_metrics = evaluate(model, val_loader, device)
            val_box_map = val_metrics["box_mAP_50"]
            val_mask_map = val_metrics["mask_mAP_50"]

        save_checkpoint(
            last_dir / "model.pth", model, optimizer, scheduler, epoch, best_mask_map
        )

        improved = False
        if val_loader is not None:
            if val_mask_map > best_mask_map:
                best_mask_map = val_mask_map
                best_box_map = val_box_map
                best_epoch = epoch
                save_checkpoint(
                    best_dir / "model.pth",
                    model, optimizer, scheduler, epoch, best_mask_map,
                )
                patience_counter = 0
                improved = True
                logger.info(
                    f"*** NEW BEST: mask_mAP@50={best_mask_map:.4f} "
                    f"(box_mAP@50={best_box_map:.4f}) at epoch {epoch} ***"
                )
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement for {patience_counter}/{args.patience} epochs "
                    f"(best mask_mAP@50={best_mask_map:.4f} at epoch {best_epoch})"
                )
                if patience_counter >= args.patience:
                    logger.info(
                        f"Early stopping triggered after {args.patience} epochs without improvement."
                    )
                    early_stop_epoch = epoch
                    break
        else:
            best_epoch = epoch

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} wall time: {epoch_time:.1f}s")

    total_duration = time.time() - start_time
    dur_h = int(total_duration // 3600)
    dur_m = int((total_duration % 3600) // 60)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best mask mAP@50:  {best_mask_map:.4f}")
    logger.info(f"Best box mAP@50:   {best_box_map:.4f}")
    logger.info(f"Best epoch:        {best_epoch}")
    logger.info(f"Total epochs:      {args.epochs}")
    logger.info(f"Training duration: {dur_h}h {dur_m}m")
    logger.info(f"Best model:        {best_dir / 'model.pth'}")
    logger.info(f"Last model:        {last_dir / 'model.pth'}")
    logger.info("=" * 60)

    log_path = PROJECT_ROOT / "training" / "run_log.md"
    run_idx = get_existing_run_index(log_path)

    best_box_str = f"{best_box_map:.4f}" if best_box_map else "N/A"
    best_mask_str = f"{best_mask_map:.4f}" if best_mask_map else "N/A"

    run_data = {
        "run_number": run_idx,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_epochs": args.epochs,
        "early_stop_epoch": early_stop_epoch if early_stop_epoch else None,
        "best_box_map50": best_box_str,
        "best_mask_map50": best_mask_str,
        "duration_hours": dur_h,
        "duration_min": dur_m,
        "hardware": hw["device_name"] + (" (GPU)" if hw["gpu_available"] else " (CPU)"),
    }
    append_run_log(log_path, run_data)


if __name__ == "__main__":
    main()
