"""
Post-processing for instance segmentation outputs.

Handles confidence filtering, mask-level NMS, global ID assignment,
geometry verification, touching detection, and tiling pipeline for
large or dense images.
"""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ml.config import get_project_root, get_threshold

PROJECT_ROOT = get_project_root()

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Post-processes raw instance segmentation detections.

    Provides confidence filtering, mask IoU NMS, global ID assignment,
    geometry verification, confidence labelling, touching detection,
    and a tiling pipeline for large images.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        mask_iou_threshold: float = 0.5,
        min_grain_area: int = 50,
        tile_size: int = 640,
        tile_overlap: int = 128,
    ):
        self.conf_threshold = conf_threshold
        self.mask_iou_threshold = mask_iou_threshold
        self.min_grain_area = min_grain_area
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

    def filter_by_confidence(
        self, detections: List[Dict], threshold: float
    ) -> List[Dict]:
        """
        Remove detections whose confidence is below the threshold.

        Each detection is expected to have keys: bbox, mask, confidence,
        category_id (plus optional extras).
        """
        return [d for d in detections if float(d.get("confidence", 0.0)) >= threshold]

    @staticmethod
    def _compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Compute IoU between two binary masks (same shape or broadcastable)."""
        a = mask_a.astype(bool)
        b = mask_b.astype(bool)
        intersection = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        if union == 0:
            return 0.0
        return intersection / union

    def mask_iou_nms(
        self, detections: List[Dict], iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Secondary NMS at mask level.

        For pairs where box IoU passed the first stage but mask IoU > threshold,
        keep the detection with higher confidence and discard the other.
        """
        if len(detections) <= 1:
            return list(detections)

        sorted_indices = sorted(
            range(len(detections)),
            key=lambda i: float(detections[i].get("confidence", 0.0)),
            reverse=True,
        )

        keep: List[int] = []
        suppressed: set = set()

        for idx_i, i in enumerate(sorted_indices):
            if i in suppressed:
                continue
            keep.append(i)
            det_i = detections[i]
            mask_i = det_i.get("mask")
            if mask_i is None:
                continue
            for j in sorted_indices[idx_i + 1 :]:
                if j in suppressed:
                    continue
                det_j = detections[j]
                mask_j = det_j.get("mask")
                if mask_j is None:
                    continue
                if mask_i.shape != mask_j.shape:
                    h = max(mask_i.shape[0], mask_j.shape[0])
                    w = max(mask_i.shape[1], mask_j.shape[1])
                    mi = np.zeros((h, w), dtype=bool)
                    mj = np.zeros((h, w), dtype=bool)
                    mi[: mask_i.shape[0], : mask_i.shape[1]] = mask_i.astype(bool)
                    mj[: mask_j.shape[0], : mask_j.shape[1]] = mask_j.astype(bool)
                    iou = self._compute_mask_iou(mi, mj)
                else:
                    iou = self._compute_mask_iou(mask_i, mask_j)
                if iou > iou_threshold:
                    suppressed.add(j)

        return [detections[k] for k in keep]

    def assign_global_ids(self, detections: List[Dict]) -> List[Dict]:
        """
        Add a sequential 'id' field starting at 1.

        Ordering: by confidence descending, then by y (bbox top), then by x
        (bbox left), so IDs are stable and visually intuitive.
        """
        def sort_key(d: Dict):
            bbox = d.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x, y = float(bbox[0]), float(bbox[1])
            else:
                x, y = 0.0, 0.0
            return (-float(d.get("confidence", 0.0)), y, x)

        ordered = sorted(detections, key=sort_key)
        for seq, det in enumerate(ordered, start=1):
            det["id"] = seq
        return ordered

    def verify_geometry(
        self,
        detections: List[Dict],
        min_area: int,
        img_hw: Tuple[int, int],
    ) -> List[Dict]:
        """
        Remove detections whose mask area is below min_area or whose bbox
        lies fully outside the image.
        """
        img_h, img_w = img_hw
        kept: List[Dict] = []
        for det in detections:
            mask = det.get("mask")
            if mask is None:
                continue
            area = int(np.sum(mask > 0))
            if area < min_area:
                continue
            bbox = det.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = bbox
                if x + w <= 0 or y + h <= 0 or x >= img_w or y >= img_h:
                    continue
            kept.append(det)
        return kept

    @staticmethod
    def compute_confidence_label(confidence: float) -> str:
        """Return 'HIGH', 'MEDIUM', or 'LOW' label for a confidence score."""
        if confidence >= 0.80:
            return "HIGH"
        if confidence >= 0.50:
            return "MEDIUM"
        return "LOW"

    def detect_touching(self, detections: List[Dict]) -> List[Dict]:
        """
        For each detection, set is_touching=True when its mask dilated with a
        3x3 kernel (1 iteration) intersects any *other* detection's mask.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        n = len(detections)
        for det in detections:
            det["is_touching"] = False

        masks = [
            (d.get("mask") if isinstance(d.get("mask"), np.ndarray) else None)
            for d in detections
        ]
        dilated = []
        for m in masks:
            if m is None:
                dilated.append(None)
                continue
            m_bin = (m > 0).astype(np.uint8)
            dilated.append(cv2.dilate(m_bin, kernel, iterations=1))

        for i in range(n):
            if dilated[i] is None:
                continue
            for j in range(i + 1, n):
                if dilated[j] is None:
                    continue
                di = dilated[i]
                dj = masks[j]
                if di is None or dj is None:
                    continue
                if di.shape != dj.shape:
                    h = max(di.shape[0], dj.shape[0])
                    w = max(di.shape[1], dj.shape[1])
                    di_p = np.zeros((h, w), dtype=np.uint8)
                    dj_p = np.zeros((h, w), dtype=np.uint8)
                    di_p[: di.shape[0], : di.shape[1]] = di
                    dj_p[: dj.shape[0], : dj.shape[1]] = (dj > 0).astype(np.uint8)
                    overlap = float(np.logical_and(di_p > 0, dj_p > 0).sum())
                else:
                    overlap = float(np.logical_and(di > 0, dj > 0).sum())
                if overlap > 0:
                    detections[i]["is_touching"] = True
                    detections[j]["is_touching"] = True
        return detections

    # ------------------------------------------------------------------
    # Tiling pipeline
    # ------------------------------------------------------------------

    def split_into_tiles(
        self,
        image: np.ndarray,
        tile_size: int,
        overlap: int,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Split an image into overlapping tiles.

        Returns list of (tile_image, (global_x1, global_y1, tile_w, tile_h)).
        """
        h, w = image.shape[:2]
        tiles: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        step = tile_size - overlap
        if step <= 0:
            step = tile_size

        y = 0
        while y < h:
            x = 0
            while x < w:
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                tile = image[y:y2, x:x2].copy()
                tiles.append((tile, (x, y, x2 - x, y2 - y)))
                if x2 >= w:
                    break
                x += step
            if y2 >= h:
                break
            y += step
        return tiles

    @staticmethod
    def map_tile_to_global(
        detection: Dict, tile_offset: Tuple[int, int]
    ) -> Dict:
        """
        Offset bbox and mask coordinates from tile-local to global image space.

        tile_offset = (global_x1, global_y1) for the top-left of the tile.
        """
        out = dict(detection)
        ox, oy = tile_offset

        bbox = out.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bx, by, bw, bh = bbox
            out["bbox"] = [int(bx + ox), int(by + oy), int(bw), int(bh)]

        mask = out.get("mask")
        if isinstance(mask, np.ndarray):
            mh, mw = mask.shape[:2]
            if ox > 0 or oy > 0:
                # Pad mask in-place for positive offsets; for negative offsets
                # the caller should not pass negative offsets (tiles always
                # start at non-negative global coordinates).
                pad_y = int(oy)
                pad_x = int(ox)
                new_mask = np.zeros(
                    (mh + pad_y, mw + pad_x), dtype=mask.dtype
                )
                new_mask[pad_y : pad_y + mh, pad_x : pad_x + mw] = mask
                out["mask"] = new_mask

        return out

    def merge_tile_detections(
        self,
        tile_results: List[Tuple[List[Dict], Tuple]],
        mask_iou_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Merge detections from multiple tiles into a single global list.

        tile_results: list of (detections_list, (gx1, gy1, tw, th))
        For any pair coming from different tiles whose mask IoU > threshold,
        keep the one with higher confidence.
        """
        all_dets: List[Dict] = []
        for dets, (gx1, gy1, _tw, _th) in tile_results:
            for d in dets:
                global_d = self.map_tile_to_global(d, (gx1, gy1))
                all_dets.append(global_d)

        if len(all_dets) <= 1:
            return all_dets

        sorted_indices = sorted(
            range(len(all_dets)),
            key=lambda i: float(all_dets[i].get("confidence", 0.0)),
            reverse=True,
        )

        keep: List[int] = []
        suppressed: set = set()

        for idx_i, i in enumerate(sorted_indices):
            if i in suppressed:
                continue
            keep.append(i)
            mask_i = all_dets[i].get("mask")
            if mask_i is None:
                continue
            for j in sorted_indices[idx_i + 1 :]:
                if j in suppressed:
                    continue
                mask_j = all_dets[j].get("mask")
                if mask_j is None:
                    continue
                if mask_i.shape != mask_j.shape:
                    h = max(mask_i.shape[0], mask_j.shape[0])
                    w = max(mask_i.shape[1], mask_j.shape[1])
                    mi = np.zeros((h, w), dtype=bool)
                    mj = np.zeros((h, w), dtype=bool)
                    mi[: mask_i.shape[0], : mask_i.shape[1]] = mask_i.astype(bool)
                    mj[: mask_j.shape[0], : mask_j.shape[1]] = mask_j.astype(bool)
                    iou = self._compute_mask_iou(mi, mj)
                else:
                    iou = self._compute_mask_iou(mask_i, mask_j)
                if iou > mask_iou_threshold:
                    suppressed.add(j)

        return [all_dets[k] for k in keep]

    # ------------------------------------------------------------------
    # Full pipeline orchestration
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        raw_detections: List[Dict],
        image_shape: Tuple[int, int],
        use_tiling: bool = False,
    ) -> List[Dict]:
        """
        Orchestrate the complete post-processing stack.

        Steps:
          1. confidence filter (threshold stored on this instance)
          2. (tile merge only if use_tiling is True and detections carry tile
             grouping info — here use_tiling skips tile merge when false)
          3. mask IoU NMS
          4. verify geometry (area + image bounds)
          5. assign global IDs
          6. detect touching
          7. attach confidence_label

        Returns the final detection list with enriched metadata.
        """
        detections = self.filter_by_confidence(raw_detections, self.conf_threshold)

        if use_tiling:
            grouped: List[Tuple[List[Dict], Tuple]] = []
            current_offset = (0, 0, 0, 0)
            bucket: List[Dict] = []
            for d in detections:
                offset = d.get("_tile_offset")
                if offset is None:
                    bucket.append(d)
                else:
                    grouped.append(([d], tuple(offset)))
            if bucket:
                grouped.append((bucket, current_offset))
            if grouped:
                detections = self.merge_tile_detections(
                    grouped, mask_iou_threshold=self.mask_iou_threshold
                )

        detections = self.mask_iou_nms(detections, iou_threshold=self.mask_iou_threshold)
        detections = self.verify_geometry(detections, self.min_grain_area, image_shape)
        detections = self.assign_global_ids(detections)
        detections = self.detect_touching(detections)

        for d in detections:
            conf = float(d.get("confidence", 0.0))
            d["confidence_label"] = self.compute_confidence_label(conf)

        return detections
