"""
Rice-presence gate (Case 1): "Does this image actually contain RICE?"

This is the single central decision point used by ml.pipeline BEFORE any
rice-grain analysis. It exists because "objects were detected" and "rice was
detected" are two different things:

    CASE A  foreign matter only   (e.g. 18 stones, 0 rice)   -> NOT RICE (stop)
    CASE B  rice + foreign matter (e.g. 15 rice, 2 stones)   -> rice analysis
            continues; the non-rice objects remain in the foreign-matter channel

A detection is accepted as RICE only when BOTH conditions hold:

    1. its class is the project's rice class, read from the model configuration
       (models/segmentation/class_mapping.json -> class_id 1, "rice_grain"), and
    2. its class confidence >= rice_gate.rice_confidence_threshold
       (configs/thresholds.json).

Foreign-matter classes (models/foreign_matter/class_mapping.json ->
stone | inorganic | organic | other_foreign_matter) are NEVER accepted as rice.

Detector status (honest limitation)
-----------------------------------
The Mask R-CNN "rice_grain" head is not trained yet (models/segmentation/model.pth
is absent, registry status "fallback_active"), so the per-object rice class
confidence below is produced by the project's classical-CV fallback classifier
(lightness / chroma / aspect ratio / surface texture) while keeping the project's
own class names and class IDs. When the segmentation model is trained, this gate
consumes real rice_grain detections without any class-name changes.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    joblib = None

from ml.config import get_model_info, get_project_root, get_threshold
from ml.foreign_matter import classify_non_rice_object

logger = logging.getLogger(__name__)

RICE_GATE_METHOD = "classical_cv_rice_gate"

STATUS_RICE = "RICE"
STATUS_NOT_RICE = "NOT_RICE"
STATUS_NOT_RICE_SPARSE = "NOT_RICE_SPARSE"
STATUS_NO_ANALYSABLE_RICE = "NO_ANALYSABLE_RICE"
STATUS_NO_RICE_CLASS = "NO_RICE_CLASS"

MESSAGE_RICE = "Rice grains detected — rice analysis continues."
MESSAGE_NOT_RICE = "No rice grains were detected in the uploaded image."
MESSAGE_NOT_RICE_SPARSE = (
    "Rice-like objects were detected, but too few of the detected objects are rice for "
    "this image to be treated as a rice sample. Rice analysis stopped."
)
MESSAGE_NO_ANALYSABLE_RICE = (
    "Rice-like objects were detected, but no valid rice grain instances could be "
    "analysed. Rice analysis stopped."
)
MESSAGE_NO_RICE_CLASS = (
    "The current detector has no rice class, so a reliable rice/no-rice gate cannot "
    "be implemented using this detector alone."
)

_LIMITATION = (
    "Classical-CV fallback classifier: the trained Mask R-CNN rice_grain head is not "
    "available yet (models/segmentation/model.pth missing). Gate confidence is a "
    "documented heuristic, not a trained model probability."
)

_MODEL_STATUS = "fallback_active (Mask R-CNN rice_grain head not trained)"

_DEFAULT_GATE_CONFIG: Dict[str, Any] = {
    "rice_confidence_threshold": 0.65,
    "min_object_area_pixels": 50,
    "min_rice_fraction_of_detections": 0.15,
    "min_objects_for_fraction_rule": 25,
    "weights": {
        "lightness": 0.35,
        "chroma": 0.05,
        "aspect_ratio": 0.5,
        "surface_texture": 0.1,
    },
    "lightness": {
        "full_credit_l_star": 78.0,
        "zero_credit_l_star": 45.0,
        "bright_background_l_star": 70.0,
        "full_credit_background_gap_fraction": 0.08,
        "zero_credit_background_gap_fraction": 0.30,
    },
    "chroma": {"full_credit_c_star": 12.0, "zero_credit_c_star": 30.0},
    "aspect_ratio": {"full_credit": 2.2, "zero_credit": 1.05},
    "surface_texture": {"full_credit_roughness": 0.02, "zero_credit_roughness": 0.08},
}

_CLASS_MAPPING_CACHE: Optional[Dict[str, Any]] = None
_RICE_GATE_MODEL_CACHE: Optional[Any] = None


def get_rice_gate_feature_names() -> List[str]:
    """Ordered feature vector used by the learned rice/no-rice classifier."""
    return [
        "mean_l_star",
        "chroma",
        "aspect_ratio",
        "surface_roughness",
        "mean_gray",
        "background_gap_ratio",
    ]


def _feature_vector_from_object(features: Dict[str, Any]) -> np.ndarray:
    """Convert a gate object's visual features into a learned-model feature vector."""
    mean_l = float(features.get("mean_l_star", 0.0))
    chroma = float(features.get("chroma", 0.0))
    aspect_ratio = float(features.get("aspect_ratio", 1.0))
    roughness = float(features.get("surface_roughness", 0.0))
    mean_gray = float(features.get("mean_gray", 0.0))
    background_gray = float(features.get("background_gray", 255.0))
    gap_ratio = max(0.0, background_gray - mean_gray) / max(background_gray, 1.0)
    return np.array([mean_l, chroma, aspect_ratio, roughness, mean_gray, gap_ratio], dtype=np.float32)


def load_rice_gate_model() -> Optional[Any]:
    """Load the learned rice-vs-non-rice model bundle if a trained artifact is present."""
    global _RICE_GATE_MODEL_CACHE
    if _RICE_GATE_MODEL_CACHE is not None:
        return _RICE_GATE_MODEL_CACHE

    if joblib is None:
        return None

    model_path = get_project_root() / "models" / "rice_gate" / "model.joblib"
    scaler_path = get_project_root() / "models" / "rice_gate" / "scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        _RICE_GATE_MODEL_CACHE = {"model": model, "scaler": scaler}
        return _RICE_GATE_MODEL_CACHE
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not load rice gate model from %s: %s", model_path, exc)
        return None


def _predict_learned_rice_probability(features: Dict[str, Any], model_bundle: Any) -> float:
    """Run the learned binary gate on a single object's features."""
    vector = _feature_vector_from_object(features).reshape(1, -1)

    model = model_bundle.get("model") if isinstance(model_bundle, dict) else model_bundle
    scaler = model_bundle.get("scaler") if isinstance(model_bundle, dict) else None
    if scaler is not None:
        vector = scaler.transform(vector)

    try:
        proba = model.predict_proba(vector)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(np.clip(proba[0, 1], 0.0, 1.0))
    except Exception:
        pass

    try:
        pred = model.predict(vector)
        if isinstance(pred, (list, tuple, np.ndarray)):
            pred = pred[0]
        if isinstance(pred, (float, int, np.floating, np.integer)):
            return float(np.clip(float(pred), 0.0, 1.0))
        return float(np.clip(float(pred[1] if hasattr(pred, "__len__") and len(pred) > 1 else pred), 0.0, 1.0))
    except Exception:
        return 0.0


def _gate_config() -> Dict[str, Any]:
    """Load the rice-gate thresholds, falling back to documented defaults."""
    cfg = dict(_DEFAULT_GATE_CONFIG)
    for key, default in _DEFAULT_GATE_CONFIG.items():
        loaded = get_threshold("rice_gate", key, None)
        if loaded is None:
            continue
        if isinstance(default, dict) and isinstance(loaded, dict):
            merged = dict(default)
            merged.update(loaded)
            cfg[key] = merged
        else:
            cfg[key] = loaded
    return cfg


def _read_class_mapping(model_name: str) -> Dict[str, int]:
    """Read a model class mapping file -> {class_name: class_id}."""
    info = get_model_info(model_name) or {}
    rel_path = info.get("class_mapping_path")
    if not rel_path:
        return {}
    path = get_project_root() / rel_path
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - defensive
        logger.warning("Could not read class mapping for %s: %s", model_name, exc)
        return {}
    return {
        str(name): int(class_id)
        for class_id, name in raw.items()
        if str(class_id).isdigit()
    }


def load_project_class_mappings(force_reload: bool = False) -> Dict[str, Any]:
    """
    Read the project's REAL class names / class IDs from the model configuration.

    The rice class is discovered from the segmentation class mapping (any class
    whose name contains "rice" and not "foreign"); the foreign-matter classes come
    from the foreign-matter class mapping. No class name is invented here.
    """
    global _CLASS_MAPPING_CACHE
    if _CLASS_MAPPING_CACHE is not None and not force_reload:
        return _CLASS_MAPPING_CACHE

    segmentation = _read_class_mapping("segmentation")
    foreign_matter = _read_class_mapping("foreign_matter")

    rice_classes = {
        name: class_id
        for name, class_id in segmentation.items()
        if "rice" in name.lower() and "foreign" not in name.lower()
    }
    rice_class_name = next(iter(rice_classes), None)

    _CLASS_MAPPING_CACHE = {
        "segmentation": segmentation,
        "foreign_matter": foreign_matter,
        "rice_class_name": rice_class_name,
        "rice_class_id": rice_classes.get(rice_class_name) if rice_class_name else None,
        "foreign_matter_class_names": list(foreign_matter.keys()),
        "class_mapping_source": {
            "segmentation": str(get_model_info("segmentation").get("class_mapping_path", "")),
            "foreign_matter": str(get_model_info("foreign_matter").get("class_mapping_path", "")),
        },
    }
    return _CLASS_MAPPING_CACHE


def _score_between(value: float, zero_credit: float, full_credit: float) -> float:
    """Linear ramp: 0.0 at `zero_credit`, 1.0 at `full_credit` (works ascending or descending)."""
    if full_credit == zero_credit:
        return 1.0 if value >= full_credit else 0.0
    ratio = (value - zero_credit) / (full_credit - zero_credit)
    return float(min(1.0, max(0.0, ratio)))


def _object_features(
    image_rgb: np.ndarray,
    object_mask: np.ndarray,
    contour: np.ndarray,
    background_gray: float,
) -> Dict[str, Any]:
    """Colour / shape / surface-texture features for one detected object."""
    mask_bool = object_mask > 0
    if int(np.count_nonzero(mask_bool)) == 0:
        return {}

    x, y, w, h = cv2.boundingRect(contour)
    crop = image_rgb[y:y + h, x:x + w]
    crop_mask = mask_bool[y:y + h, x:x + w]
    if crop.size == 0 or int(np.count_nonzero(crop_mask)) == 0:
        return {}

    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_px = lab[crop_mask]
    mean_l_255 = float(lab_px[:, 0].mean())
    mean_a = float(lab_px[:, 1].mean()) - 128.0
    mean_b = float(lab_px[:, 2].mean()) - 128.0

    mean_rgb = crop[crop_mask].astype(np.float32).mean(axis=0)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mean_gray = float(gray[crop_mask].mean())

    # Surface roughness on interior pixels only — object/background edges would
    # otherwise dominate the Laplacian response.
    interior = cv2.erode(
        crop_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    if int(np.count_nonzero(interior)) < 10:
        interior = crop_mask
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    roughness = float(np.abs(laplacian[interior]).mean()) / (mean_gray + 1.0)

    if len(contour) >= 5:
        _, (rect_w, rect_h), _ = cv2.minAreaRect(contour)
    else:
        rect_w, rect_h = float(w), float(h)
    aspect_ratio = max(rect_w, rect_h) / (min(rect_w, rect_h) + 1e-6)

    area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0

    saturation = float(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[:, :, 1][crop_mask].mean())

    return {
        "area_pixels": int(np.count_nonzero(mask_bool)),
        "bbox": [int(x), int(y), int(w), int(h)],
        "mean_rgb": [round(float(v), 2) for v in mean_rgb],
        "mean_gray": round(mean_gray, 2),
        "mean_l_star": round(mean_l_255 * 100.0 / 255.0, 2),
        "mean_a_star": round(mean_a, 2),
        "mean_b_star": round(mean_b, 2),
        "chroma": round(float(np.hypot(mean_a, mean_b)), 2),
        "saturation": round(saturation, 2),
        "aspect_ratio": round(float(aspect_ratio), 3),
        "solidity": round(float(solidity), 3),
        "surface_roughness": round(roughness, 5),
        "background_gray": round(float(background_gray), 2),
    }


def score_rice_confidence(
    features: Dict[str, Any], cfg: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Rice-class confidence in [0, 1] for one object, plus the per-cue breakdown.

    Cues (all configurable in configs/thresholds.json -> rice_gate):
      * lightness       — rice is bright; a bright background adds a penalty when
                          the object is far darker than that background
      * chroma          — milled rice is achromatic (low C*)
      * aspect_ratio    — rice grains are elongated
      * surface_texture — rice surface is smooth; stones are speckled / rough
    """
    weights = cfg["weights"]
    light_cfg = cfg["lightness"]
    chroma_cfg = cfg["chroma"]
    aspect_cfg = cfg["aspect_ratio"]
    texture_cfg = cfg["surface_texture"]

    lightness_score = _score_between(
        features["mean_l_star"],
        light_cfg["zero_credit_l_star"],
        light_cfg["full_credit_l_star"],
    )

    background_gray = float(features.get("background_gray", 0.0))
    background_l_star = float(
        cv2.cvtColor(
            np.array([[[int(round(background_gray))] * 3]], dtype=np.uint8),
            cv2.COLOR_RGB2LAB,
        )[0, 0, 0]
    ) * 100.0 / 255.0
    contrast_score = 1.0
    if background_l_star >= light_cfg["bright_background_l_star"]:
        gap_fraction = max(0.0, background_gray - features["mean_gray"]) / max(
            background_gray, 1.0
        )
        contrast_score = 1.0 - _score_between(
            gap_fraction,
            light_cfg["full_credit_background_gap_fraction"],
            light_cfg["zero_credit_background_gap_fraction"],
        )
        lightness_score *= 0.5 + 0.5 * contrast_score

    chroma_score = 1.0 - _score_between(
        features["chroma"],
        chroma_cfg["full_credit_c_star"],
        chroma_cfg["zero_credit_c_star"],
    )
    aspect_score = _score_between(
        features["aspect_ratio"], aspect_cfg["zero_credit"], aspect_cfg["full_credit"]
    )
    texture_score = 1.0 - _score_between(
        features["surface_roughness"],
        texture_cfg["full_credit_roughness"],
        texture_cfg["zero_credit_roughness"],
    )

    confidence = (
        weights["lightness"] * lightness_score
        + weights["chroma"] * chroma_score
        + weights["aspect_ratio"] * aspect_score
        + weights["surface_texture"] * texture_score
    )

    cues = {
        "lightness": round(lightness_score, 4),
        "background_contrast": round(contrast_score, 4),
        "chroma": round(chroma_score, 4),
        "aspect_ratio": round(aspect_score, 4),
        "surface_texture": round(texture_score, 4),
    }
    return float(min(1.0, max(0.0, confidence))), cues


def _gate_payload(
    detections: List[Dict[str, Any]],
    mappings: Dict[str, Any],
    threshold: float,
    cfg: Dict[str, Any],
    status: str,
    message: str,
    warnings: List[str],
) -> Dict[str, Any]:
    """
    Assemble the gate payload: counts, class mapping, detections, debug line.

    Sparsity guard: in crowded scenes (>= `min_objects_for_fraction_rule` detected
    objects) at least `min_rice_fraction_of_detections` of the objects must be
    rice-class. This stops a natural pebble bed — where a few elongated smooth
    pebbles happen to look rice-like — from being treated as a rice sample. Images
    with few objects (one grain, a handful of grains, rice + a couple of stones)
    are decided purely by the per-object rule.
    """
    total = len(detections)
    rice = [d for d in detections if d["is_rice"]]
    foreign = [d for d in detections if not d["is_rice"]]
    rice_confidences = [d["confidence"] for d in rice]
    rice_fraction = (len(rice) / total) if total else 0.0

    crowded = total >= int(cfg.get("min_objects_for_fraction_rule", 25))
    min_fraction = float(cfg.get("min_rice_fraction_of_detections", 0.0))
    sparse = bool(rice) and crowded and rice_fraction < min_fraction

    if sparse:
        status = STATUS_NOT_RICE_SPARSE
        message = MESSAGE_NOT_RICE_SPARSE

    has_rice = bool(rice) and not sparse
    gate_result = "PASSED" if has_rice else "FAILED"
    reason = (
        "rice_detections_too_sparse"
        if sparse
        else ("rice_detected" if rice else "no_rice_class_detections")
    )

    debug_line = (
        f"Total detections: {total} | Rice detections: {len(rice)} | "
        f"Foreign matter detections: {len(foreign)} | "
        f"Rice fraction: {rice_fraction:.3f} "
        f"(min {min_fraction:.2f} when >= {int(cfg.get('min_objects_for_fraction_rule', 25))} objects)"
        f" | Rice confidence threshold: {threshold} | Rice gate: {gate_result}"
    )

    return {
        "status": status,
        "has_rice": has_rice,
        "analysis_stopped": not has_rice,
        "message": message,
        "reason": reason,
        "rice_class_name": mappings.get("rice_class_name"),
        "rice_class_id": mappings.get("rice_class_id"),
        "foreign_matter_classes": mappings.get("foreign_matter_class_names", []),
        "rice_confidence_threshold": threshold,
        "total_detections": total,
        "rice_detections": len(rice),
        "foreign_matter_detections": len(foreign),
        "rice_fraction": round(rice_fraction, 4),
        "min_rice_fraction_required": min_fraction if crowded else None,
        "rice_confidence_mean": (
            round(float(np.mean(rice_confidences)), 4) if rice_confidences else 0.0
        ),
        "rice_confidence_max": (
            round(float(np.max(rice_confidences)), 4) if rice_confidences else 0.0
        ),
        "detections": detections,
        "class_mapping": {
            "segmentation": mappings.get("segmentation", {}),
            "foreign_matter": mappings.get("foreign_matter", {}),
        },
        "class_mapping_source": mappings.get("class_mapping_source", {}),
        "method": RICE_GATE_METHOD,
        "model_status": _MODEL_STATUS,
        "limitation": _LIMITATION,
        "debug": debug_line,
        "warnings": warnings,
    }


def _empty_gate_payload(
    mappings: Dict[str, Any],
    threshold: float,
    cfg: Dict[str, Any],
    status: str,
    message: str,
    warnings: List[str],
) -> Dict[str, Any]:
    """Gate payload for images without any usable object (blank / no rice class)."""
    payload = _gate_payload([], mappings, threshold, cfg, status, message, warnings)
    payload["confidence"] = 0.0
    payload["estimated_grain_count"] = 0
    payload["total_objects_detected"] = 0
    return payload


def detect_rice_detections(image_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect foreground objects and classify EACH one as rice or foreign matter.

    Rice-class detections use the model configuration's rice class; every other
    object keeps the project's foreign-matter class vocabulary.
    """
    cfg = _gate_config()
    mappings = load_project_class_mappings()
    threshold = float(cfg["rice_confidence_threshold"])

    # Same background-aware foreground extraction as the grain segmenter, so the
    # gate and the segmenter always look at the same objects.
    from ml.segmentation import extract_foreground_mask

    binary, _is_dark_bg, meta = extract_foreground_mask(image_rgb)
    background_gray = float(meta.get("bg_gray", 255.0))
    if meta.get("is_blank", False) or int(np.count_nonzero(binary)) == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = float(cfg["min_object_area_pixels"])
    detections: List[Dict[str, Any]] = []
    object_id = 0

    learned_model = load_rice_gate_model()
    model_active = learned_model is not None

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        object_mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)

        features = _object_features(image_rgb, object_mask, contour, background_gray)
        if not features:
            continue

        heuristic_confidence, cues = score_rice_confidence(features, cfg)
        if model_active:
            learned_confidence = _predict_learned_rice_probability(features, learned_model)
            confidence = 0.7 * learned_confidence + 0.3 * heuristic_confidence
            cues["learned_probability"] = round(learned_confidence, 4)
            cues["fused_probability"] = round(confidence, 4)
            model_source = "learned_rice_gate"
        else:
            learned_confidence = heuristic_confidence
            confidence = heuristic_confidence
            model_source = "classical_cv_rice_gate"

        min_heuristic = max(0.45, threshold * 0.7)
        min_learned = 0.55
        is_rice = (
            confidence >= threshold
            and heuristic_confidence >= min_heuristic
            and learned_confidence >= min_learned
        )

        if is_rice:
            class_name = mappings["rice_class_name"]
            class_id = mappings["rice_class_id"]
        else:
            # Foreign matter is NOT rice — keep the project's class vocabulary.
            class_name = classify_non_rice_object(features["mean_rgb"])
            class_id = mappings.get("foreign_matter", {}).get(class_name, -1)

        object_id += 1
        detections.append(
            {
                "object_id": object_id,
                "class_id": class_id,
                "class_name": class_name,
                "is_rice": is_rice,
                "confidence": round(confidence, 4),
                "confidence_threshold": threshold,
                "bbox": features["bbox"],
                "area_pixels": features["area_pixels"],
                "rice_likeness_cues": cues,
                "model_source": model_source,
                "features": features,
            }
        )

    return detections


def evaluate_rice_presence(image_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Decide whether the image contains RICE (not merely "objects").

    Returns the gate payload consumed by ml.pipeline. Backwards-compatible keys
    (`has_rice`, `confidence`, `estimated_grain_count`, `total_objects_detected`,
    `message`, `method`) are preserved for existing callers.
    """
    cfg = _gate_config()
    mappings = load_project_class_mappings()
    threshold = float(cfg["rice_confidence_threshold"])
    warnings: List[str] = []

    if not mappings.get("rice_class_name"):
        logger.error(
            "Detector class configuration has no rice class — rice gate cannot run "
            "(expected a rice class in models/segmentation/class_mapping.json)."
        )
        payload = _empty_gate_payload(
            mappings, threshold, cfg, STATUS_NO_RICE_CLASS, MESSAGE_NO_RICE_CLASS, warnings
        )
        payload["debug"] = (
            "Total detections: 0 | Rice detections: 0 | Foreign matter detections: 0 | "
            f"Rice confidence threshold: {threshold} | Rice gate: FAILED (no rice class configured)"
        )
        logger.error(payload["debug"])
        return payload

    detections = detect_rice_detections(image_rgb)

    status = STATUS_RICE if any(d["is_rice"] for d in detections) else STATUS_NOT_RICE
    message = MESSAGE_RICE if status == STATUS_RICE else MESSAGE_NOT_RICE
    if not detections:
        warnings.append(
            "No foreground object met the rice-gate minimum area of "
            f"{float(cfg['min_object_area_pixels']):g} pixels."
        )

    payload = _gate_payload(
        detections=detections,
        mappings=mappings,
        threshold=threshold,
        cfg=cfg,
        status=status,
        message=message,
        warnings=warnings,
    )
    payload["method"] = (
        "learned_rice_gate"
        if any(d.get("model_source") == "learned_rice_gate" for d in detections)
        else RICE_GATE_METHOD
    )
    payload["model_status"] = (
        "trained (learned rice-vs-non-rice gate active)"
        if any(d.get("model_source") == "learned_rice_gate" for d in detections)
        else _MODEL_STATUS
    )

    # Backwards-compatible summary keys (previously returned by
    # ml.segmentation.detect_rice_presence)
    payload["confidence"] = payload["rice_confidence_mean"] or payload["rice_confidence_max"]
    payload["estimated_grain_count"] = payload["rice_detections"]
    payload["total_objects_detected"] = payload["total_detections"]

    logger.info("Rice gate: %s. %s", status, payload["debug"])
    for detection in detections:
        logger.debug(
            "Rice gate object #%s -> %s (confidence %.4f, threshold %.2f, bbox %s)",
            detection["object_id"],
            detection["class_name"],
            detection["confidence"],
            threshold,
            detection["bbox"],
        )

    return payload
