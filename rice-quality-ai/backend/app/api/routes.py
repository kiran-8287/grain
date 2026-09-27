"""
API Routes for Rice Quality AI.
"""

import base64
import io
import json
import logging
from typing import Any, Optional, List, Dict

import numpy as np
from fastapi import APIRouter, Body, File, Form, HTTPException, Path, Response, UploadFile
from fastapi.responses import PlainTextResponse

import cv2


def _sanitize_numpy(obj: Any) -> Any:
    """
    Recursively convert numpy scalars / arrays to native Python types so that
    FastAPI's default JSON serializer never encounters a numpy.int32 /
    numpy.float64 / numpy.bool_ and raises a 500.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_numpy(v) for v in obj)
    return obj

from backend.app.schemas.models import (
    AnalysisResultResponse,
    HealthResponse,
    JobStatusResponse,
    ModelsResponse,
    StandardsResponse,
)
from backend.app.services.export import export_result_csv, export_result_json
from backend.app.services.job_manager import job_manager
from ml.config import load_model_registry, load_standards, get_project_root

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint."""
    import torch
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="Rice Quality AI Backend",
        gpu_available=gpu_available,
        device=device,
    )


@router.get("/models")
def get_models():
    """Return model registry status and metadata."""
    registry = load_model_registry()
    return {"models": registry.get("models", {})}


@router.get("/standards")
def get_standards():
    """Return current India KMS 2026-27 standards definition."""
    try:
        standards_data = load_standards()
        return standards_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Standards file not found.")


@router.post("/analyze")
async def analyze_rice_image(
    file: UploadFile = File(...),
    grade: str = Form("grade_a"),
    reference_pixels: Optional[float] = Form(None),
    reference_mm: Optional[float] = Form(None),
):
    """
    Analyze uploaded rice grain image.
    Supports JPG, PNG, WEBP, TIFF.
    """
    # Read file bytes
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    manual_scale = None
    if reference_pixels is not None and reference_mm is not None and reference_pixels > 0:
        manual_scale = {
            "reference_pixels": reference_pixels,
            "reference_mm": reference_mm,
        }

    # Execute analysis synchronously (or via job manager)
    result = job_manager.process_sync(
        image_bytes=contents,
        filename=file.filename or "uploaded.jpg",
        manual_scale=manual_scale,
        grade=grade,
    )

    if not result.get("success", False) and "error" in result:
        # If it was an image validation/decoding error
        raise HTTPException(status_code=400, detail=result.get("error"))

    # Sanitize all numpy scalars/arrays → native Python types before FastAPI
    # serializes to JSON.  numpy.int32 bboxes and numpy.float64 colour scores
    # cause a 500 "Object of type int32 is not JSON serializable" otherwise.
    return _sanitize_numpy(result)


@router.get("/analysis/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Check the status of an analysis job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress_stage=job["progress_stage"],
        progress_percent=job["progress_percent"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
        error=job["error"],
    )


@router.get("/analysis/{job_id}/result")
def get_job_result(job_id: str):
    """Retrieve full analysis result for a completed job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet (current status: {job['status']}).",
        )
    return job["result"]


@router.get("/analysis/{job_id}/image")
def get_annotated_image(job_id: str):
    """Retrieve the annotated JPEG image binary."""
    job = job_manager.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job result not found.")
    
    b64_data = job["result"].get("annotated_image_base64")
    if not b64_data or not b64_data.startswith("data:image/jpeg;base64,"):
        raise HTTPException(status_code=404, detail="Annotated image not available.")
    
    raw_b64 = b64_data.split(",", 1)[1]
    image_bytes = base64.b64decode(raw_b64)
    return Response(content=image_bytes, media_type="image/jpeg")


@router.get("/analysis/{job_id}/export/json")
def export_job_json(job_id: str):
    """Download analysis result as a JSON file."""
    job = job_manager.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job result not found.")
    
    json_str = export_result_json(job["result"])
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=rice_analysis_{job_id[:8]}.json"},
    )


@router.get("/analysis/{job_id}/export/csv")
def export_job_csv(job_id: str):
    """Download analysis result as a CSV file."""
    job = job_manager.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job result not found.")
    
    csv_str = export_result_csv(job["result"])
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rice_analysis_{job_id[:8]}.csv"},
    )


# =============================================================================
# PHASE 1 ENDPOINTS
# =============================================================================


def _decode_image_bytes(contents: bytes) -> np.ndarray:
    """
    Decode raw image bytes into a uint8 HxWxC RGB numpy array.
    Raises HTTPException(400) on failure.
    """
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        arr = np.frombuffer(contents, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file as bytes: {exc}",
        )

    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        # Try with unchanged flag for alpha/grayscale
        img_any = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img_any is None or img_any.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file: could not decode. Supported formats: JPG, PNG, WEBP, TIFF, BMP.",
            )
        if len(img_any.shape) == 2:
            img_bgr = cv2.cvtColor(img_any, cv2.COLOR_GRAY2BGR)
        elif img_any.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_any, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = img_any

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def _encode_image_base64(img_rgb: np.ndarray, fmt: str = ".png") -> str:
    """Encode an RGB(HxWxC) image to a data URL base64 string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(fmt, img_bgr)
    if not ok:
        return ""
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    raw = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:{mime};base64,{raw}"


def _build_phase1_overlay(
    img_rgb: np.ndarray,
    analysis: Dict[str, Any],
) -> np.ndarray:
    """
    Thin wrapper: delegated to the shared renderer in ``ml.postprocessing`` so
    the Phase 1 demo endpoint and the full dashboard annotated image produce
    pixel-identical mask colours, ID labels, FM boxes, and legend.

    Kept here (and not deleted) to avoid breaking existing callers that import
    it privately from this module; its body is now one line plus method-badge
    label assembly.
    """
    from ml.postprocessing import render_phase1_overlay

    method = analysis.get("method") or "seg"
    version = analysis.get("model_version") or ""
    legend_label = f"{method}  v{version}" if version else method
    return render_phase1_overlay(
        img_rgb=img_rgb,
        grains=analysis.get("grains", []) or [],
        foreign_matter=analysis.get("foreign_matter", []) or [],
        include_legend=True,
        legend_method_label=legend_label,
    )



@router.post("/phase1/analyze")
async def phase1_analyze(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = Form(0.25),
    method: Optional[str] = Form("auto"),
):
    """
    Phase 1 image analysis endpoint.

    Accepts an image upload and runs the instance segmentation analysis
    via `ml.inference.analyze_image`, returning the structured analysis
    along with an overlay image (coloured grain masks + grain IDs +
    foreign-matter boxes) encoded as base64.

    Form fields:
      - file (required): image file (JPG, PNG, WEBP, TIFF, BMP)
      - conf_threshold (optional float): confidence cutoff (default 0.25)
      - method (optional string): "auto" | "yolov8" | "maskrcnn" | "classical"
                                  (default "auto")
    """
    MAX_SIZE_BYTES = 64 * 1024 * 1024  # 64 MiB

    # --- 1) Read and validate file size ---
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}",
        )

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file too large ({len(contents)} bytes). Max size: {MAX_SIZE_BYTES} bytes.",
        )

    # Validate method parameter
    valid_methods = {"auto", "yolov8", "maskrcnn", "classical"}
    if method is not None and method.lower() not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method '{method}'. Must be one of: {sorted(valid_methods)}",
        )

    # Validate confidence threshold
    if conf_threshold is not None:
        if conf_threshold < 0.0 or conf_threshold > 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid conf_threshold={conf_threshold}. Must be between 0.0 and 1.0.",
            )

    # --- 2) Decode image bytes → RGB numpy array ---
    try:
        img_rgb = _decode_image_bytes(contents)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {exc}",
        )

    if img_rgb.size == 0 or img_rgb.shape[0] < 2 or img_rgb.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Decoded image is too small (shape={img_rgb.shape}).",
        )

    # --- 3) Run analysis via ml.inference.analyze_image ---
    try:
        from ml.inference import analyze_image as ml_analyze_image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import ml.inference.analyze_image: {exc}",
        )

    try:
        # The method form value is mapped for reference; analyze_image()
        # itself follows its own priority order via _load_model().
        _ = method  # reserved for future use (per-request override)
        analysis_raw = ml_analyze_image(img_rgb)
    except Exception as exc:
        logger.exception(f"Phase1 analyze_image() failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Model analysis failed: {exc}",
        )

    if not isinstance(analysis_raw, dict):
        raise HTTPException(
            status_code=500,
            detail="Model analysis returned an invalid result (expected dict).",
        )

    analysis = _sanitize_numpy(analysis_raw)

    # --- 4) Encode original image as base64 (optional, for convenience) ---
    original_base64 = ""
    try:
        original_base64 = _encode_image_base64(img_rgb, fmt=".jpg")
    except Exception:
        original_base64 = ""

    # --- 5) Generate overlay image and encode as base64 ---
    overlay_base64 = ""
    try:
        overlay_rgb = _build_phase1_overlay(img_rgb, analysis)
        overlay_base64 = _encode_image_base64(overlay_rgb, fmt=".png")
    except Exception as exc:
        logger.warning(f"Failed to build phase1 overlay image: {exc}")
        overlay_base64 = ""

    # --- 6) Assemble the Phase1AnalysisResponse-compatible payload ---
    response_payload = {
        "success": True,
        "rice_detected": bool(analysis.get("rice_detected", False)),
        "rice_count": int(analysis.get("rice_count", 0)),
        "foreign_matter_count": int(analysis.get("foreign_matter_count", 0)),
        "unresolved_cluster_count": int(analysis.get("unresolved_cluster_count", 0)),
        "grains": analysis.get("grains", []),
        "foreign_matter": analysis.get("foreign_matter", []),
        "unresolved_clusters": analysis.get("unresolved_clusters", []),
        "processing": analysis.get("processing", {}),
        "method": analysis.get("method", "unknown"),
        "model_version": analysis.get("model_version", "1.0.0"),
        "parameters": {
            "conf_threshold": float(conf_threshold) if conf_threshold is not None else 0.25,
            "method": method if method is not None else "auto",
            "original_filename": file.filename or "",
            "image_shape": [int(img_rgb.shape[0]), int(img_rgb.shape[1])],
        },
        "original_image_base64": original_base64,
        "overlay_base64": overlay_base64,
        "warnings": analysis.get("warnings", []),
    }

    return _sanitize_numpy(response_payload)


@router.get("/phase1/models")
def phase1_models():
    """
    Phase 1 model availability endpoint.

    Returns a JSON payload describing which segmentation models are
    currently available (installed + weights present on disk), which
    method auto-priority will pick, paths to the model weights, and
    CPU/GPU availability info.
    """
    project_root = get_project_root()

    # --- Device / GPU availability ---
    device = "cpu"
    gpu_available = False
    try:
        import torch
        gpu_available = bool(torch.cuda.is_available())
        device = "cuda" if gpu_available else "cpu"
    except Exception:
        device = "cpu"
        gpu_available = False

    # --- YOLOv8 ---
    yolo_weights_rel = "models/yolo_seg/run1_baseline/weights/best.pt"
    yolo_weights_path = project_root / yolo_weights_rel
    yolov8_weights_exist = yolo_weights_path.exists()
    yolov8_installed = False
    try:
        from ultralytics import YOLO  # noqa: F401
        yolov8_installed = True
    except ImportError:
        yolov8_installed = False
    except Exception:
        yolov8_installed = False

    yolov8_available = yolov8_weights_exist and yolov8_installed

    # --- Mask R-CNN ---
    maskrcnn_weights_rel = ""
    maskrcnn_weights_exist = False
    try:
        from ml.config import get_model_info
        seg_info = get_model_info("segmentation") or {}
        weights_rel = seg_info.get("weights_path", "")
        if weights_rel:
            maskrcnn_weights_rel = weights_rel
            p = project_root / weights_rel
            maskrcnn_weights_exist = p.exists()
    except Exception:
        maskrcnn_weights_exist = False
        maskrcnn_weights_rel = ""

    maskrcnn_installed = False
    try:
        import torch  # noqa: F401
        from torchvision.models.detection import maskrcnn_resnet50_fpn  # noqa: F401
        maskrcnn_installed = True
    except ImportError:
        maskrcnn_installed = False
    except Exception:
        maskrcnn_installed = False

    maskrcnn_available = maskrcnn_weights_exist and maskrcnn_installed

    # --- Classical CV (always available) ---
    classical_available = True

    # --- Determine best method in auto priority order ---
    if yolov8_available:
        best_method = "yolov8"
    elif maskrcnn_available:
        best_method = "maskrcnn"
    else:
        best_method = "classical"

    model_paths = {
        "yolov8": str(yolo_weights_path) if yolov8_weights_exist else None,
        "yolov8_relative": yolo_weights_rel,
        "maskrcnn": str(project_root / maskrcnn_weights_rel) if maskrcnn_weights_rel and maskrcnn_weights_exist else None,
        "maskrcnn_relative": maskrcnn_weights_rel or None,
        "classical": None,
    }

    payload = {
        "available": {
            "yolov8": bool(yolov8_available),
            "maskrcnn": bool(maskrcnn_available),
            "classical": True,
        },
        "best_method": best_method,
        "model_paths": model_paths,
        "device": device,
        "gpu_available": bool(gpu_available),
        "yolov8_installed": bool(yolov8_installed),
        "yolov8_weights_exist": bool(yolov8_weights_exist),
        "maskrcnn_installed": bool(maskrcnn_installed),
        "maskrcnn_weights_exist": bool(maskrcnn_weights_exist),
    }

    return _sanitize_numpy(payload)


@router.post("/phase1/grain_crop/{grain_id}")
async def phase1_grain_crop(
    grain_id: int = Path(..., ge=1, description="Grain ID (positive integer)"),
    body: Dict[str, Any] = Body(..., embed=False),
):
    """
    Phase 1 grain crop endpoint.

    Extract and return a PNG image crop for a single grain, given the
    source image (base64), the grain bounding box, and optionally the
    mask polygon.  Outside the mask polygon, pixels are set to
    transparent (alpha=0) so the output is a PNG with transparency.

    Path:
      - grain_id: positive integer identifier for the grain (used for
                  logging / response filename only)

    JSON body keys:
      - image_base64 (string, required): base64 data URL or raw base64
                                         of the source image (any format
                                         that OpenCV can decode).
      - grain_bbox (list[4 int], required): [x, y, w, h] bounding box
                                            in the source image coords.
      - mask_polygon (list[list[2 float]], optional): polygon points
                                              [[x,y], ...] inside the
                                              image; anything outside
                                              this polygon becomes
                                              transparent.  If omitted,
                                              the full bbox is returned
                                              without masking.
      - padding (int, optional, default=10): number of pixels of extra
                                             context to include around
                                             the bbox (clamped to image
                                             borders).

    Returns:
      PNG image as `image/png` response (binary).  The response has a
      `Content-Disposition` header suggesting the filename
      `grain_{grain_id}.png`.
    """
    # --- 1) Validate and read body fields ---
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    image_base64_raw = body.get("image_base64")
    grain_bbox = body.get("grain_bbox")
    mask_polygon = body.get("mask_polygon")
    padding = body.get("padding", 10)

    if image_base64_raw is None or not isinstance(image_base64_raw, str) or not image_base64_raw.strip():
        raise HTTPException(status_code=400, detail="Missing or empty required field 'image_base64'.")

    if grain_bbox is None or not isinstance(grain_bbox, (list, tuple)) or len(grain_bbox) != 4:
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid required field 'grain_bbox'. Must be [x, y, w, h].",
        )

    try:
        bx, by, bw, bh = [int(float(v)) for v in grain_bbox]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid 'grain_bbox' values. Must be numeric [x, y, w, h].",
        )

    if bw <= 0 or bh <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid 'grain_bbox' dimensions: w={bw}, h={bh}. Must be > 0.",
        )

    try:
        padding = int(padding) if padding is not None else 10
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'padding': must be an integer.")
    if padding < 0:
        raise HTTPException(status_code=400, detail="Invalid 'padding': must be >= 0.")

    # --- 2) Decode base64 image to bytes, then to RGB numpy array ---
    b64_str = image_base64_raw.strip()
    if "," in b64_str and b64_str.startswith("data:"):
        # Strip data URL prefix
        b64_str = b64_str.split(",", 1)[1]
    if not b64_str:
        raise HTTPException(status_code=400, detail="Invalid 'image_base64': empty data.")

    try:
        image_bytes = base64.b64decode(b64_str, validate=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 encoding in 'image_base64': {exc}",
        )

    try:
        img_rgb = _decode_image_bytes(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image from 'image_base64': {exc}",
        )

    H, W = img_rgb.shape[:2]
    if H < 2 or W < 2:
        raise HTTPException(status_code=400, detail=f"Image too small: shape=({H}, {W}).")

    # --- 3) Compute padded crop region (clamped to image borders) ---
    x1 = max(0, bx - padding)
    y1 = max(0, by - padding)
    x2 = min(W, bx + bw + padding)
    y2 = min(H, by + bh + padding)

    if x2 <= x1 or y2 <= y1:
        raise HTTPException(
            status_code=400,
            detail=f"Computed crop region is empty: x1={x1},x2={x2},y1={y1},y2={y2} (img W={W}, H={H}).",
        )

    # --- 4) Extract the crop from RGB and create BGRA (with alpha) ---
    crop_rgb = img_rgb[y1:y2, x1:x2]
    ch, cw = crop_rgb.shape[:2]
    crop_bgra = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGRA)
    # Alpha channel defaults to fully opaque
    crop_bgra[:, :, 3] = 255

    # --- 5) Apply mask polygon if provided (outside → alpha=0) ---
    if mask_polygon is not None:
        if not isinstance(mask_polygon, list) or len(mask_polygon) < 3:
            raise HTTPException(
                status_code=400,
                detail="Invalid 'mask_polygon': must be a list of at least 3 [x,y] points.",
            )

        try:
            global_pts: List[List[float]] = []
            for pt in mask_polygon:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    raise ValueError(f"polygon point must be [x,y], got {pt!r}")
                px, py = float(pt[0]), float(pt[1])
                global_pts.append([px, py])
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid 'mask_polygon' point format: {exc}",
            )

        try:
            # Convert to crop-local integer coordinates
            local_arr = np.array(
                [[int(round(px - x1)), int(round(py - y1))] for [px, py] in global_pts],
                dtype=np.int32,
            ).reshape(-1, 1, 2)

            mask_canvas = np.zeros((ch, cw), dtype=np.uint8)
            cv2.fillPoly(mask_canvas, [local_arr], 255)

            # Set alpha to 0 where mask is 0
            crop_bgra[:, :, 3] = mask_canvas
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(f"Failed to apply mask_polygon for grain #{grain_id}: {exc}")
            # Leave fully opaque as a sensible fallback (do not fail the whole crop)

    # --- 6) Encode BGRA → PNG bytes ---
    try:
        ok, png_buf = cv2.imencode(".png", crop_bgra)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode PNG: {exc}",
        )

    if not ok or png_buf.size == 0:
        raise HTTPException(status_code=500, detail="Failed to encode PNG (imencode returned empty).")

    png_bytes = png_buf.tobytes()

    filename = f"grain_{int(grain_id)}.png"
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Grain-Id": str(int(grain_id)),
            "X-Crop-Rect": f"x={x1},y={y1},w={cw},h={ch}",
        },
    )
