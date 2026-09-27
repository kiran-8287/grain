# CURRENT STATUS — GRAIN QUALITY ANALYZER

> **Generated:** 2026-09-28
> **Scope:** Phase 1 audit. Grading and 14-parameter analysis are out of scope.

---

## 1. What Currently Works

| Component | Status | Notes |
|-----------|--------|-------|
| **FastAPI Backend** | ✅ Running | `backend/app/main.py`, uvicorn |
| **React Frontend** | ✅ Running | Vite + Tailwind, port 5173 |
| **Classical CV Segmentation** | ✅ Working | Watershed + distance transform |
| **Background Detection** | ✅ Working | Border-pixel sampling + Otsu |
| **Grain Splitting** | ⚠️ Partial | Fails on tight clusters, overlaps |
| **Unique Grain IDs** | ✅ Working | Sequential assignment |
| **Annotated Overlay** | ✅ Working | Coloured masks + grain IDs |
| **Rice Gate (Heuristic)** | ✅ Working | Heuristic, not trained model |
| **Foreign Matter (Heuristic)** | ✅ Working | Colour/shape outlier heuristics |
| **API: /api/analyze** | ✅ Working | Image upload + analysis |

## 2. What Does NOT Work

| Component | Status | Reason |
|-----------|--------|--------|
| **Mask R-CNN Inference** | ❌ Missing | No trained weights |
| **YOLOv8 Segmentation** | ❌ Missing | Not installed, no training script |
| **Touching grain separation** | ❌ Unreliable | Watershed merges tight clusters |
| **Overlapping grain separation** | ❌ Unreliable | Distance transform cannot recover |
| **Dense scene (100+ grains)** | ❌ Unreliable | Merges large grain portions |
| **Ground-truth evaluation** | ❌ Missing | No measured mAP, no count error |
| **`train_segmentation.py`** | ❌ Stub | Only writes JSON manifests |

## 3. Model Weights Status

| Model | Path | Status |
|-------|------|--------|
| Mask R-CNN | `models/segmentation/` | JSON configs only — NO WEIGHTS |
| YOLOv8l-seg | `models/yolo_seg/` | Does not exist yet |

## 4. Dataset Status

| Dataset | Format | Purpose |
|---------|--------|---------|
| GrainSet Rice v3 | Images + PNG masks | Instance segmentation (mask type TBD) |
| GrainDet Rice v2 | 8-class classification | Synthetic data generation |

**Critical:** GrainSet v3 mask type unknown — run `scripts/audit_dataset.py` first.

## 5. Phase 1 Scope

Phase 1 implements: Detection + Instance Segmentation + Foreign Matter.
Phase 1 does NOT implement: grading, 14-parameter analysis, defect classification.
