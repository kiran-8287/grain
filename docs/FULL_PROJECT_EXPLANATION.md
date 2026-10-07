# GRAIN QUALITY ANALYZER — DETAILED PROJECT STRUCTURE EXPLANATION

## Project Root: `A:\grain\`

This document provides a comprehensive explanation of every major directory and file in the GRAIN QUALITY ANALYZER project.

---

## Top-Level Directory Map

```
A:\grain\
├── .git/                      # Git repository metadata
├── .kilo/                     # Kilo AI configuration
├── .pytest_cache/             # Python test cache
├── .venv/                     # Python virtual environment
├── .vscode/                   # VS Code workspace settings
├── __pycache__/               # Python bytecode cache
├── 27 tests/                  # Test artifacts directory
├── ai_work_log/               # AI progress and work logs
├── assets/                    # Static assets (logo, etc.)
├── backend/                   # FastAPI backend application
├── configs/                   # Configuration JSON files
├── current_objective.txt      # Current task/objective tracking
├── data/                      # Runtime data and audit files
├── DATASET_AUDIT_REPORT.md    # Dataset audit documentation
├── datasets/                  # Dataset storage (8 datasets)
├── docs/                      # Documentation files
├── frontend/                  # React + Vite frontend application
├── grain_profiles/            # Grain profile data
├── ml/                        # Machine learning modules
├── MODEL_EXPLANATION.md       # Model documentation
├── models/                    # Trained model artifacts
├── node_modules/              # Frontend dependencies
├── package.json               # Root package.json (concurrently)
├── package-lock.json          # Root lock file
├── progress/                  # Progress tracking
├── README.md                  # Project readme
├── render.yaml                # Render deployment config
├── runs/                      # Run history/logs
├── scripts/                   # Utility and audit scripts
├── standards/                 # Standards data files (JSON)
├── test_image_results/        # Test image output results
├── testing_images/            # Test input images
├── tests/                     # Backend pytest tests
└── AGENTS.md                  # Agent working instructions
```

---

## 1. BACKEND (`A:\grain\backend\`)

The backend is a **FastAPI** application written in **Python** that handles all image processing, ML inference, and API serving.

### Structure:

```
backend/
├── .env                        # Environment variables (secrets, API keys)
├── .env.example                # Environment template
├── .pytest_cache/              # Test cache
├── package-lock.json           # Lock file
├── requirements.txt            # Python dependencies
└── app/
    ├── main.py                 # FastAPI application entry point
    ├── api/
    │   └── routes.py           # API endpoint definitions
    ├── schemas/
    │   └── models.py           # Pydantic request/response models
    └── services/
        ├── export.py           # JSON/CSV export functionality
        ├── job_manager.py      # Job lifecycle management
        └── run_logger.py       # Run/audit logging
```

### Key Files:

| File | Purpose |
|------|---------|
| `backend/app/main.py` | Creates FastAPI app, configures CORS, mounts frontend, includes API routes |
| `backend/app/api/routes.py` | Defines all REST endpoints (`/api/analyze`, `/health`, `/models`, etc.) |
| `backend/app/schemas/models.py` | Pydantic models for request/response serialization |
| `backend/app/services/job_manager.py` | Manages analysis job state and lifecycle |
| `backend/app/services/export.py` | Handles CSV/JSON export of results |
| `backend/app/services/run_logger.py` | Logs analysis runs for audit purposes |

### Backend Entry Point:

The `main.py` file:
- Adds project root to `sys.path` so `ml` and `backend` modules can be imported
- Creates the FastAPI app with title "GRAIN QUALITY ANALYZER API"
- Configures CORS middleware using `FRONTEND_ORIGIN` environment variable
- Includes API routes at `/api` prefix
- Mounts frontend `dist/` directory if it exists (for production serving)
- Provides a root `/` endpoint with API info when frontend is not built

---

## 2. ML MODULES (`A:\grain\ml\`)

The ML directory contains all machine learning and computer vision logic, organized into three sub-packages.

```
ml/
├── __init__.py                 # Package init
├── __pycache__/                # Python bytecode cache
├── config.py                   # ML-specific configuration
├── quality/                    # Quality classification modules
├── segmentation/               # Segmentation pipeline modules
└── standards/                  # Standards comparison modules
```

### 2.1 Segmentation (`ml/segmentation/`)

```
ml/segmentation/
├── __pycache__/
├── inference.py                # Model inference logic
├── pipeline.py                 # End-to-end segmentation pipeline
├── postprocessing.py           # Post-segmentation cleanup
├── preprocessing.py            # Image preprocessing
├── rice_gate.py                # Rice/non-rice detection gate
└── segmentation.py             # Core segmentation algorithms
```

| File | Purpose |
|------|---------|
| `pipeline.py` | Orchestrates the full segmentation workflow |
| `segmentation.py` | Core Classical CV segmentation (contours, morphology, watershed) |
| `rice_gate.py` | Determines if image contains rice or foreign matter |
| `preprocessing.py` | Image loading, resizing, normalization |
| `postprocessing.py` | Mask refinement, noise removal, instance separation |
| `inference.py` | Model loading and inference execution |

### 2.2 Quality (`ml/quality/`)

```
ml/quality/
├── __pycache__/
├── __init__.py
├── admixture.py                # Admixture analysis
├── calibration.py              # Physical calibration (pixels to mm)
├── classifiers.py              # Quality classifiers (broken, damaged, etc.)
├── colour.py                   # Colour-based defect detection
├── foreign_matter.py           # Foreign matter detection
├── geometry.py                  # Per-grain geometry calculations
├── profiles.py                  # Grain profile definitions
├── quality.py                   # Quality analysis orchestration
└── texture.py                   # Texture-based defect detection
```

| File | Purpose |
|------|---------|
| `geometry.py` | Calculates length, breadth, area, L/B ratio, major/minor axis |
| `classifiers.py` | Implements defect classifiers (broken, damaged, etc.) |
| `quality.py` | Orchestrates quality analysis pipeline |
| `calibration.py` | Converts pixel measurements to physical units |
| `colour.py` | Colour space analysis for chalky/red/dehusked detection |
| `texture.py` | Texture features for damaged/immature detection |
| `foreign_matter.py` | Foreign matter detection heuristics |
| `admixture.py` | Admixture/outlier detection |

### 2.3 Standards (`ml/standards/`)

```
ml/standards/
└── standards.py                # Standards comparison and grading
```

Compares computed quality metrics against reference standards.

---

## 3. FRONTEND (`A:\grain\frontend\`)

The frontend is a **React + TypeScript** application built with **Vite** and styled with **Tailwind CSS**.

### Structure:

```
frontend/
├── .env.example                # Environment template
├── dist/                       # Production build output
├── index.html                  # HTML entry point
├── node_modules/               # Dependencies
├── package.json                # Dependencies and scripts
├── package-lock.json           # Lock file
├── postcss.config.js           # PostCSS configuration
├── public/                     # Static assets
├── src/
│   ├── App.tsx                 # Main application component
│   ├── components/             # Reusable UI components
│   ├── index.css               # Global styles
│   ├── main.tsx                # React entry point
│   └── types.ts                # TypeScript type definitions
├── tailwind.config.js          # Tailwind CSS configuration
├── tsconfig.json               # TypeScript configuration
└── vite.config.ts              # Vite build configuration
```

### Key Frontend Components (`frontend/src/components/`):

| Component | Purpose |
|-----------|---------|
| `App.tsx` | Main app component, manages state, handles analysis flow |
| `UploadSection.tsx` | Image upload interface |
| `ProcessingState.tsx` | Loading spinner during analysis |
| `GlobalMetrics.tsx` | Displays total grains, whole/broken counts, percentages |
| `AnnotatedViewer.tsx` | Canvas displaying annotated image with masks/boxes/IDs |
| `GrainTable.tsx` | Table of all grains with geometry data |
| `GrainDetailPanel.tsx` | Selected grain detailed view |
| `ExportControls.tsx` | Download JSON/CSV buttons |
| `Navbar.tsx` | Top navigation bar |
| `Phase1Demo.tsx` | Phase 1 demo component |
| `QualityWarningsPanel.tsx` | Warning display for unvalidated results |
| `StandardsScreening.tsx` | Standards comparison display |
| `ProcessingState.tsx` | Processing indicator |

### Main App Flow (`App.tsx`):

1. **Check backend health** on mount (polling every 10s)
2. **Handle image upload** via `handleAnalyze()`
3. **POST to `/api/analyze`** with FormData
4. **Display non-rice message** if `rice_detected` is false
5. **Display results dashboard** with:
   - `GlobalMetrics` - summary statistics
   - `AnnotatedViewer` - annotated image canvas
   - `GrainTable` - grain list
   - `GrainDetailPanel` - selected grain details
   - `ExportControls` - download buttons
6. **Manage display preferences** (show/hide masks, boxes, IDs) via localStorage

---

## 4. CONFIGS (`A:\grain\configs\`)

JSON configuration files for the application:

| File | Purpose |
|------|---------|
| `app.json` | General application settings |
| `inference.json` | Inference-time settings (thresholds, batch size, etc.) |
| `models.json` | Model paths and mapping configuration |
| `parameters.json` | Analysis parameters |
| `thresholds.json` | Classification thresholds (e.g., broken grain threshold) |

---

## 5. DATASETS (`A:\grain\datasets\`)

Stores all training and validation datasets:

| Directory | Purpose |
|-----------|---------|
| `01_Mendeley_Rice_Variety/` | Mixed variety dataset (~13,406 images) |
| `02_Grainalyze/` | Defect/segmentation dataset (~2,470 images) |
| `03_Rice_Variety/` | Variety segmentation dataset (~8,017 images) |
| `04_RiceVigor/` | Single-grain segmentation (~1,200 images) |
| `05_Ricee/` | Mixed quality dataset (~3,484 images) |
| `06_Rice_Grain_Segmentation/` | Single-class segmentation (~104 images, used for YOLO training) |
| `07_Raw_Rice_Seed/` | Dense/quality dataset (~81 images) |
| `08_Rice_Grain/` | Rice/broken dataset (~63 images) |

---

## 6. MODELS (`A:\grain\models\`)

Trained model artifacts:

```
models/
├── foreign_matter/             # Foreign matter detection models
└── segmentation/
    └── experimental/
        └── yolo11n_seg_baseline/
            └── weights/
                ├── best.pt    # Best YOLO11n-seg checkpoint (rice-trained)
                └── last.pt    # Last YOLO11n-seg checkpoint
```

Also exists: `models/segmentation/experimental/yolo11n-seg-pretrained.pt` (generic COCO-pretrained, not rice-specific).

---

## 7. TESTS (`A:\grain\tests\`)

Pytest test files:

| File | Purpose |
|------|---------|
| `test_api.py` | API endpoint tests |
| `test_rice_gate.py` | Rice/no-rice gate tests |

---

## 8. STANDARDS (`A:\grain\standards\`)

Reference standards data files, including:

```
standards/
└── india_kms_2026_27_raw_rice.json  # KMS 2026-27 reference limits (marked as unverified)
```

---

## 9. DATA (`A:\grain\data\`)

Runtime data:

```
data/
└── runs/                       # Analysis run outputs
```

---

## 10. SCRIPTS (`A:\grain\scripts\`)

Utility scripts for dataset preparation, auditing, and training.

---

## 11. DOCUMENTATION (`A:\grain\docs\`)

Project documentation files including this file.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React, TypeScript, Vite, Tailwind CSS, Lucide icons |
| **Backend** | FastAPI, Python, Pydantic, Uvicorn |
| **ML/CV** | OpenCV, NumPy, scikit-image, Ultralytics YOLO (experimental) |
| **Deployment** | Render (render.yaml) |
| **Testing** | Pytest |
| **Package Management** | npm (root), pip (backend) |

---

## Data Flow

```
User uploads image via React frontend
        ↓
Frontend sends POST /api/analyze with FormData
        ↓
FastAPI receives request
        ↓
job_manager initializes job
        ↓
RiceQualityPipeline.analyze()
        ↓
┌─────────────────────────────────┐
│ 1. Rice Gate (rice_gate.py)     │ ← Rice/No-Rice detection
│ 2. Preprocessing               │
│ 3. Segmentation (pipeline.py)   │ ← Classical CV or YOLO
│ 4. Postprocessing              │
│ 5. Geometry (geometry.py)       │ ← Per-grain measurements
│ 6. Quality (quality.py)         │ ← Defect classification
│ 7. Standards (standards.py)     │ ← Standards comparison
└─────────────────────────────────┘
        ↓
API response with AnalysisResult
        ↓
Frontend displays results dashboard
```

---

## Current Phase

**Phase 1 — Rice Grain Segmentation & Geometry**

The system currently performs:
- Rice/non-rice detection
- Instance segmentation of individual grains
- Per-grain geometric measurements (length, breadth, L/B ratio)
- Total grain count

Future phases will add:
- Defect classification (broken, damaged, chalky, red, etc.)
- Foreign matter detection
- Admixture analysis
- Standards screening and official grading
