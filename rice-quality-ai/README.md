# Rice Quality AI — Automated Raw Milled Rice Quality Assessment System

An image-analysis project for raw milled rice. It reports per-grain image predictions and measurements, observed image fractions, and historical/reference screening limits as separate outputs. The current KMS 2026-27 specification has not been verified in this repository; this software does not determine official lot compliance.

---

## 🌱 Phase 1 — Instance Segmentation (Current Sprint)

This repository is currently in **Phase 1**, focused exclusively on building and training a production-quality rice instance-segmentation system. The 14-parameter grading, defect classification, and quality analysis described below is the project's long-term target and is explicitly **out of scope for Phase 1**.

### Phase 1 Deliverables (What is being implemented NOW)

| Component | Status | Description |
|-----------|--------|-------------|
| Dataset Audit Script | ✅ Implemented | `scripts/audit_dataset.py` — verifies mask type, counts instances, finds duplicates |
| COCO Converter | ✅ Implemented | `scripts/convert_to_coco.py` — semantic or instance masks → COCO JSON |
| Synthetic Data Generator | ✅ Implemented | `scripts/generate_synthetic.py` — touching/overlapping/dense composites |
| Train/Val/Test Splits | ✅ Implemented | `scripts/create_splits.py` — stratified 70/15/15 split |
| YOLOv8l-seg Training | ✅ Pipeline Ready | `training/train_yolo_seg.py` — real training (not stub), 100 epochs, AdamW, copy-paste aug |
| Mask R-CNN Baseline | ✅ Pipeline Ready | `training/train_maskrcnn.py` — comparison baseline, 50 epochs, SGD |
| Evaluation Framework | ✅ Implemented | `scripts/evaluate_model.py` — 9 categories, mAP/AP, count error, failure classification |
| Post-Processing Pipeline | ✅ Implemented | `ml/postprocessing.py` — conf filter, mask IoU NMS, tiling, global IDs, touching detect |
| Clean Inference API | ✅ Implemented | `ml/inference.py` — `analyze_image()` public function, structured output |
| Grain Crop Extraction | ✅ Implemented | `scripts/extract_grain_crops.py` — RGBA per-grain crops + masks + overlay |
| Failure Analysis Report | ✅ Implemented | `scripts/analyze_failures.py` — auto-generates docs/FAILURE_ANALYSIS.md |
| YOLOv8 Inference Path | ✅ Integrated | `ml/segmentation.py` — priority auto, falls back gracefully |
| Phase 1 API Endpoints | ✅ Added | `/phase1/analyze`, `/phase1/models`, `/phase1/grain_crop/{id}` |
| Frontend Phase 1 Demo | ✅ Added | Tabbed interface: upload → analyze → overlay viewer → per-grain details |
| Model Training | ⏳ PENDING Data | Requires datasets downloaded + audited (see docs/DATASET.md) |
| Model Evaluation | ⏳ PENDING Training | Requires trained checkpoint + test annotations |

### Phase 1 Quick Start

```bash
# 1. Install dependencies
cd rice-quality-ai
pip install -r backend/requirements.txt

# 2. Audit datasets (CRITICAL FIRST STEP before training)
python scripts/audit_dataset.py --dataset all --preview-count 20
# → Check datasets/inspection/audit_report.json for mask type

# 3. Convert to COCO format
python scripts/convert_to_coco.py --mask-type auto

# 4. Generate synthetic touching/overlapping/dense examples
python scripts/generate_synthetic.py

# 5. Create 70/15/15 stratified splits
python scripts/create_splits.py

# 6. Train YOLOv8l-seg (main model)
python training/train_yolo_seg.py --epochs 100 --name run1_baseline

# 7. Evaluate trained model
python scripts/evaluate_model.py --model-type yolo --generate-failure-images

# 8. Analyze failures → docs/FAILURE_ANALYSIS.md
python scripts/analyze_failures.py --eval-dir results/evaluation --append-docs

# 9. Launch backend + Phase 1 frontend
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
# → http://localhost:8000, click "Phase 1 Demo" tab
```

### Architecture Decision

**Primary model: YOLOv8l-seg**

Selected for:
- Single-package install (`ultralytics`)
- Anchor-free detector → better small/dense objects
- Native copy-paste augmentation → critical for touching grain separation
- Real-time inference (30-80ms CPU) → responsive frontend
- Single `.pt` file deployment
- Largest literature ecosystem for rice grain domain

Comparison baseline: **Mask R-CNN ResNet50-FPN-v2** (validation only, not production)
Emergency fallback: **Classical CV Watershed** (existing code, never deleted)

### Phase 1 In Scope / Out of Scope

✅ **IN SCOPE (now):**
- Rice detection (presence/absence)
- Individual rice grain **instance** segmentation (not semantic)
- Correctly separate touching grains
- Attempt to separate overlapping grains
- Handle dense scenes (50+, 100+ grains) via tiling
- Foreign matter detection (segmentation if labels available, else bbox)
- No-rice detection → `NO_RICE_DETECTED`
- Unresolved rice clusters → NEVER misclassified as foreign matter
- Unique grain ID assignment
- Per-grain confidence + HIGH/MEDIUM/LOW labels
- Per-grain crop extraction (RGBA PNG with mask alpha)
- Frontend demo: upload → analyze → click grain → view details
- Honest measured metrics, 9-category evaluation, failure analysis

❌ **OUT OF SCOPE (deferred to Phase 2+):**
- 14-parameter classification (Broken, Damaged, Discoloured, Chalky, Red, Dehusked, Immature, Sprouted/Weevilled, Length, Breadth, L/B, Foreign Matter %, Admixture, Total)
- Grading / Grade A / Grade B decisions
- KMS 2026-27 compliance screening
- Defect classification on per-grain crops
- Any claim about official lot compliance

### Required Data Downloads Before Training

See [docs/DATASET.md](docs/DATASET.md) for full details. Required datasets are NOT stored in this repository due to size. You must download:

| Dataset | Public Source | Local Path | Size Estimate |
|---------|---------------|------------|---------------|
| GrainSet Rice v3 | Figshare DOI 10.6084/m9.figshare.22987292.v3 | `data/raw/grainset_rice_v3/` | ~5–10 GB (images + masks) |
| GrainDet Rice v2 | Figshare DOI 10.6084/m9.figshare.23686368.v2 | `data/raw/graindet_rice_v2/` | ~2–5 GB (8 classification folders) |

After downloading, run Step 2 (audit) to verify mask encoding before training.

---

---

## 🌾 Key System Highlights

- **Regression-tested cases**: 1-grain, 2-grain, mixed-object, and larger test images are covered. Arbitrary production images are not universally validated.
- **Rice/non-rice gate**: Existing Case 1/Case 2 behavior is retained. The optional learned gate artifact used synthetic feature vectors; its score is experimental, not validated on real images.
- **No-rice path**: If the gate rejects the image, processing stops with:
  > *"No rice grains detected. Please upload an image containing rice grains."*
- **Exact 14 Parameters**:
  - **Per-Grain**: Broken, Damaged/Slightly Damaged, Discoloured, Chalky, Red, Dehusked, Immature/Shrunken, Sprouted/Weevilled, Length, Breadth, L/B Ratio.
  - **Sample-Level**: Foreign Matter, Admixture of Lower Class, Total Count.
- **Explicit model status**: Chalky is Undetermined without real labeled data. Damaged and sprouted/weevilled checkpoints are synthetic demonstrations, not validated models. Foreign Matter uses a labeled heuristic fallback while YOLO data/weights are unavailable.
- **Standards screening**: Configured values are historical/reference values; current KMS 2026-27 is unverified. Image counts/areas are not official weight-based measurements and do not establish lot compliance.
- **No 12 MP Rejection Gate**: Evaluates post-segmentation pixels-per-grain and Laplacian blur variance rather than total megapixels.
- **Dual Calibration Engine**: Supports Mode A (physical reference / ArUco markers or manual mm scale) and Mode B (honest pixel reporting when uncalibrated).
- **Interactive Modern Dashboard**: React + Vite + TypeScript + Tailwind CSS UI with zoomable annotated viewer, per-grain inspector, standards screening table, and CSV/JSON export.

---

## 📁 Repository Structure

```
rice-quality-ai/
├── backend/
│   ├── app/
│   │   ├── api/routes.py            # FastAPI endpoints
│   │   ├── main.py                  # App entrypoint with CORS & static mount
│   │   ├── schemas/models.py        # Pydantic v2 schemas
│   │   └── services/
│   │       ├── export.py            # CSV & JSON export formatting
│   │       └── job_manager.py       # Job tracking & async orchestrator
│   └── requirements.txt
├── configs/
│   ├── app.json                     # Server settings
│   ├── inference.json               # Batching & inference configs
│   ├── models.json                  # Model registry & artifact metadata
│   ├── parameters.json              # 14 parameter provenance/status registry
│   └── thresholds.json              # Configurable thresholds with provenance
├── data/
│   ├── manifests/                   # Dataset manifests
│   ├── processed/                   # Processed datasets
│   └── raw/                         # Raw training imagery
├── docs/
│   ├── architecture.md              # System design & component diagrams
│   ├── edge-cases.md                # 1 to 5000+ grains, touching grains, etc.
│   ├── model-training.md            # Training pipelines & metrics
│   ├── parameter-methods.md         # Exact formulas for all 14 parameters
│   ├── standards.md                 # India KMS 2026-27 specifications & limitations
│   ├── viva-explanation.md          # Viva questions & examiner explanations
│   └── model-validation-audit.md    # Model and dataset evidence audit
├── frontend/
│   ├── dist/                        # Production compiled bundle
│   ├── src/
│   │   ├── components/              # Dashboard, viewer, inspector, tables
│   │   ├── App.tsx                  # Main application coordinator
│   │   └── types.ts                 # Full TypeScript type interfaces
│   ├── package.json
│   └── vite.config.ts
├── ml/
│   ├── admixture.py                 # Geometry outlier diagnostic; not a lower-class classifier
│   ├── calibration.py               # ArUco marker & manual scale calibration
│   ├── classifiers.py               # Experimental synthetic damaged/sprouted checkpoints
│   ├── colour.py                    # LAB DeltaE discolouration & red grain detection
│   ├── config.py                    # Dynamic configuration loader
│   ├── foreign_matter.py            # Heuristic fallback; YOLO checkpoint unavailable
│   ├── geometry.py                  # Ellipse fitting, L/B ratio, robust whole kernel
│   ├── pipeline.py                  # End-to-end RiceQualityPipeline orchestrator
│   ├── preprocessing.py             # EXIF rotation & image loading guards
│   ├── quality.py                   # Laplacian blur & quality tier assignment
│   ├── segmentation.py              # Rice detector & watershed instance segmentation
│   ├── standards.py                 # Historical/reference screening engine
│   └── texture.py                   # GLCM & brightness features for chalkiness
├── models/
│   ├── chalky/                      # Logistic Regression model & scaler
│   ├── damaged/                     # VGG-19 model weights & class mapping
│   ├── foreign_matter/              # YOLO11n dataset configs & manifests
│   ├── segmentation/                # Mask R-CNN configs & manifests
│   ├── sprouted_weevilled/          # ResNet-18 model weights & class mapping
│   └── evaluation_report.json       # Artifact metadata; synthetic metrics are invalidated
├── standards/
│   └── india_kms_2026_27_raw_rice.json # Historical/reference values; current season unverified
├── tests/
│   ├── test_api.py                  # API endpoint tests
│   └── test_pipeline_24_cases.py    # Automated suite of all 24 required test cases
├── training/
│   ├── evaluate_models.py           # Evaluation compiler
│   ├── train_chalky.py              # Logistic Regression training script
│   ├── train_damaged.py             # VGG-19 transfer learning script
│   ├── train_foreign_matter.py      # YOLO fine-tuning setup
│   ├── train_segmentation.py        # Mask R-CNN setup
│   └── train_sprouted.py            # ResNet-18 transfer learning script
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Backend Setup & Run

```bash
cd c:\Users\saikiran\Desktop\grain\rice-quality-ai

# Activate your Python environment or use system python
# Install dependencies if not already installed:
pip install -r backend/requirements.txt

# Start the unified backend (serves API + compiled frontend):
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The application is now accessible at:
- Web Application: **http://localhost:8000**
- Interactive Swagger API Docs: **http://localhost:8000/docs**
- Health Check: **http://localhost:8000/health**

---

### 2. Frontend Development Server (Optional)

If developing the frontend with hot module reloading:

```bash
cd c:\Users\saikiran\Desktop\grain\rice-quality-ai\frontend
npm install
npm run dev
```
Open **http://localhost:5173** (proxies API requests to port 8000).

---

### 3. Running Automated Tests

Run the complete backend test suite:

```bash
cd c:\Users\saikiran\Desktop\grain\rice-quality-ai
python -m pytest tests -v
```

The regression suite is not a real-world model validation study. It covers the API, gate cases and clean one-/two-grain paths; see the audit for limits.

---

### 4. Training / Retraining Models

Real labeled training data is not stored under `data/raw` or `data/processed`. External candidates, licenses and gaps are tracked in `data/manifests/datasets.json`. Training must stop if the real labeled manifest is missing.

Trainer status:

```bash
# Requires real grain images/masks and data/processed/chalky/manifest.csv:
python training/train_chalky.py

# Intentionally blocked until real data and a reviewed trainer exist:
python training/train_damaged.py

# Requires a real manifest; synthetic crops are disabled:
python training/train_sprouted.py

# Writes proposed setup only; it does not train YOLO:
python training/train_foreign_matter.py

# Writes setup only; no local segmentation annotations/checkpoint are present:
python training/train_segmentation.py

# Compiles artifact metadata; does not calculate new evaluation metrics:
python training/evaluate_models.py
```

### Dataset Download and Placement

Download candidate archives into the project root's `data/raw/<dataset_id>/` folder and extract unchanged copies into `data/processed/<dataset_id>/`. Keep the original archive and citation/license text in `data/raw`; never put images inside `data/manifests`.

| Candidate | Public page | Local placement | What it may support |
|---|---|---|---|
| GrainDet-Rice | [Figshare v2](https://doi.org/10.6084/m9.figshare.23686368.v2) | `data/raw/graindet_rice_v2/`, then `data/processed/graindet_rice_v2/` | Single-grain broken, sprouted, unripe, pest-attacked and impurity candidates. Reported totals conflict; inspect archive labels/groups first. Pest-attacked is not confirmed as weevilled. IDE download attempt returned HTTP 403; use the browser and verify archive size/checksum. |
| GrainSet Rice | [Figshare v3](https://doi.org/10.6084/m9.figshare.22987292.v3) | `data/raw/grainset_rice_v3/`, then `data/processed/grainset_rice_v3/` | Single-grain images/masks and expert quality metadata candidates. License records conflict (Figshare says CC BY 4.0; project page says CC BY-NC-SA 4.0); do not use/publish until the publisher confirms terms. |
| Rice Grain Defects v1 | [Roboflow version](https://universe.roboflow.com/rice-grain-wmeky/rice-grain-defects/dataset/1) | `data/raw/roboflow_rice_grain_defects_v1/`, then `data/processed/roboflow_rice_grain_defects_v1/` | 2,000 detection images; public class IDs are not named. Do not map/train until the export's class file and source grouping are reviewed. |
| Rice quality parameters v1 | [Roboflow version](https://universe.roboflow.com/smart-bin-4f8rw/rice-grain-quality-detection/dataset/1) | `data/raw/roboflow_rice_quality_parameters_v1/`, then `data/processed/roboflow_rice_quality_parameters_v1/` | 224 images, `broken_rice`, `chalky_rice`, `foreign_object`, `head_rice`, `unhulled_rice`; exploratory only, source groups/classes per original sample remain uncertain. |
| Rice variety images | [Kaggle page](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) | `data/raw/murat_koklu_rice_image_dataset_v1/`, then `data/processed/murat_koklu_rice_image_dataset_v1/` | Five variety folders, 75,000 images total. Variety labels are not verified clean/normal defect labels; manual review is needed before using any as chalky hard negatives. |
| Wheat grain v1 | [Roboflow version](https://universe.roboflow.com/yusuf-curum-qkv4e/wheat-grain-1lgcb/dataset/1) | `data/raw/roboflow_wheat_grain_gate_negatives_v1/` | 5,155 classification images, `foreign_particles`, `grain-diseased`, `grain-healthy`, CC BY 4.0. Potential wheat non-rice negatives only; not downloaded or approved. Verify class balance and original source groups before use. |

The trainer input manifests, when real examples have been reviewed, belong at:

```text
data/processed/chalky/manifest.csv
  image_path,mask_path,label,source_group

data/processed/sprouted_weevilled/manifest.csv
  image_path,mask_path,label,source_group

data/processed/foreign_matter/
  images/train|val|test and labels/train|val|test (YOLO format)
```

`source_group` must identify the original physical sample/acquisition group, not a transformed copy. Preserve publisher splits when they are trustworthy; do not make random image-level splits across related samples. After downloading, update `data/manifests/datasets.json` with actual file counts, class counts, checksum, license and inspection findings. A dataset in these folders is not automatically approved for training.

---

## 📊 Summary of 14 Project Parameters

| # | Parameter | Scope | Current method/status | Reference status |
|---|---|---|---|---|
| 1 | **Broken** | Per-Grain | Geometry estimator; Undetermined when reference is insufficient | Historical/reference only; count is not weight |
| 2 | **Damaged** | Per-Grain | Synthetic VGG-19 demo; experimental/uncalibrated | Historical/reference only |
| 3 | **Discoloured** | Per-Grain | LAB color-distance heuristic | Historical/reference only |
| 4 | **Chalky** | Per-Grain | Unavailable; returns Undetermined | Historical/reference only |
| 5 | **Red Grain** | Per-Grain | LAB/HSV visual proxy | Historical/reference only |
| 6 | **Dehusked** | Per-Grain | Visual bran proxy; not chemical staining | Historical/reference only |
| 7 | **Immature / Shrunken** | Per-Grain | Relative-geometry proxy | Experimental proxy |
| 8 | **Sprouted / Weevilled** | Per-Grain | Synthetic ResNet-18 demo; uncalibrated | Experimental; real classes not validated |
| 9 | **Foreign Matter** | Sample-Level | Color/contour fallback; YOLO unavailable | Historical/reference only; area/count is not weight |
| 10 | **Admixture of Lower Class** | Sample-Level | Unsupported; geometry outliers diagnostic only | Historical/reference only |
| 11 | **Length** | Per-Grain | Major-axis pixels unless calibrated | Project measurement |
| 12 | **Breadth** | Per-Grain | Minor-axis pixels unless calibrated | Project measurement |
| 13 | **L/B Ratio** | Per-Grain | Dimensionless geometric ratio | Project measurement |
| 14 | **Total Count** | Sample-Level | Accepted rice instances after existing gate/segmentation | Image count |

Configured standards are historical/reference values. Current KMS 2026-27 limits are not verified. Image count/area is not an official weight percentage, and 30 grains is only a project screening threshold, not a government sampling requirement.
