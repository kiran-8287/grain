# Rice Quality AI — Automated Raw Milled Rice Quality Assessment System

An image-analysis project for raw milled rice. It reports per-grain image predictions and measurements, observed image fractions, and historical/reference screening limits as separate outputs. The current KMS 2026-27 specification has not been verified in this repository; this software does not determine official lot compliance.

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
