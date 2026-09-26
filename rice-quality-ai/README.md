# Rice Quality AI — Automated Raw Milled Rice Quality Assessment System

An end-to-end computer vision and machine learning platform for assessing the quality of raw milled rice grains from arbitrary photographs. Evaluates all 14 project parameters against the Government of India Kharif Marketing Season 2026-27 (KMS 2026-27) Uniform Specification.

---

## 🌾 Key System Highlights

- **Analyzes Any Grain Count**: From 1 grain ($N=1$) to several thousand grains on white, black, wooden, or textured backgrounds.
- **Strict Anti-Fabrication Guarantee**: If no rice grains are present, processing stops immediately with the exact required message:
  > *"No rice grains detected. Please upload an image containing rice grains."*
- **Exact 14 Parameters**:
  - **Per-Grain**: Broken, Damaged/Slightly Damaged, Discoloured, Chalky, Red, Dehusked, Immature/Shrunken, Sprouted/Weevilled, Length, Breadth, L/B Ratio.
  - **Sample-Level**: Foreign Matter, Admixture of Lower Class, Total Count.
- **Multi-Label Defect Engine**: Individual grains can simultaneously possess multiple defects (e.g. Broken + Damaged + Chalky).
- **Official Standards Engine**: Compares assessable metrics against India KMS 2026-27 limits without falsely claiming an official government certification.
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
│   ├── standards.json               # Standards configuration
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
│   └── viva-explanation.md          # Viva questions & examiner explanations
├── frontend/
│   ├── dist/                        # Production compiled bundle
│   ├── src/
│   │   ├── components/              # Dashboard, viewer, inspector, tables
│   │   ├── App.tsx                  # Main application coordinator
│   │   └── types.ts                 # Full TypeScript type interfaces
│   ├── package.json
│   └── vite.config.ts
├── ml/
│   ├── admixture.py                 # Robust Mahalanobis admixture outlier detector
│   ├── calibration.py               # ArUco marker & manual scale calibration
│   ├── classifiers.py               # VGG-19 damaged & ResNet-18 sprouted models
│   ├── colour.py                    # LAB DeltaE discolouration & red grain detection
│   ├── config.py                    # Dynamic configuration loader
│   ├── foreign_matter.py            # YOLO foreign matter detection
│   ├── geometry.py                  # Ellipse fitting, L/B ratio, robust whole kernel
│   ├── pipeline.py                  # End-to-end RiceQualityPipeline orchestrator
│   ├── preprocessing.py             # EXIF rotation & image loading guards
│   ├── quality.py                   # Laplacian blur & quality tier assignment
│   ├── segmentation.py              # Rice detector & watershed instance segmentation
│   ├── standards.py                 # KMS 2026-27 screening engine
│   └── texture.py                   # GLCM & brightness features for chalkiness
├── models/
│   ├── chalky/                      # Logistic Regression model & scaler
│   ├── damaged/                     # VGG-19 model weights & class mapping
│   ├── foreign_matter/              # YOLO11n dataset configs & manifests
│   ├── segmentation/                # Mask R-CNN configs & manifests
│   ├── sprouted_weevilled/          # ResNet-18 model weights & class mapping
│   └── evaluation_report.json       # Consolidated validation metrics
├── standards/
│   └── india_kms_2026_27_raw_rice.json # Official specification source of truth
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

Run the complete test suite covering all 24 required test cases:

```bash
cd c:\Users\saikiran\Desktop\grain\rice-quality-ai
python -m pytest tests/test_pipeline_24_cases.py tests/test_api.py -v
```

All 31 tests pass with 100% verified status.

---

### 4. Training / Retraining Models

To retrain the machine learning components:

```bash
# 1. Chalky grain Logistic Regression:
python training/train_chalky.py

# 2. Damaged grain VGG-19 Transfer Learning:
python training/train_damaged.py

# 3. Sprouted/Weevilled ResNet-18 Transfer Learning:
python training/train_sprouted.py

# 4. Foreign Matter YOLO Setup:
python training/train_foreign_matter.py

# 5. Mask R-CNN Segmentation Setup:
python training/train_segmentation.py

# 6. Compile Unified Evaluation Report:
python training/evaluate_models.py
```

---

## 📊 Summary of 14 Project Parameters

| # | Parameter | Scope | Method / Algorithm | KMS 2026-27 Ref Limit |
|---|---|---|---|---|
| 1 | **Broken** | Per-Grain | Robust Iterative Median ($< 0.75 L_{\text{whole}}$) | 25.0% max |
| 2 | **Damaged / Slightly Damaged** | Per-Grain | VGG-19 CNN ($224 \times 224$ padded crop) | 3.0% max |
| 3 | **Discoloured** | Per-Grain | Adaptive CIELAB $\Delta E > 15.0$ | 3.0% max |
| 4 | **Chalky** | Per-Grain | Logistic Regression on LAB + GLCM features | 5.0% max |
| 5 | **Red Grain** | Per-Grain | Red cuticle surface coverage $\ge 25\%$ | 3.0% max |
| 6 | **Dehusked** | Per-Grain | Visual bran coverage proxy (labeled non-chemical) | 13.0% max |
| 7 | **Immature / Shrunken** | Per-Grain | Geometric breadth & area profile proxy | Experimental Proxy |
| 8 | **Sprouted / Weevilled** | Per-Grain | ResNet-18 CNN on tip/cavity morphology | Experimental ML |
| 9 | **Foreign Matter** | Sample-Level | Full-image YOLO11n object detector | 0.5% max |
| 10 | **Admixture of Lower Class**| Sample-Level | Robust Mahalanobis outlier detection | 6.0% max (Grade A) |
| 11 | **Length** | Per-Grain | Ellipse major axis (mm or pixels) | Varietal |
| 12 | **Breadth** | Per-Grain | Ellipse minor axis (mm or pixels) | Varietal |
| 13 | **L/B Ratio** | Per-Grain | Dimensionless ratio ($\text{Length} / \text{Breadth}$) | Varietal |
| 14 | **Total Count** | Sample-Level | Accepted instance segmentation count | Sample Count |
