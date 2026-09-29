# GRAIN QUALITY ANALYZER

GRAIN QUALITY ANALYZER is a rice-grain image analysis application for inspecting uploaded grain images, segmenting individual grains, and computing mask-based geometric measurements for each detected grain.

## Current Phase

**Phase 1 — Rice Grain Segmentation & Geometry**

The system is currently performing:
- Rice/non-rice detection and instance segmentation
- Individual grain mask extraction and visualization
- Per-grain geometric measurements (length, breadth, L/B ratio)
- Total grain count from actual segmentation instances

Future phases will add defect classification, standards screening, and official grading once validated models and datasets are approved.

## Project Structure

```
A:\grain\
├── .git/                      # Git repository metadata
├── .kilo/                     # Kilo configuration
├── .pytest_cache/             # Python test cache
├── 27 tests/                  # Test artifacts
├── ai_work_log/               # AI progress and work logs
├── assets/                    # Static assets (logo, etc.)
├── backend/                   # FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py     # API endpoints
│   │   ├── schemas/
│   │   │   └── models.py     # Pydantic models
│   │   ├── services/
│   │   │   ├── export.py     # JSON/CSV export
│   │   │   ├── job_manager.py
│   │   │   └── run_logger.py
│   │   └── main.py           # FastAPI entry point
│   ├── .env.example
│   ├── package-lock.json
│   └── requirements.txt
├── configs/                   # Configuration files
│   ├── app.json
│   ├── inference.json
│   ├── models.json
│   ├── parameters.json
│   └── thresholds.json
├── data/                      # Data and audit files
│   ├── audit/
│   ├── runs/
│   └── datasets/
├── datasets/                  # Dataset storage
├── docs/                      # Documentation
├── frontend/                  # React + Vite frontend
│   ├── src/
│   │   ├── components/       # UI components
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── types.ts
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig.json
├── logs/                      # Application logs
├── ml/                        # Machine learning modules
│   ├── config.py
│   ├── quality/               # Quality classification modules
│   ├── segmentation/          # Segmentation pipeline modules
│   └── standards/             # Standards comparison modules
├── models/                    # Model artifacts and mappings
│   ├── foreign_matter/
│   └── segmentation/
├── node_modules/              # Frontend dependencies
├── package.json               # Root package.json (concurrently)
├── package-lock.json
├── render.yaml                # Render deployment config
├── scripts/                   # Utility and audit scripts
├── standards/                 # Standards data files
├── tests/                     # Backend tests
├── uploads/                   # Uploaded images
├── .env.example               # Environment variables template
├── .gitignore
├── AGENTS.md
├── DATASET_AUDIT_REPORT.md
├── README.md
├── newtasks.md
├── render.yaml
└── tasks.txt
```

## Quick Start

From the repository root (`A:\grain\`):

```bash
# Install root dependencies (concurrently)
npm install

# Start both backend and frontend
npm run dev
```

This starts:
- **Backend**: FastAPI server at `http://localhost:8000`
- **Frontend**: Vite dev server at `http://localhost:5173`

### Run services separately

```bash
# Backend only
npm run dev:backend

# Frontend only
npm run dev:frontend
```

### Build frontend

```bash
npm run build
```

## Backend

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

## Current Capabilities

### Phase 1 — Segmentation & Geometry
- Upload rice grain images
- Instance segmentation of individual grains
- Per-grain mask visualization with unique IDs
- Total grain count from segmentation results
- Per-grain geometric measurements:
  - Length (pixels or mm with calibration)
  - Breadth (pixels or mm with calibration)
  - L/B Ratio
  - Mask area, solidity, confidence
- CSV/JSON export of current measurements

### Not Yet Implemented
- Defect classification (broken, damaged, discolored, chalky, red, dehusked, immature, sprouted/weevilled)
- Foreign matter detection
- Admixture analysis
- Official standards screening and grading

## Notes

- No training pipeline is committed in this branch.
- No dataset downloads or model checkpoints are included.
- Future dataset selection and model training will be implemented in a separate phase after manifest review and source-group safety checks.
