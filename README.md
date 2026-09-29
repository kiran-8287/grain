# Rice Quality AI

Rice Quality AI is a rice-grain image analysis application for inspecting uploaded grain images, computing per-grain measurements, and comparing the observed sample against the configured quality thresholds.

This repository has been reset to a clean baseline: no old dataset archives, model checkpoints, training scripts, or obsolete evaluation artifacts remain in the project. The next phase can begin only after datasets are manually selected and reviewed.

## Current project scope

- Backend API for image analysis and export
- Frontend dashboard for viewing results
- Quality rules and standards comparison
- General rice-image processing utilities
- Clean repository ready for future dataset-driven model work

## Run the backend

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

## Notes

- No training pipeline is committed in this reset branch.
- No dataset downloads or model artifacts are included.
- Future dataset selection and model training will be implemented in a separate phase.
