# Grain Quality Analysis — Project Context

## Purpose
This document captures the current repository state, implementation scope, and validation status so AI coding agents and reviewers have a single authoritative reference.

## Recovery Status
- Repository recovered from `https://github.com/kiran-8287/grain.git` after accidental deletion.
- Current working tree matches the committed history; all expected baseline files are present.

## Verified Baseline
- Backend tests pass: 34/34 tests green (unit and integration coverage).
- Frontend builds successfully from `frontend/`.
- Safe cleanup completed: removed unused assets, source `__pycache__/`, `.pytest_cache/`, and outdated lock files.

## Production Pipeline Status
- Active pipeline: **classical CV segmentation + rule-based classifiers**.
- Trained YOLO / Mask R-CNN model weights are **not committed** in this repository.
- Model registry paths (`configs/models.json`) are **placeholders** and should not be treated as ready-to-run.

## Rice-Presence Gate
- Implemented: `ml/segmentation/rice_gate.py`.
- Behavior: distinguishes “objects detected” from “rice detected”.
- Verified by tests: foreign-matter-only images, empty/background-only images, single-grain images, and mixed scenes.

## Whole vs Broken
- Criterion: length ratio vs a reference whole-kernel length (default threshold 0.75).
- Reference sources:
  - Profile supplied by caller: classified as **Reliable**.
  - Sample-derived candidate population: classified as **Proxy / Sample-Derived**.
  - No reference available: classified as **Undetermined**.
- Default `grain_profiles/default_rice.json` is **proxy/pixel-based** and only valid at the original capture distance; do not treat it as a universal physical reference.

## Segmentation Behavior
- Classical CV fallback is the active code path.
- Merge handling: touching grains and unresolved clusters are detected and reported.
- Uncertain or rejected instances are surfaced in per-grain tables and summary counts.

## Foreign Matter and Admixture
- Foreign-matter detection uses heuristic methods in `ml/quality/foreign_matter.py`.
- Admixture estimation is available; results are reported separately and are not merged into the rice count.

## Image Quality
- Quality tiering: GOOD / FAIR / POOR / UNRELIABLE.
- Hard-failure rules can mark a run UNRELIABLE based on blur, grain area, segmentation confidence, uncertain-grain fraction, illumination uniformity, and clipped-pixel fraction.
- Quality result is exposed in the API response under `quality`.

## Calibration
- Supported: ArUco marker calibration (configurable dictionary and marker size in `configs/thresholds.json`).
- When no calibration is supplied, analysis runs in pixel units.

## Frontend
- Upload and camera capture flow present.
- Per-grain table supports selection, unit display, and length/breadth/L-B-ratio columns.
- Export controls: annotated image download, CSV export, JSON export, and reset flow.

## API Contract (selected fields)
- Response envelope: `success`, `job_id`, `rice_detected`, `sample`, `calibration`, `quality`, `grains`, `foreign_matter`, `summary`, `standards`, `warnings`, `processing_time_seconds`.
- Summary fields used by the UI: `total_count`, `whole_count`, `broken_count`, `broken_percent`, `whole_reference_length`, `reference_source`, `measurement_unit`.

## Standards Screening
- A standards/screening layer exists.
- Results are labeled as `NOT ASSESSABLE`, `NOT DETERMINABLE`, or `WITHIN REFERENCE LIMIT` / `EXCEEDS REFERENCE LIMIT` where supported.
- Do not assert official lot-grade compliance or KMS limits unless a verified authoritative source and matching measurement basis are explicitly available.

## Documentation Boundaries
- Validated: classical CV pipeline, rice gate, logging/job lifecycle, whole-vs-broken rules, quality tiering, merge detection, calibration path, and current UI flows.
- Experimental / Proxy / Not-Yet-Implemented: trained model inference, official mass-based grading, universal physical references, and committed model weights.
- No architectural changes, backend rewrites, or frontend redesigns were performed during baseline verification.
