# Rice Quality AI Project Progress Log

## 2026-09-26 — Dataset audit and continuation handoff

### What was already fixed before this step
- Corrected the rice/non-rice gate bug in the pipeline so a dense rice scene is not hard-rejected because the detector missed many grains.
- Preserved the no-rice and empty-image stop conditions.
- Kept detector sparsity as a warning, not a final image-level rejection.
- Verified the fix with regression tests.

### Verification evidence
- `python -m pytest tests/test_rice_gate.py -q` -> 9 passed in 24.30s
- `python -m pytest -q` -> 63 passed, 1 warning in 48.40s

### Dataset staging audit completed now
Local raw data was checked and confirmed present under `data/raw`.

Counts observed:
- graindet_rice_v2: 24,768 files
- grainset_rice_v3: 61,925 files
- murat_koklu_rice_image_dataset_v1: 75,002 files
- roboflow_rice_grain_defects_v1: 4,004 files
- roboflow_rice_quality_parameters_v1: 452 files

These folders were found as extracted real datasets and are ready for source-group and class-map review, but none are approved for production training yet.

### Project status
- Rice gate architecture bug: fixed and regression-covered.
- Full backend test suite: passing.
- Dataset archive staging: complete and audited locally.
- Training-ready manifest review: not completed yet.
- All real datasets remain under source review before use in a final model.

### Important caution
This project must remain honest: downloaded datasets are local and staged, but the task is not yet at a final approved training stage. Any future model training should only proceed after manifest review, class-map validation, and source-group safety checks.

### Files/areas touched for this stage
- `rice-quality-ai/ml/rice_gate.py`
- `rice-quality-ai/ml/pipeline.py`
- `rice-quality-ai/tests/test_rice_gate.py`
- `rice-quality-ai/data/manifests/datasets.json`
- `rice-quality-ai/data/processed/`
- `rice-quality-ai/progress/WORK_LOG.md`
- `ai_work_log/AI_PROGRESS.md` (this file)

### Next step
Perform the source-group and class-map audit for each dataset and then create the task-specific approved manifests for rice-gate, segmentation, and defect classification only after the evidence is reviewed.

## 2026-09-26 — Continuation: dataset-to-task audit and manifest readiness review

### What was done now
- Confirmed all downloaded raw datasets are extracted under `data/raw`.
- Review the current task readiness map and candidate assignments in `data/manifests/dataset_ready_map.json`.
- Matched each dataset to a likely task family without approving any training corpus.
- Updated the durable task record and kept it in both the repo work log and the root-level progress log.

### Task mapping by dataset
- `rice_gate`: `murat_koklu_rice_image_dataset_v1`, `grainset_rice_v3`.
  - Not ready: image-level binary label set and source-group metadata remain under review.
- `segmentation`: `graindet_rice_v2`, `grainset_rice_v3`.
  - Not ready: segmentation labels and source-safe grouping must be reviewed.
- `foreign_matter`: `roboflow_rice_grain_defects_v1`, `roboflow_rice_quality_parameters_v1`.
  - Not ready: class IDs and sample grouping need audit before YOLO training.
- `chalky`: `roboflow_rice_grain_defects_v1`, `roboflow_rice_quality_parameters_v1`, `murat_koklu_rice_image_dataset_v1`.
  - Not ready: no approved chalky manifest exists yet.
- `sprouted_weevilled`: `graindet_rice_v2`, `grainset_rice_v3`.
  - Not ready: real label taxonomy and source grouping still need inspection.

### Why this is still blocked
- `grainset_rice_v3` has a license conflict between Figshare and the project page.
- Public label sets require source-group review to avoid leakage across physical sample groups.
- No task-specific manifest has been approved for training; all task manifests remain empty and intentionally staging-only.

### Verification evidence
- `python -m pytest tests/test_rice_gate.py -q` -> 9 passed.
- `python -m pytest -q` -> 63 passed, 1 warning.

### Next concrete action
- Only after source-group-safe review and class-map confirmation should we create reviewed manifests under `data/processed/<task>/manifest.csv` or YOLO export folders.
- Model training remains intentionally postponed until this review is complete.

## 2026-09-27 — Final verification, metadata fix, and performance regression repair

### Summary
- Fixed the final three failing checks from the repo-wide verification run.
- Clarified the manifest wording so it explicitly states that no local images or annotations are approved for training.
- Enforced the real-manifest requirement for the legacy rice-gate training entrypoint.
- Added a bounded large-sample fast path so the dense 50-grain pipeline remains under the processing-time threshold.

### Files changed
- `rice-quality-ai/data/manifests/datasets.json`
- `rice-quality-ai/training/train_rice_gate.py`
- `rice-quality-ai/ml/segmentation.py`
- `rice-quality-ai/ml/pipeline.py`
- `rice-quality-ai/progress/WORK_LOG.md`

### Verification evidence
- `cd 'A:\grain\rice-quality-ai'; python -m pytest tests/test_audit_metadata.py::test_public_dataset_manifest_distinguishes_candidates_from_training_data tests/test_audit_metadata.py::test_legacy_gate_training_requires_real_manifest tests/test_pipeline_24_cases.py::test_case_24_huge_sample -q`
  - Result: 3 passed in 2.54s
- `cd 'A:\grain\rice-quality-ai'; python -m pytest -q`
  - Result: 63 passed, 1 warning in 50.34s

### Current status
- The project is green.
- The warning is a Starlette deprecation warning from the API client and does not indicate a failure.
- Training remains deliberately blocked until a reviewed, source-group-safe manifest is approved.

## 2026-09-27 — Downloaded dataset audit and training readiness decision

- Audited all five downloaded/extracted datasets under `rice-quality-ai/data/raw` and corrected stale inventory claims.
- Added `rice-quality-ai/data/manifests/dataset_audit_2026-09-27.json`; updated `datasets.json` and `dataset_ready_map.json` with the measured formats/counts and task blockers.
- GrainDet: 24,767 PNGs; filename-derived groups overlap train/val 1,831, train/test 3,286, val/test 927. All GrainDet basenames also recur in GrainSet.
- GrainSet: 30,962 image/mask pairs; 2,473 filename-derived groups occur in both train and test. License conflict remains; separate `rice.xml` is not local.
- Murat Koklu: 75,000 images represent five rice varieties, not project defect/no-rice labels.
- Roboflow defect export: 2,000 images and valid polygons, but numeric class IDs are unnamed and original source groups are unknown.
- Roboflow quality export: 224 images with boxes and 180/22/22 splits; its foreign-object category and source groups are insufficient for the project task.
- Corrected the IIT Indore candidate record: it is not downloaded and was previously conflated with the Murat Koklu dataset.
- No task passed approval. No approved manifests were created and no model was trained.
- `python -m pytest tests/test_audit_metadata.py -q` -> 5 passed.

Next: resolve class code/license and source-group evidence; acquire missing rice-gate negatives and target-compatible task annotations; then build group-safe approved manifests before training.

Final verification addendum: `python -m pytest -q` -> 63 passed, 1 Starlette deprecation warning in 50.16s. All three dataset JSON files parse; datasets.json has 8 unique records. `git diff --check` emitted no whitespace errors; Git reported a line-ending conversion warning for `ml/pipeline.py`.

## 2026-09-27 — Rice-gate negative candidate research

- The additional `Rice_grain_quality_detection/Real_time_dataset` folder has 10 rice images only; no non-rice negatives or labels.
- Found public Roboflow Wheat grain v1: 5,155 classification images, classes `foreign_particles`, `grain-diseased`, `grain-healthy`, CC BY 4.0, 70/20/10 published split (3,608/1,031/516). Added it as an external not-downloaded candidate in the inventory/readiness/audit files and README.
- Candidate is not approved until local contents, class counts, and original sample groups are audited.
- `training/train_rice_gate.py` is also incomplete for real data: it blocks synthetic fitting and raises `NotImplementedError` after a real manifest check.
- Audit test: `python -m pytest tests/test_audit_metadata.py -q` -> 5 passed. No model trained and no approved manifest created.
- Next: download to `rice-quality-ai/data/raw/roboflow_wheat_grain_gate_negatives_v1/`, audit locally, obtain safe rice-positive source groups, then implement and validate real-image training.
