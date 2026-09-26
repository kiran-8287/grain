# Rice Quality Analyzer Work Log

This log is the durable handoff for IDE AI work in this repository. Append a dated entry at the end of each task. Do not replace earlier entries. Keep it Markdown/plain text, factual, and concise.

## 2026-09-26 — Model Audit and Master-Prompt Pass

### Starting state

- The shared worktree already contained uncommitted Case 3/model-audit work from the preceding session. This entry records the state and work completed, but does not claim every pre-existing changed file was authored in this pass.
- `data/raw` and `data/processed` contain only `.gitkeep`; no usable image/annotation training corpus is present.
- Rice-gate and segmentation behavior are protected by existing tests and have not been changed in this pass.

### Changes completed in this pass

- Created `configs/parameters.json` describing exactly 14 project parameters, provenance type/status, dataset, units and official-equivalence flags.
- Created `data/manifests/datasets.json` with publicly inspected candidates, source links, reported sizes/classes/splits/license details, suitability, access blockers and provenance caveats.
- Added `progress/WORK_LOG.md` and a project instruction requiring future AI tasks to append their changed files, tests, blockers and next steps here.
- Added external candidate evidence: GrainDet-Rice Figshare record (CC BY 4.0); GrainSet Rice Figshare record (license conflicts with the project page); two Roboflow rice-defect/quality datasets (CC BY 4.0); Murat Koklu variety images (Kaggle CC0); UCI Cammeo/Osmancik feature table (CC BY 4.0); IIT Indore milled-rice damage paper/repository record.
- Attempted the public GrainDet-Rice archive download. Figshare redirected and the request ended with HTTP 403 at 146,966,036 bytes of an advertised 1,835,500,275-byte archive. The incomplete file is named `data/downloads/graindet_rice.zip.partial`; it was not extracted or used. No real model was trained.
- Marked the synthetic rice-gate artifact experimental in registry/metadata/output and changed its legacy training entrypoint to require a real manifest. Gate inference decisions and segmentation code were not modified.
- Marked synthetic VGG-19 damaged outputs experimental/uncalibrated and blocked the synthetic trainer from writing fake metrics/checkpoints.
- Kept ResNet-18 sprouted/weevilled experimental; its real-data trainer rejects a dataset missing either the sprouted or weevilled positive class unless the corpus explicitly labels a combined class.
- Kept chalky inference at Undetermined without real labeled images. Existing mask-only LAB/GLCM feature extraction and real-manifest training setup remain available.
- Marked YOLO foreign matter unavailable and retained the heuristic warning/fallback. Setup manifests now list candidate sources separately from verified/local datasets.
- Changed admixture to unsupported: counts/fraction are null. Mahalanobis output is identified only as a geometry-outlier diagnostic, not lower-class admixture.
- Made image quality use accepted grain-mask pixels for blur, illumination and clipping. Added explicit UNRELIABLE hard-failure criteria; they are engineering thresholds requiring human-rated calibration. Added black-vs-white background invariance and reachability regression checks.
- Updated README, audit, architecture, edge-case, standards, training and parameter docs to remove stale synthetic-validation and current-KMS claims. Thirty grains is labeled a project threshold, not a government requirement.
- Regenerated `models/evaluation_report.json` from current artifact metadata; invalid synthetic metrics are null, and the report script says it compiles metadata rather than evaluating models.

### Verification so far

- Dedicated rice gate tests: 8 passed after gate provenance changes.
- Focused audit registry tests: 5 passed.
- Focused image-quality tests: 3 passed, including reachable UNRELIABLE and black/white background invariance.
- Focused admixture diagnostic test: 1 passed.
- Frontend production build succeeded after the quality UI/types update; a final build is still required after the latest UI edits.
- JSON registries/manifests parsed successfully; edited training/ML modules compiled in earlier checks.
- The full backend suite has not yet been rerun after the latest master-prompt changes.

### Known limits / next steps

- Do not claim any model is real-data validated. No candidate archive has been locally inspected end-to-end or used for training.
- GrainDet-Rice counts conflict between the Figshare API and project page; pest-attacked does not establish weevilled; source grouping and archive labels need inspection after obtaining the full archive.
- GrainSet license metadata conflicts between the Figshare API and project page; resolve with publisher before use. Its 2.62 GB archive is not downloaded.
- Roboflow 2,000-image class IDs are unnamed in the card. The 224-image dataset has only 180/22/22 split and no verified independent sample groups; do not use for production metrics.
- IIT Indore reports 8,048 images/seven damage classes, but its repository record has no attached data or explicit reuse license.
- Verify remaining README/docs/config claims, rerun all backend tests, the dedicated Case 1/2 suite, one- and two-grain Case 3 suite, frontend build and `git diff --check` before closing this task.
- Do not proceed to Case 4 or alter rice gate/segmentation behavior without a regression requiring it.

## 2026-09-26 — Dataset Download Guidance

- User asked whether required datasets can be downloaded online and where to put them.
- Recommended the public GrainDet-Rice and GrainSet Rice archives for candidate rice-grain annotations; documented the GrainDet HTTP 403 partial download and conflicting GrainSet license metadata.
- Identified small Roboflow CC BY 4.0 datasets for exploratory broken/chalky/foreign-object work only; they lack verified source groups/class support for production evaluation.
- Provided the exact `data/raw/<dataset_id>/` archive and `data/processed/<dataset_id>/` extraction destinations, plus training manifest paths for chalky/damaged/sprouted and YOLO directory layout.
- No real model was trained and no partial archive was treated as usable data.

## 2026-09-26 — Rice gate architecture fix and regression verification

### Summary

- Fixed the root cause behind the false NOT_RICE outcome in crowded rice scenes.
- The gate no longer treats a low rice-object fraction as an image-level rejection when rice is present but missed by the detector.
- Detector sparsity is now recorded as a warning only, while the image remains valid when at least one rice detection passes the gate.

### Files changed

- [ml/rice_gate.py](../ml/rice_gate.py)
- [ml/pipeline.py](../ml/pipeline.py)
- [tests/test_rice_gate.py](../tests/test_rice_gate.py)

### What was corrected

- Removed the hard sparse-object rejection path that forced many-rice scenes into NOT_RICE.
- Kept the background-only and no-rice cases stopping the pipeline as before.
- Preserved the warning path so poor detector recall is surfaced without poisoning the final decision.
- Added the many-rice regression to protect against reintroducing the false rejection.

### Verification evidence

- `python -m pytest tests/test_rice_gate.py -q`
  Result: 9 passed in 24.30s
- `python -m pytest -q`
  Result: 63 passed, 1 warning in 48.40s

### Notes

- The warning is a Starlette deprecation warning from the API test client and does not indicate a project failure.
- The gate fix is complete based on the passing regression suite and the full backend suite.

### Next step

- Once real public datasets are downloaded and placed under the project data folders, the next phase is to train the image-level rice gate on real labeled corpora rather than relying on heuristics or synthetic proxy artifacts.
