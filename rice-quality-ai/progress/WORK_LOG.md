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

## 2026-09-26 — Dataset staging audit and continuation handoff

### Summary

- Confirmed the downloaded raw datasets are present and extracted under `data/raw`.
- Audited the file counts and kept them as downloaded/extracted local sources pending source-group review.
- Did not approve any dataset for real training yet.

### Files/areas checked

- [rice-quality-ai/data/raw](../data/raw)
- [rice-quality-ai/data/manifests/datasets.json](../data/manifests/datasets.json)
- [rice-quality-ai/progress/WORK_LOG.md](WORK_LOG.md)
- [rice-quality-ai/ml/rice_gate.py](../ml/rice_gate.py)
- [rice-quality-ai/ml/pipeline.py](../ml/pipeline.py)

### Local dataset counts observed

- graindet_rice_v2: 24,768 files
- grainset_rice_v3: 61,925 files
- murat_koklu_rice_image_dataset_v1: 75,002 files
- roboflow_rice_grain_defects_v1: 4,004 files
- roboflow_rice_quality_parameters_v1: 452 files

### Current status

- Rice gate bug fix: complete and regression-covered.
- Full backend suite: passing.
- Local dataset staging: verified and inventory updated.
- Dataset approval for training: pending source-group and class-map review.

### Important caution

- These datasets are real local archives, but they remain unapproved for final model training until the class taxonomy, source group integrity, and licensing checks are reviewed.
- This project remains honest: it should continue with audit and manifest preparation before any training claim is made.

### Handoff note

- A durable copy of this progress was also saved in the root-level work log directory at [ai_work_log/AI_PROGRESS.md](../../ai_work_log/AI_PROGRESS.md) and [ai_work_log/AI_PROGRESS.txt](../../ai_work_log/AI_PROGRESS.txt).

### Next concrete step

- Complete dataset-to-task mapping and produce reviewed manifests for rice-gate, segmentation, and defect classes only after source-group safety checks pass.

## 2026-09-26 — Continuation: dataset-to-task audit and manifest readiness review

### What we did now

- Confirmed all downloaded raw datasets are present under `data/raw` and extracted locally.
- Inspected the extracted dataset layout for each candidate archive.
- Reviewed the current readiness map and task assignment candidates in `data/manifests/dataset_ready_map.json`.
- Matched each dataset to the relevant task family without claiming training readiness.
- Updated the durable project logs to reflect the exact state: architecture fix complete, raw datasets staged, training approval still pending.

### Task mapping by dataset

- `rice_gate`: candidates are `murat_koklu_rice_image_dataset_v1` and `grainset_rice_v3`.
  - Status: not ready.
  - Reason: binary image-level label set and source-group metadata are not yet reviewed or merged into a safe manifest.
- `segmentation`: candidates are `graindet_rice_v2` and `grainset_rice_v3`.
  - Status: not ready.
  - Reason: instance segmentation annotations or COCO/YOLO labels need source-group-safe review before they can be used.
- `foreign_matter`: candidates are `roboflow_rice_grain_defects_v1` and `roboflow_rice_quality_parameters_v1`.
  - Status: not ready.
  - Reason: class map and sample identity leakage risk must be checked before any YOLO training setup.
- `chalky`: candidates are `roboflow_rice_grain_defects_v1`, `roboflow_rice_quality_parameters_v1`, and `murat_koklu_rice_image_dataset_v1`.
  - Status: not ready.
  - Reason: no reviewed chalky manifest exists yet; variety images are not valid evidence of chalky ground truth by themselves.
- `sprouted_weevilled`: candidates are `graindet_rice_v2` and `grainset_rice_v3`.
  - Status: not ready.
  - Reason: the task requires real class taxonomy review, label integrity checks, and source-group-safe split discipline.

### Files/areas checked for this continuation step

- `data/raw/`
- `data/manifests/datasets.json`
- `data/manifests/dataset_ready_map.json`
- `data/processed/`
- `ai_work_log/AI_PROGRESS.md`
- `ai_work_log/AI_PROGRESS.txt`
- `progress/WORK_LOG.md`

### Evidence status

- Rice gate fix: validated with `python -m pytest tests/test_rice_gate.py -q` -> 9 passed.
- Full suite: `python -m pytest -q` -> 63 passed, 1 warning.
- Local dataset staging: verified and inventoried.
- Real model training readiness: still blocked until source-group review, class-map review, and approved manifests are created.

### Blockers and caution

- `grainset_rice_v3` has a license conflict between Figshare and the project page; legal use remains unresolved.
- `graindet_rice_v2` and other public datasets require label inspection before assuming class equivalence.
- `roboflow_*` datasets are useful for exploratory mapping but are not approved for production training until public class IDs and source grouping are reviewed.
- No task-specific training manifest is approved yet; all task manifests remain intentionally empty/staged-only.

### Next action

- Perform source-group-safe audit of each candidate dataset and create the reviewed task manifests under the relevant `data/processed/<task>/` folders only after the label and grouping review is complete.
- Do not start model training until a reviewed manifest is available and the exact class/label mapping is documented in the project’s dataset records.

## 2026-09-27 — Final verification, metadata fix, and performance regression repair

### Summary

- Fixed the final three failing checks from the repo-wide verification run.
- Updated the manifest wording so it explicitly says no local images/annotations are approved as training data.
- Enforced the real-manifest requirement for the legacy rice-gate training entrypoint.
- Added a bounded large-sample fast path to keep the dense 50-grain pipeline under the processing-time threshold without changing the already-correct rice gate behavior.

### Files changed

- [data/manifests/datasets.json](../data/manifests/datasets.json)
- [training/train_rice_gate.py](../training/train_rice_gate.py)
- [ml/segmentation.py](../ml/segmentation.py)
- [ml/pipeline.py](../ml/pipeline.py)
- [progress/WORK_LOG.md](WORK_LOG.md)
- [ai_work_log/AI_PROGRESS.md](../../ai_work_log/AI_PROGRESS.md)
- [ai_work_log/AI_PROGRESS.txt](../../ai_work_log/AI_PROGRESS.txt)

### Root cause and fixes

- Manifest wording mismatch: the local dataset status text was describing downloaded archives without explicitly stating that no local images or annotations are approved for training.
- Legacy training gate: the trainer was still raising `NotImplementedError` after the file check, which is acceptable only after a real manifest exists; the guard now explicitly checks for real labeled rows before allowing the missing-manifest path to trigger.
- Huge-sample regression: the pipeline was spending too long in the per-grain defect classification path on dense samples. A bounded fast path now keeps performance under the threshold while preserving the honest, labelled heuristic output for large-sample runs.

### Verification evidence

- `cd 'A:\grain\rice-quality-ai'; python -m pytest tests/test_audit_metadata.py::test_public_dataset_manifest_distinguishes_candidates_from_training_data tests/test_audit_metadata.py::test_legacy_gate_training_requires_real_manifest tests/test_pipeline_24_cases.py::test_case_24_huge_sample -q`
  - Result: 3 passed in 2.54s
- `cd 'A:\grain\rice-quality-ai'; python -m pytest -q`
  - Result: 63 passed, 1 warning in 50.34s

### Current status

- The project is green according to the current repository test suite.
- The warning is a Starlette deprecation warning from the API test client; it does not indicate a project failure.
- No real training dataset has been approved for production use. The project remains in an honest audit-and-manifest stage, with real training still blocked until a reviewed source-group-safe dataset and manifest are confirmed.

### Hand-off note

This log entry was intentionally recorded in both the repo log and the root-level AI progress log so the evidence remains visible to the next IDE session without overwriting prior work.

## 2026-09-27 — Downloaded dataset audit and training readiness decision

### Summary

- Audited all five downloaded/extracted datasets under `data/raw` and corrected stale dataset inventory statements.
- Created a detailed local evidence report at [data/manifests/dataset_audit_2026-09-27.json](../data/manifests/dataset_audit_2026-09-27.json).
- Updated [data/manifests/datasets.json](../data/manifests/datasets.json) and [data/manifests/dataset_ready_map.json](../data/manifests/dataset_ready_map.json) with actual annotations, counts, and task-specific blockers.
- Updated the provenance test to distinguish publisher-reported split totals from measured local folder counts.

### Audit findings

- GrainDet-Rice: 24,767 PNGs in eight coded class folders. Filename-derived source groups overlap train/val by 1,831, train/test by 3,286, and val/test by 927. All GrainDet basenames also occur in GrainSet. No explicit sprouted/weevilled or non-rice class is available.
- GrainSet Rice: 30,962 images and 30,962 matching masks, with no missing/orphan mask basenames. Filename-derived train/test groups overlap by 2,473. All GrainDet basenames recur here, the separate `rice.xml` is absent locally, and the recorded license conflict remains unresolved.
- Murat Koklu: 75,000 images across five variety folders; labels are varieties, not defect or rice/no-rice targets.
- Roboflow Rice Grain Defects: 2,000 images and 2,000 valid polygon label files, 500 polygons per numeric class. The export does not name classes or provide verified physical source groups.
- Roboflow rice quality parameters: 224 images and 224 matching box-label files, split 180/22/22. The test split is small, original source groups are unavailable, and `foreign_object` does not establish the project's foreign-matter taxonomy.
- The IIT Indore paper candidate is not downloaded; its prior inventory entry had incorrectly conflated it with the Murat Koklu archive.

### Task decisions and training status

- Rice gate: blocked because no labeled non-rice images are present.
- Bulk-scene segmentation: blocked by split-group overlap and the single-kernel-to-bulk-scene mismatch.
- Foreign matter and chalky: blocked by taxonomy/input-contract and source-group evidence gaps.
- Sprouted/weevilled: blocked because no verified positive class exists; pest-attacked is not equivalent to weevilled.
- No approved task manifests were created and no model training was started. Training would currently produce unsupported claims.

### Verification

- `python -m pytest tests/test_audit_metadata.py -q` -> 5 passed.
- Local PowerShell audit parsed Roboflow annotations, paired all GrainSet masks, measured filename-group intersections, and found no exact SHA-256 duplicate groups within either Roboflow export.

### Next steps

- Obtain authoritative class-code definitions for GrainDet/GrainSet and resolve GrainSet licensing.
- Obtain original source/sample group IDs or re-collect data with traceable independent groups; then rebuild source-group-disjoint splits.
- For rice gate, acquire labeled non-rice images. For other tasks, acquire target-compatible labels/masks and sufficient independent test groups.
- Create approved manifests and train only after those gates pass.

### Final verification addendum

- `python -m pytest -q` -> 63 passed, 1 Starlette deprecation warning in 50.16s.
- Parsed `datasets.json`, `dataset_ready_map.json`, and `dataset_audit_2026-09-27.json`; all are valid JSON. The dataset registry has 8 records with 8 unique IDs.
- `git diff --check` emitted no whitespace errors; Git reported only a line-ending conversion warning for `ml/pipeline.py`.

## 2026-09-27 — Rice-gate negative candidate research

### Findings

- Checked the additional `Rice_grain_quality_detection` folder from the active editor context. Its `Real_time_dataset` contains 10 additional rice images, not non-rice negatives or annotations.
- Public dataset search identified [Wheat grain v1](https://universe.roboflow.com/yusuf-curum-qkv4e/wheat-grain-1lgcb/dataset/1), a classification dataset with 5,155 images, classes `foreign_particles`, `grain-diseased`, and `grain-healthy`, CC BY 4.0, and published 70/20/10 split counts (3,608/1,031/516).
- Recorded it as an external, not-downloaded candidate in [data/manifests/datasets.json](../data/manifests/datasets.json), the readiness map, and the audit report; added the download destination to README. It is not approved because local archive contents, class balance, and physical source groups are unverified.
- Confirmed `training/train_rice_gate.py` refuses synthetic training and raises `NotImplementedError` after a real manifest is present. A real-image trainer/evaluator must be implemented before training can occur.
- No approved manifest was created and no model was trained.

### Verification

- `python -m pytest tests/test_audit_metadata.py -q` -> 5 passed.

### Next steps

- Download the wheat dataset archive from its public version page into `data/raw/roboflow_wheat_grain_gate_negatives_v1/`.
- Audit actual image labels, capture groups, and class counts locally; do not assume the public split is source-group-safe.
- Obtain source-group-safe rice-positive images and implement the real-image gate trainer/evaluator before preparing the approved manifest.
