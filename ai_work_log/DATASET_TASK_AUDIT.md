# Dataset Task Audit

## Date
2026-09-27

## Scope
This audit continues the project from the verified gate fix and the staged dataset inventory. The goal is to map each downloaded dataset to the most relevant model task and record whether it is ready for manifest approval.

## Evidence reviewed
- `data/raw/roboflow_rice_grain_defects_v1/Rice Grain Defects.v1i.yolov11/data.yaml`
- `data/raw/roboflow_rice_quality_parameters_v1/rice-grain-quality-detection.v1-rice-quality-parameter-dataset.yolov11/data.yaml`
- `data/raw/graindet_rice_v2/rice/train/`
- `data/raw/grainset_rice_v3/rice (1)/`
- `data/manifests/dataset_ready_map.json`

## Findings

### 1) Roboflow Rice Grain Defects v1
- Path: `data/raw/roboflow_rice_grain_defects_v1/Rice Grain Defects.v1i.yolov11/`
- Export format: YOLOv11 object detection
- Split directories: `train`, `valid`, `test`
- Class names in `data.yaml`: `['0', '1', '2', '3']`
- The project card says 2000 images and the export is `Resize to 256x256 (Stretch)`.
- Status: relevant for `foreign_matter` and `chalky` review, but not approved for training because the numeric class IDs are not mapped to a reviewed defect taxonomy and there is no confirmed source-group safety review.

### 2) Roboflow Rice Quality Parameters v1
- Path: `data/raw/roboflow_rice_quality_parameters_v1/rice-grain-quality-detection.v1-rice-quality-parameter-dataset.yolov11/`
- Export format: YOLOv11 object detection
- Split directories: `train`, `valid`, `test`
- Class names in `data.yaml`: `['broken_rice', 'chalky_rice', 'foreign_object', 'head_rice', 'unhulled_rice']`
- The project card says 224 images and the export uses resize to 640x640.
- Status: useful for exploratory mapping and class review, but still not approved for production training because the split is small and source/sample grouping is unclear.

### 3) GrainDet Rice v2
- Path: `data/raw/graindet_rice_v2/rice/`
- Directory structure: `train/val` plus class folders under `train`.
- Observed labels: `0_NOR`, `1_F&S`, `2_SD`, `3_MY`, `4_AP`, `5_BN`, `6_UN`, `7_IM`
- This is a grain-level taxonomy dataset and is relevant to segmentation and sprouted/weevilled style defects.
- Status: candidate for segmentation and damage/sprouted review, but not ready for manifest approval until raw label mapping and source grouping are audited.

### 4) GrainSet Rice v3
- Path: `data/raw/grainset_rice_v3/rice (1)/`
- Raw image naming pattern indicates many single-grain or controlled-acquisition rice images from repeated time-stamped captures.
- This dataset is a strong candidate for rice/no-rice and segmentation-related work, but it is still not approved for final training because the exact class mapping and license usage terms remain unresolved.
- Status: candidate only; not ready.

## Task readiness decision

### Rice gate
- Candidates: `murat_koklu_rice_image_dataset_v1`, `grainset_rice_v3`
- Decision: not ready
- Reason: binary image-level labels and source-group metadata are not yet reviewed or included in an approved manifest.

### Segmentation
- Candidates: `graindet_rice_v2`, `grainset_rice_v3`
- Decision: not ready
- Reason: instance annotation provenance and source-group safety must be cleared before any training manifest is created.

### Foreign matter / defect mapping
- Candidates: `roboflow_rice_grain_defects_v1`, `roboflow_rice_quality_parameters_v1`
- Decision: not ready
- Reason: class IDs and sample group integrity remain under review.

### Chalky
- Candidates: `roboflow_rice_grain_defects_v1`, `roboflow_rice_quality_parameters_v1`, `murat_koklu_rice_image_dataset_v1`
- Decision: not ready
- Reason: no reviewed chalky-labeled manifest exists yet; variety datasets do not count as direct chalky ground truth without manual review.

### Sprouted / weevilled
- Candidates: `graindet_rice_v2`, `grainset_rice_v3`
- Decision: not ready
- Reason: class taxonomy and source-safe splitting have not been validated.

## Conclusion
The current evidence supports a disciplined next step: source-group-safe audit, class-map validation, and only then manifest creation. No task-specific model training is approved yet.
