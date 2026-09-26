# Machine Learning Model Training & Evaluation Guide

This document describes the training pipelines, datasets, loss functions, architectures, and evaluation metrics for each model.

---

## 1. Chalky Grain Classifier (Logistic Regression)

- **Script**: `training/train_chalky.py`
- **Output Artifacts**:
  - `models/chalky/model.joblib`
  - `models/chalky/scaler.joblib`
  - `models/chalky/feature_schema.json`
  - `models/chalky/metadata.json`
- **Algorithm**: `sklearn.linear_model.LogisticRegression(class_weight='balanced')`
- **Input**: 10 engineered color & GLCM texture features.
- **Data**: `data/processed/chalky/manifest.csv` with `image_path,mask_path,label,source_group`; positives must be genuinely chalky grains. Clean negatives must include translucent and naturally bright grains, multiple varieties/appearances, and different backgrounds. `source_group` must identify the source image or acquisition batch so related crops cannot cross splits.
- **Preprocessing**: Training calls the same mask-only LAB/GLCM feature extractor as inference.
- **Evaluation**: Source-group-disjoint train/validation/test splits; reports precision, recall, F1, ROC-AUC, confusion matrix, clean-rice false-positive rate, and Brier score. The validation split selects a decision threshold. Raw Logistic Regression probabilities remain uncalibrated.
- **Current status**: The checked-in model was fit to synthetic feature vectors and is disabled. No real chalky/clean training corpus is checked in, so real-grain performance is not established.

To train:
```bash
python training/train_chalky.py
```

---

## 2. Damaged Grain Classifier (Experimental VGG-19 Checkpoint)

- **Script**: `training/train_damaged.py`
- **Output Artifacts**:
  - `models/damaged/model.pth`
  - `models/damaged/class_mapping.json`
  - `models/damaged/metadata.json`
- **Architecture**: VGG-19 checkpoint with classes `normal` and `damaged`; it has no distinct slightly-damaged class.
- **Checkpoint provenance**: Synthetic generated illustrations only: 80 train crops and 30 test crops, no validation set. Synthetic test metrics were removed from active metadata.
- **Status**: Experimental and uncalibrated. `training/train_damaged.py` refuses to overwrite artifacts with synthetic results. A reviewed real-data trainer and labeled rice dataset are not available.
- **Inference preprocessing**: Instance-mask crop, black exterior fill, aspect-preserving 224 square, and ImageNet normalization. The softmax score is not calibrated.

Training is intentionally blocked: the real labeled dataset and a reviewed real-data trainer are not present. Do not use the legacy synthetic crop generator to produce published metrics.

---

## 3. Sprouted / Weevilled Classifier (ResNet-18 Transfer Learning)

- **Script**: `training/train_sprouted.py`
- **Output Artifacts**:
  - `models/sprouted_weevilled/model.pth`
  - `models/sprouted_weevilled/class_mapping.json`
  - `models/sprouted_weevilled/metadata.json`
- **Architecture**: ResNet-18 pretrained backbone fine-tuned for germinated shoot protrusions and weevil bored cavities.
- **Data**: `data/processed/sprouted_weevilled/manifest.csv` with `image_path,mask_path,label,source_group`; include both clean grains and known sprouted/weevilled grains. Split by source image/acquisition group.
- **Preprocessing**: Shared instance-mask crop, black exterior fill, aspect-preserving pad to 224px, ImageNet normalization in training and inference.
- **Evaluation**: Group-disjoint train/validation/test; reports precision, recall, F1, ROC-AUC, confusion matrix, clean-rice false-positive rate, and a validation-derived F1 threshold. Softmax scores are not calibrated.
- **Current status**: The checked-in checkpoint was trained on synthetic demonstration crops. Its predictions are experimental and do not establish performance on real rice grains.

To train:
```bash
python training/train_sprouted.py
```

---

## 4. Foreign Matter Detection (YOLO11n)

- **Script**: `training/train_foreign_matter.py`
- **Output Artifacts**:
  - `models/foreign_matter/class_mapping.json`
  - `models/foreign_matter/dataset.yaml`
  - `models/foreign_matter/metadata.json`
  - `data/manifests/foreign_matter_manifest.json`
- **Classes**: `0: stone`, `1: inorganic`, `2: organic`, `3: other_foreign_matter`.

To configure / train:
```bash
python training/train_foreign_matter.py
```

---

## 5. Instance Segmentation (Mask R-CNN)

- **Script**: `training/train_segmentation.py`
- **Output Artifacts**:
  - `models/segmentation/class_mapping.json`
  - `models/segmentation/metadata.json`
  - `data/manifests/segmentation_manifest.json`
- **Fallback**: Classical CV watershed segmentation pipeline with distance transform and morphology runs when Mask R-CNN weights are not loaded.

To configure:
```bash
python training/train_segmentation.py
```

---

## 6. Unified Model Evaluation

To generate a consolidated evaluation report across all models:
```bash
python training/evaluate_models.py
```
Output is saved to `models/evaluation_report.json`.
