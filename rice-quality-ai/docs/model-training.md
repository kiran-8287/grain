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
- **Evaluation**: Evaluated on independent test split with Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.

To train:
```bash
python training/train_chalky.py
```

---

## 2. Damaged Grain Classifier (VGG-19 Transfer Learning)

- **Script**: `training/train_damaged.py`
- **Output Artifacts**:
  - `models/damaged/model.pth`
  - `models/damaged/class_mapping.json`
  - `models/damaged/metadata.json`
- **Architecture**: Pretrained VGG-19 with frozen convolutional feature extractor and fine-tuned fully connected classifier head (`Dropout(0.5) -> Linear(4096, 2)`).
- **Aspect Ratio Safeguard**: Crops are padded to square ($224 \times 224$) via `pad_to_square` (never stretched).
- **Optimization**: Adam optimizer ($lr=10^{-3}$) with CrossEntropyLoss.

To train:
```bash
python training/train_damaged.py
```

---

## 3. Sprouted / Weevilled Classifier (ResNet-18 Transfer Learning)

- **Script**: `training/train_sprouted.py`
- **Output Artifacts**:
  - `models/sprouted_weevilled/model.pth`
  - `models/sprouted_weevilled/class_mapping.json`
  - `models/sprouted_weevilled/metadata.json`
- **Architecture**: ResNet-18 pretrained backbone fine-tuned for germinated shoot protrusions and weevil bored cavities.
- **Limitation Note**: Data is limited/cross-domain; lower confidence is explicitly communicated in the UI.

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
